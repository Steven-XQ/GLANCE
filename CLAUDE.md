# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diff-IP2D is a diffusion-based model for jointly forecasting future hand trajectories and object affordances from 2D egocentric videos (IROS 2025). It uses a denoising diffusion probabilistic model (DDPM) with an iterative non-autoregressive (iter-NAR) paradigm, providing bidirectional temporal constraints. Dataset: EPIC-Kitchens 100.

## Commands

All training/evaluation uses PyTorch distributed launch (even single-GPU):

```bash
# Train (30 epochs, batch_size=32, EK100)
bash train.sh

# Evaluate trajectory prediction (ADE/FDE metrics)
bash val_traj.sh

# Evaluate affordance prediction
bash val_affordance.sh
```

The `.sh` scripts wrap `torch.distributed.launch` calls to `run_train.py` / `run_val_traj.py` / `run_val_affordance.py`, which in turn invoke `traineval.py` with different flags.

For SLURM cluster submission, use `submit_train.sh` and `submit_eval.sh`. These activate the `diffip` conda environment.

### Environment Setup

```bash
conda create -n diffip python=3.8 pip
conda activate diffip
pip install -r requirements.txt
```

## Architecture

### Data Flow

1. **Data loading** (`datasets/holoaders.py`): LMDB features + pickle labels → `EpicHODataset` with SIFT-based homography alignment between frames
2. **HOI Encoder** (`networks/transformer.py`): `ObjectTransformerModel` - spatial-temporal transformer processes multi-frame RGB features (1024-D)
3. **Pre-encoding** (`diffip2d/pre_encoder.py`): `SideFusionEncoder` fuses global/hand/object features (3×512 → 512-D); `MotionEncoder` encodes homography matrices
4. **Diffusion denoising** (`diffip2d/gaussian_diffusion.py` + `diffip2d/transformer_model.py`): `HOIDiffusion` runs iterative denoising through `MADT` (Motion-Aware Denoising Transformer) over 1000 timesteps
5. **Post-decoding** (`diffip2d/post_decoder.py`): `TrajDecoder` maps 512-D → 2D hand coordinates
6. **Affordance/Trajectory heads** (`networks/traj_decoder.py`, `networks/affordance_decoder.py`): CVAEs for trajectory and object contact point prediction

### Key Modules

- **`traineval.py`**: Main entry point. Builds all models, runs `TrainValLoop`
- **`basic_utils.py`**: `create_network_and_diffusion()` assembles diffusion components
- **`netscripts/epoch_feat.py`**: `TrainValLoop` class - core training/validation loop with loss computation, distributed training coordination
- **`netscripts/get_optimizer.py`**: Per-module parameter groups (sf_encoder, model_denoise, traj_decoder, motion_encoder, model_hoi, obj_head) with warmup + cosine annealing
- **`options/netsopts.py`**: Network hyperparameters (hidden_dim=512, diffusion_steps=1000, noise_schedule="sqrt", loss weights lambda_*)
- **`options/expopts.py`**: Experiment config (ek_version, seq_len_obs=10, seq_len_unobs=4, sample_times=10, fast_test flag, paths)

### Loss Function

Weighted combination: `lambda_traj * traj_loss + lambda_obj * obj_loss + lambda_traj_kl * traj_kl + lambda_obj_kl * obj_kl` plus diffusion denoising loss.

### Evaluation

- **Trajectory**: ADE (Average Displacement Error) and FDE (Final Displacement Error) in `evaluation/traj_eval.py`
- **Affordance**: Contact point accuracy with farthest sampling and Gaussian heatmaps in `evaluation/affordance_eval.py`
- Inference generates multiple stochastic samples (default 10) for prediction clusters

## Important Details

- Distributed training uses `LOCAL_RANK` env variable extensively; all entry points expect it
- Homography matrices are computed via SIFT on first run, then cached to `data/homos_train/` and `data/homos_test/`
- Features are stored in LMDB format (train split across part1/part2)
- Checkpoints saved to `diffip_weights/`; predictions collected to `collected_pred_traj/` and `collected_pred_aff/`
- Set `fast_test=True` in options for faster inference with minimal accuracy loss
- `preprocess/ho_types.py` defines core data types (HandSide, HandState, BBox, etc.) with protobuf serialization

## EGTEA-Gaze+ Adaptation

The model has been adapted for the EGTEA-Gaze+ dataset with gaze integration. Use `--ek_version=egtea` for all EGTEA experiments.

### EGTEA-specific settings
- `--num_classes=106`, `--seq_len_obs=10`, `--seq_len_unobs=3`
- Gaze data: `.txt` files per video, loaded from path configured in `datasets/holoaders.py:108`
- Test set: 442 samples for trajectory, 69 for affordance

### Deterministic Training
Training is made fully deterministic via settings in `traineval.py`:
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `evaluation/affordance_eval.py`: deterministic farthest_sampling (start_idx=0) and fixed-seed AUC jitter
- `datasets/holoaders.py`: `worker_init_fn` seeds each DataLoader worker

Results are reproducible across runs on the same GPU. Different GPU architectures may give different numbers.

### Gaze Integration Flags (`options/netsopts.py`)

| Flag | Description |
|------|-------------|
| `--use_gaze` | Enable gaze (dual-stream: heatmap CNN + coord MLP in GazeEncoder, GazeSideFusionEncoder gating, MADT cross-attention with LayerScale) |
| `--gaze_coord_only` | Use only coordinate MLP, skip heatmap CNN |
| `--gaze_heatmap_only` | Use only heatmap CNN, skip coordinate MLP (handled in epoch_feat.py, not model config) |
| `--gaze_fusion_only` | Gaze only in SideFusionEncoder gating, no MADT cross-attention |
| `--gaze_alpha_clamp=X` | Clamp LayerScale gaze_alpha max to X (0=no clamp) |
| `--gaze_last_n_blocks=N` | Gaze cross-attention only in last N of 6 MADT blocks (0=all) |
| `--gaze_detach_diffusion` | Detach gaze features before passing to diffusion/MADT |
| `--gaze_fixed_delta=N` | Hard mask: hand[t] attends only to gaze[max(0,t-N)] |
| `--gaze_bias_init_delta=N` | Initialize learnable temporal bias with Gaussian bump at offset N |
| `--gaze_bias_init_amp=X` | Amplitude of the Gaussian bump init (default 2.0) |
| `--gaze_before_motion` | Apply gaze cross-attention BEFORE egomotion cross-attention in DecoderBlock |
| `--gaze_cfg_dropout=X` | Zero entire gaze stream for X fraction of training batches (CFG-style) |

### Gaze Architecture
- **GazeEncoder** (`diffip2d/gaze_modules.py`): CNN (32→64→128→256→pool→fc→512) + coord MLP (3→64→512), outputs summed
- **GazeSideFusionEncoder** (`diffip2d/pre_encoder.py`): Sigmoid gate on object features using gaze (only gates observation frames)
- **GazeTemporalCrossAttention** (`diffip2d/gaze_modules.py`): Multi-head cross-attention from hand queries to gaze keys/values, with learnable temporal bias (or fixed/Gaussian-init variants)
- **DecoderBlock** (`diffip2d/transformer_model.py`): Gaze added as 3rd residual stream with zero-init LayerScale (`gaze_alpha`). Order: self-attn → [gaze if before_motion] → motion cross-attn → [gaze if after motion] → MLP

### Generic experiment runner
`run_experiment.py` passes all CLI args to traineval.py. Used with torch.distributed.launch:
```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_experiment.py <args>
```

### Checkpoint cleanup
`netscripts/epoch_feat.py` deletes old checkpoints after each epoch save, keeping only the latest (to avoid disk quota issues).

## Gaze Experiment Results (Deterministic)

All results below are deterministic (identical across reruns on the same GPU).

### Version descriptions
| Ver | Description | Key flags |
|-----|-------------|-----------|
| Baseline | No gaze | (none) |
| v2 | Full gaze: dual-stream encoder, LayerScale, learnable temporal bias | `--use_gaze` |
| v4 | Coordinate-only (no heatmap CNN) | `--use_gaze --gaze_coord_only` |
| v6 | SideFusionEncoder gating only, no MADT cross-attention | `--use_gaze --gaze_fusion_only` |
| v7 | Clamped LayerScale alpha max=0.1 (identical to v2 — alpha never exceeds 0.1) | `--use_gaze --gaze_alpha_clamp=0.1` |
| v8 | Gaze in last 2 of 6 MADT blocks only | `--use_gaze --gaze_last_n_blocks=2` |
| v9 | Detach gaze features from diffusion gradient path | `--use_gaze --gaze_detach_diffusion` |
| v10 | Hard temporal mask: hand[t] → gaze[t-2] only | `--use_gaze --gaze_fixed_delta=2` |
| v11 | Learnable bias init with Gaussian bump at delta=2, amp=2.0 | `--use_gaze --gaze_bias_init_delta=2 --gaze_bias_init_amp=2.0` |
| v12 | Learnable bias init with Gaussian bump at delta=3, amp=0.5 | `--use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5` |
| v12a | v12 + heatmap only (no coord MLP) | v12 flags + `--gaze_heatmap_only` |
| v12b | v12 + CFG dropout 10% | v12 flags + `--gaze_cfg_dropout=0.1` |
| v12c | v12 + gaze cross-attention BEFORE egomotion | v12 flags + `--gaze_before_motion` |
| v13 | v12c + detach (= v12c, detach has no practical effect) | v12 flags + `--gaze_before_motion --gaze_detach_diffusion` |

### 30-epoch results (reference)
| Ver | WDE ↓ | FDE ↓ | SIM ↑ | AUC-J ↑ | NSS ↑ |
|-----|-------|-------|-------|---------|-------|
| Baseline | 0.407 | 0.210 | 0.208 | 0.733 | 0.858 |
| v2 | 0.408 | 0.222 | 0.226 | 0.749 | 0.963 |
| v4 | 0.398 | 0.215 | 0.219 | 0.738 | 0.919 |
| v6 | 0.415 | 0.226 | 0.224 | 0.759 | 0.975 |
| v8 | 0.418 | 0.228 | 0.203 | 0.729 | 0.807 |
| v9 | 0.397 | 0.215 | 0.220 | 0.741 | 0.928 |
| v10 | 0.412 | 0.221 | 0.218 | 0.728 | 0.902 |
| v11 | 0.401 | 0.216 | 0.223 | 0.740 | 0.961 |
| v12 | 0.405 | 0.216 | 0.224 | 0.751 | 0.958 |
| v12a | 0.408 | 0.221 | 0.212 | 0.729 | 0.866 |
| v12b | 0.415 | 0.225 | 0.225 | 0.753 | 0.951 |
| v12c | 0.407 | 0.220 | 0.222 | 0.739 | 0.952 |
| v13 | 0.404 | 0.219 | 0.225 | 0.750 | 0.959 |

### NSS across epoch counts
| Ver | 10ep | 15ep | 20ep | 25ep | 30ep | 35ep | 40ep | 45ep | 50ep | 60ep |
|-----|------|------|------|------|------|------|------|------|------|------|
| Baseline | ? | ? | 0.928 | 0.663 | 0.858 | 0.645 | 0.522 | ? | 0.925 | 0.680 |
| v2 | ? | ? | 0.571 | 0.230 | 0.963 | 0.758 | 0.467 | ? | 0.605 | 0.652 |
| v6 | ? | ? | 0.908 | 0.493 | 0.975 | 0.627 | 0.564 | ? | 0.752 | 0.630 |
| v12b | ? | ? | 0.696 | 0.376 | 0.951 | 0.936 | 0.490 | ? | 0.487 | 0.663 |

(10, 15, 45 epoch experiments pending — could not submit due to allocation limit)

### Key findings
1. **Affordance (NSS) is highly sensitive to epoch count** due to cosine LR schedule. Each --epochs=N creates a different optimization trajectory, not a continuation. NSS oscillates wildly (0.2–0.97) across epoch counts.
2. **Trajectory (WDE) is much more stable** and improves monotonically with longer training.
3. **At 30 epochs**, gaze variants significantly outperform baseline on affordance (NSS 0.95+ vs 0.858) while slightly regressing trajectory (+0.01 WDE).
4. **At other epoch counts**, baseline often matches or exceeds gaze variants on affordance.
5. **v7 = v2 exactly** — gaze_alpha (LayerScale) never exceeds 0.1, so clamping is a no-op.
6. **v13 = v12c exactly** — detaching gaze from diffusion has no effect because gaze_alpha is too small for meaningful gradient flow through MADT back to GazeEncoder.
7. **To make defensible claims about gaze benefit**: evaluate across multiple epoch counts, or use early stopping on a validation set, or evaluate at fixed intervals during training.

### Hardcoded paths to update on new machine
- `datasets/holoaders.py:75` — raw_images_base (extracted frames path)
- `datasets/holoaders.py:108` — gaze_data_base (gaze .txt files path)
- `submit_extract.sh:16` — EGTEA base dir
- `submit_train.sh:17` — project home dir
