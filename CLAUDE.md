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
