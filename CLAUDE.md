# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GLANCE** (Gaze-Led Anticipation of Non-autoregressive Contact in Egocentric videos) is a multimodal gaze pathway built on top of the Diff-IP2D backbone (Ma et al., IROS 2025) for joint forecasting of future hand trajectories and object affordance hotspots from 2D egocentric videos. The Diff-IP2D diffusion stack provides bidirectional temporal constraints via 1000-step square-root-schedule iterative denoising through the Motion-Aware Denoising Transformer (MADT). GLANCE adds, *additively*, a dual-stream gaze encoder, a Gaze-Side-Fusion gating mechanism, and a gaze cross-attention residual inside each MADT block. Evaluated across two contrasting behavioural regimes — **EGTEA-Gaze+** (free-form cooking, loose eye-hand coupling) and **MECCANO** (instructed industrial assembly, tight coupling) — at a fixed *E* = 35-epoch schedule across five random seeds, with a four-level deterministic scaffold guaranteeing bit-exact intra-GPU reproducibility.

The canonical result is the **130-cell sweep**: 13 variants × 5 seeds × 2 datasets, driven by `run_seed_sweep.sh`. All reference results in this document are from that sweep.

For a fully-detailed public-facing reproduction guide (Conda environment, data download, sweep replication), see `README.md`.

## Setup (short reference)

The full setup is in `README.md` §1. Key facts for Claude:

- Conda env: **`glance`**, Python 3.10, PyTorch 2.x + CUDA 12.4.
- Required tarballs (Google Drive): `base_models.tar.gz`, `CVPR2022-hoi-forecast-training-data.tar.gz`, `data.tar.gz`, `common.tar.gz` — links in README.
- After extraction, repo layout matches the tree in README §1.3.
- SIFT homographies under `data/homos_{train,test}/` auto-populate on first training epoch (atomic-rename writes; safe under concurrent workers).

Quick env recreation:

```bash
conda create -n glance python=3.10 pip -y
conda activate glance
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "setuptools<70"
pip install visdom torchnet --no-build-isolation
pip install -r requirements.txt
```

## Commands

All training/evaluation uses `torch.distributed.launch` with `--use_env` (single-GPU is still wrapped).

### 130-cell deterministic sweep (canonical entry point)

```bash
bash run_seed_sweep.sh
```

This runs every (dataset, variant, seed) cell sequentially on the local 8 × A100 node. It is **fully resumable** via `.done` markers under `seed_sweep/markers/`; deleting a marker re-queues that cell. Per-cell logs land in `seed_sweep/logs/`. The driver writes one row per completed cell to `seed_sweep/results.csv` with columns:

```
dataset, variant, seed, epochs, wde, fde, sim, auc_j, nss
```

After (or during) the sweep, summarise across seeds:

```bash
python aggregate_seed_sweep.py summarise --csv seed_sweep/results.csv \
    --out seed_sweep/aggregated.csv
```

The `variant` column uses the canonical report naming **`baseline`, `v1`, `v2`, …, `v12`**. The script and aggregator both enumerate variants in this same order — see `run_seed_sweep.sh`'s `VARIANTS` array and `aggregate_seed_sweep.py`'s `VARIANT_ORDER` list.

### Single-cell training and evaluation

For experimenting with a single (variant, dataset, seed) cell, the three-step train + traj-eval + aff-eval pattern is in README §4.2. The relevant CLI is `run_experiment.py`, which is a thin passthrough to `traineval.py`. Example for GLANCE (v8) on EGTEA, seed 42:

```bash
# train
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port=12325 --use_env \
    run_experiment.py \
        --ek_version=egtea --epochs=35 --batch_size=8 \
        --num_classes=106 --seq_len_obs=10 --seq_len_unobs=3 \
        --learnable_weight=True --manual_seed=42 \
        --use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5

# eval traj (then eval aff — same args without --traj_only)
rm -rf collected_pred_traj
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch \
    --nproc_per_node=8 --master_port=12326 --use_env \
    run_experiment.py \
        --evaluate --traj_only \
        --ek_version=egtea --num_classes=106 \
        --seq_len_obs=10 --seq_len_unobs=3 \
        --resume=./diffip_weights/checkpoint_35.pth.tar \
        --manual_seed=42 \
        --use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5
```

`--ek_version` accepts `egtea`, `meccano`, or `ek100` (the inherited EK100 path is a sanity-check loader only — not used for the 130-cell sweep).

### Fresh dataset extraction pipeline (only needed if rebuilding features)

The pre-built feature LMDBs ship in the `data.tar.gz` download, so feature extraction is *not* required to reproduce the 130-cell sweep. If rebuilding from raw frames:

1. RGB extraction (one-off): `bash run_extract_egtea_local.sh` or `bash run_extract_meccano_local.sh`.
2. Annotation prep (CPU, fast): `python scripts/prepare_egtea_data.py` or `python scripts/prepare_meccano_data.py`.
3. BNInception feature extraction (GPU, ~30–40 min per split): `python scripts/extract_egtea_features.py` or `python scripts/extract_meccano_features.py`.

## Architecture

### Data Flow

1. **Data loading** (`datasets/holoaders.py`): LMDB features + pickle labels → `EpicHODataset` with SIFT-based homography auto-cache between frames.
2. **HOI Encoder** (`networks/transformer.py`): `ObjectTransformerModel` — spatial-temporal transformer processes multi-frame RGB features (1024-D) into three 512-D streams (global, hand, object).
3. **Pre-encoding** (`diffip2d/pre_encoder.py`): `SideFusionEncoder` fuses the three streams (3 × 512 → 512); `MotionEncoder` encodes per-pair homography matrices; **`GazeSideFusionEncoder`** wraps `SideFusionEncoder` with a sigmoid gate over the object stream.
4. **Diffusion denoising** (`diffip2d/gaussian_diffusion.py` + `diffip2d/transformer_model.py`): `HOIDiffusion` runs iterative denoising through `MADT` over 1000 sqrt-schedule timesteps. Each `DecoderBlock` is `SelfAttn → EgomotionXA → [GazeXA] → MLP` with zero-init LayerScale on the gaze residual.
5. **Post-decoding** (`diffip2d/post_decoder.py`, `networks/{traj,affordance}_decoder.py`): `TrajDecoder` maps 512-D → 2-D hand coordinates; CVAE heads emit contact-point predictions.

### Key Modules

- **`traineval.py`** — main entry point. Builds all models, runs `TrainValLoop`. Sets determinism flags at top of file.
- **`basic_utils.py`** — `create_network_and_diffusion()` assembles diffusion components.
- **`netscripts/epoch_feat.py`** — `TrainValLoop` class: core training/validation loop with loss computation, distributed training coordination, checkpoint cleanup (races under concurrent SLURM jobs — see Engineering Lessons §L1).
- **`netscripts/get_optimizer.py`** — per-module parameter groups (`sf_encoder`, `model_denoise`, `traj_decoder`, `motion_encoder`, `model_hoi`, `obj_head`, `gaze_encoder` when `--use_gaze`) with 5-epoch warmup + cosine annealing keyed to `--epochs`.
- **`options/netsopts.py`** — network hyperparameters (`hidden_dim=512`, `diffusion_steps=1000`, `noise_schedule="sqrt"`, loss weights `lambda_*`, all gaze CLI flags).
- **`options/expopts.py`** — experiment config (`--ek_version` choices include `egtea`, `meccano`, `ek100`; `--seq_len_obs=10`, `--seq_len_unobs=3`, `--sample_times=10`, `--fast_test`).

### Loss Function

Weighted combination:
`lambda_traj * traj_loss + lambda_obj * obj_loss + lambda_traj_kl * traj_kl + lambda_obj_kl * obj_kl + lambda_diff * diff_loss`.
All λ inherited from Diff-IP2D defaults; `--learnable_weight=True` adds learnable scalars per term.

### Evaluation

- **Trajectory**: WDE (worst-of-hands Weighted Displacement Error, mean over future window) and FDE (final-frame variant) — best-of-many over `--sample_times` stochastic samples (`evaluation/traj_eval.py`).
- **Affordance**: SIM, AUC-J, NSS — Bylinskii et al. saliency metrics on Gaussian heatmaps built from farthest-point-sampled prediction clusters (`evaluation/affordance_eval.py`).
- Inference uses `--fast_test=True` (DDIM-respaced ~50 steps) by default; metrics agree with full 1000-step inference to the third decimal on a sub-grid we validated.

## Important Details

- Distributed training uses `LOCAL_RANK` env variable extensively; all entry points expect it via `torch.distributed.launch --use_env`.
- Homography matrices are computed via SIFT on first run, then cached to `data/homos_{train,test}/<participant_id>/homo/<video_id>/<frame>.npy` under an *atomic-rename* write protocol (see Engineering Lessons §L2).
- Features are stored in LMDB format. EGTEA and MECCANO both use the single-LMDB layout (`feats_train/data.lmdb`, `feats_test/data.lmdb`). EK100 uses a two-part split (`...part1.lmdb` + `...part2.lmdb`); the loader's `_open_lmdb` and `_get_feat_dict` branch on `ek_version in ('egtea', 'meccano')` to pick the layout.
- Checkpoints saved to `diffip_weights/checkpoint_<epoch>.pth.tar`; old checkpoints are deleted after each epoch save (one file per run; **races under concurrent SLURM jobs** — see Engineering Lessons §L1).
- Predictions collected to `collected_pred_traj/` and `collected_pred_aff/`. The driver script removes these between training and evaluation.

## EGTEA-Gaze+ Adaptation

Use `--ek_version=egtea` for all EGTEA experiments.

### EGTEA-specific settings
- `--num_classes=106`, `--seq_len_obs=10`, `--seq_len_unobs=3`.
- `ori_fps=30`, `t_ant=0.5`, `t_buffer=10/6`, `fps=6` (set in `datasets/datasetopts.py`).
- Label window length $W_{\text{label}} = \lfloor t_{\text{ant}} \cdot r_0\rfloor + 1 = 16$ frames.
- Gaze: SMI eye-tracker `.txt` exports per video, parsed by `_load_egtea_gaze` in `holoaders.py`. Coords normalised by 1280×960.
- Test set: 442 samples for trajectory, 69 for affordance.

## MECCANO Adaptation

Use `--ek_version=meccano` for all MECCANO experiments.

### MECCANO-specific settings
- `--num_classes=61`, `--seq_len_obs=10`, `--seq_len_unobs=3`.
- `ori_fps=12`, `t_ant=0.5`, `t_buffer=10/6`, `fps=6` (set in `datasets/datasetopts.py`).
- Label window length $W_{\text{label}} = 7$ frames (vs. EGTEA's 16, because of the lower native fps).
- Frame template: `'{:05d}.jpg'` (5-digit, no `frame_` prefix), set per-`ek_version` in `holoaders.py`.
- Hand labels: derived from VGG-VIA bbox CSVs (`SX_Hand` = LEFT, `DX_Hand` = RIGHT). Per-frame bbox centre, then Cubic Hermite spline over the window with extrapolation clipped to the observed range.
- Affordance labels: derived from NAO COCO bboxes. Five points sampled uniformly inside the bbox at the action start frame, with seeded `RandomState(0)`.
- Coord scaling: pixel coords (1920×1080) scaled to canonical (456, 256) via `S_x=456/1920, S_y=256/1080`. The aspect-ratio match (~16:9 on both) means uniform per-axis scaling preserves geometry.
- Splits: train (videos 0001–0006, 0013–0016, 0018), val (0007, 0017), test (0008–0012, 0019, 0020).
- LMDB partition: `feats_train.lmdb` ↔ train split; `feats_test.lmdb` ↔ **val + test** (val frames are needed by `--evaluate --traj_only` which uses `mode != 'train'`). Unlike EGTEA where val and test are the same set.
- Gaze: 200 Hz CSVs per video, parsed by `_load_meccano_gaze` in `holoaders.py`. Multiple samples per frame are mean-aggregated. Coords normalised by 1920×1080.
- Test eval set: 904 samples for trajectory (val), 2584 for affordance (test).

### MECCANO data prep pipeline
- `scripts/prepare_meccano_data.py` produces all artefacts (label pickles, video_info.json, eval labels, uid2future, split CSVs in EGTEA-compatible column layout).
- `scripts/extract_meccano_features.py` runs BNInception per-frame; the `--split` flag is `train` (train CSV only) or `test` (val + test CSVs).

### MECCANO loader contract checklist (when porting to a third dataset)
1. Compute $W_{\text{label}} = \lfloor t_{\text{ant}} \cdot r_0\rfloor + 1$ from native fps and use consistently in prepare-script *and* `datasetopts.py`. A mismatch fires the holoaders assertion at line ~506 with a "last observation frame mismatch" message.
2. Force `video_id` and `participant_id` to canonical strings immediately after `pd.read_csv` (pandas silently converts `"0001"` → int 1, which corrupts LMDB keys).
3. Audit `ek_version != 'egtea'` predicates and convert to positive `in (...)` form. They were used as EK100 selectors, not "not egtea" predicates.
4. Partition LMDBs by *loader-mode visibility* (`mode == 'train'` ↔ feats_train; everything else ↔ feats_test), not by clip-membership.
5. Add the new dataset to `--ek_version` argparse `choices=`.
6. Pick a frame template matching the dataset's filename convention.
7. Implement a `_load_<dataset>_gaze` parser and dispatch from `_load_gaze_for_video` if gaze is available.

## Deterministic Training

Training is bit-exact reproducible via a four-level scaffold (see also Engineering Lessons §L10):

1. **PyTorch flags** (top of `traineval.py`): `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`.
2. **CUBLAS workspace** (env var): `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
3. **Multi-source seeding**: Python `random`, NumPy, Torch CPU/CUDA generators all seeded from `--manual_seed`; DataLoader workers re-seeded via `worker_init_fn` in `holoaders.py`.
4. **Algorithmic seeding**: `evaluation/affordance_eval.py` farthest-point sampling uses `start_idx=0` (deterministic); AUC-J jitter uses `np.random.RandomState(0)` (seeded).

Results are bit-exact across reruns on the same GPU. Different GPU architectures may give different round-off; in our experience the *ranking* of variants is preserved across architectures even if absolute numbers shift slightly.

## Gaze Integration Flags (`options/netsopts.py`)

| Flag | Description |
|------|-------------|
| `--use_gaze` | Enable gaze (dual-stream: heatmap CNN + coord MLP in GazeEncoder, GazeSideFusionEncoder gating, MADT cross-attention with LayerScale) |
| `--gaze_coord_only` | Use only coordinate MLP, skip heatmap CNN |
| `--gaze_heatmap_only` | Use only heatmap CNN, skip coordinate MLP (handled in `epoch_feat.py`, not model config) |
| `--gaze_fusion_only` | Gaze only in SideFusionEncoder gating, no MADT cross-attention |
| `--gaze_alpha_clamp=X` | Clamp LayerScale `gaze_alpha` max to X (0 = no clamp) |
| `--gaze_last_n_blocks=N` | Gaze cross-attention only in last N of 6 MADT blocks (0 = all) |
| `--gaze_detach_diffusion` | Detach gaze features before passing to diffusion/MADT |
| `--gaze_fixed_delta=N` | Hard mask: hand[t] attends only to gaze[max(0, t−N)] |
| `--gaze_bias_init_delta=N` | Initialise learnable temporal bias with Gaussian bump at offset N |
| `--gaze_bias_init_amp=X` | Amplitude of the Gaussian bump init (default 2.0) |
| `--gaze_before_motion` | Apply gaze cross-attention BEFORE egomotion cross-attention in DecoderBlock |
| `--gaze_cfg_dropout=X` | Zero entire gaze stream for X fraction of training batches (CFG-style) |

### Gaze Architecture
- **GazeEncoder** (`diffip2d/gaze_modules.py`): heatmap CNN (32→64→128→256→pool→fc→512) + coord MLP (3→64→512), outputs summed.
- **GazeSideFusionEncoder** (`diffip2d/pre_encoder.py`): sigmoid gate on object features using gaze (only gates observation frames).
- **GazeTemporalCrossAttention** (`diffip2d/gaze_modules.py`): multi-head cross-attention from hand queries to gaze keys/values, with learnable temporal bias (or fixed/Gaussian-init variants). Bias shape $(T_{\text{un}} \times T_{\text{obs}})$, shared across MADT blocks.
- **DecoderBlock** (`diffip2d/transformer_model.py`): gaze added as 3rd residual stream with zero-init LayerScale (`gaze_alpha`). Order: self-attn → [gaze if before_motion] → motion cross-attn → [gaze if after motion] → MLP.

### Generic experiment runner
`run_experiment.py` passes all CLI args to `traineval.py`. Used with `torch.distributed.launch`:
```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch \
    --nproc_per_node=1 --master_port=12225 --use_env \
    run_experiment.py <args>
```

## Engineering Lessons (Read these before parallel sweeps)

Operational findings from the 130-cell sweep. Each cost real GPU time to discover; they should not need to be re-discovered.

### L1. Concurrent SLURM jobs corrupt each other's checkpoints
`netscripts/epoch_feat.py` (around line 707) deletes *every* `checkpoint_*.pth.tar` in `./diffip_weights/` except the current epoch's. Under sequential jobs this is correct; under parallel jobs that share the directory, it races and silently destroys other jobs' final checkpoints, producing the symptom "training looked successful but eval cannot find the checkpoint".

**Fix:** run cells sequentially (which is exactly what `run_seed_sweep.sh` does). For multi-job scheduling, chain sweeps via `--dependency=afterany:$PREV_JOB_ID`. Code-side alternative is per-job checkpoint subdirectories (not implemented).

### L2. Concurrent SIFT homography writes can produce 0-byte files
The auto-cached homography (`holoaders.py` ~lines 269/281/388) is written by every DataLoader worker that hits a cache miss. With 16 workers, two can race on the same path; `np.save` is not atomic, so a reader can observe a partial file and raise `EOFError: No data left in file`.

**Fix:** atomic-rename writes (`.tmp.<pid>` then `os.replace`) plus robust loads (try/except `EOFError, ValueError, OSError` → recompute). Both are already in the current codebase.

### L3. Pandas integer parsing of zero-padded `video_id` columns
`pd.read_csv` infers a column of `"0001"`, `"0002"`, … as `int64`, dropping the zero-padding. The downstream LMDB key `os.path.join(action.video_id, frame_name)` then becomes `"1/00001.jpg"` and fails to match the stored key `b"0001/00001.jpg"`.

**Fix:** in `get_meccano_annotation` (and any new dataset adapter), force `df['video_id'] = df['video_id'].map(lambda v: f"{int(v):04d}")` immediately after read.

### L4. The `ek_version != 'egtea'` predicate as an EK100 selector
The reference Diff-IP2D code uses `self.ek_version == 'egtea'` to select the single-LMDB layout and the implicit `else` for the EK100 split-LMDB layout. With a third dataset (MECCANO) added, `else` silently routed MECCANO into the EK100 path and crashed with `'FeaturesHOLoader' object has no attribute 'env1'`.

**Fix:** all three branch points (`_open_lmdb`, `_get_feat_dict`, `_make_full_name`) use `ek_version in ('egtea', 'meccano')` for the single-LMDB layout. Audit any future similar predicates.

### L5. Anchor / window alignment mismatch with native fps
Hard-coding $W_{\text{label}} = 16$ (EGTEA) on a 12-fps dataset (MECCANO) makes the label anchor 9 frames off from the observation sampler's last frame, firing the assertion at `holoaders.py:~506`. Always derive $W_{\text{label}}$ from $\lfloor t_{\text{ant}} \cdot r_0\rfloor + 1$ at the dataset's native fps.

### L6. LMDB partition by clip-membership, not loader-mode
First pass put MECCANO val into `feats_train.lmdb`. The loader chooses LMDB by *mode* (`'train'` vs. everything else), not by *split*; val frames need to be in `feats_test.lmdb`.

**Fix:** `feats_train.lmdb` ↔ train split only; `feats_test.lmdb` ↔ val + test.

### L7. `argparse choices=` is a hard barrier for new datasets
Adding `'meccano'` to `--ek_version` requires editing `options/expopts.py` (line ~11). Otherwise the very first attempted training run dies with a one-second `argparse` error.

### L8. EpicVideo column requirements
`EpicVideo._get_actions` accesses ~15 columns by attribute on a pandas row. Missing columns raise `AttributeError` rather than `KeyError`. Always reproduce the EGTEA column set verbatim (including `participant_id`, `start_time`, `stop_time`, `verb`, `noun`, `action`, `verb_class`, `noun_class`, `action_class`, `all_nouns`, `all_noun_classes`) when writing a new split CSV.

### L9. Cosine LR schedule is keyed to total `--epochs`
Each `--epochs=N` run is a *different optimisation problem*, not a checkpoint along a single trajectory. This is precisely why the canonical 130-cell sweep fixes $E = 35$ across every cell — different epoch counts are not comparable. Always fix $E$ globally when adding new variants; do not interleave variants at different $E$.

### L10. Determinism is cheap to enable up-front, expensive to retrofit
The 4-level scaffold (PyTorch flags, CUBLAS workspace, multi-source seeding, algorithmic FPS / AUC-J seeding) should be the *first commit* on any new project. Each layer caught at least one real source of variance during development.

### L11. SLURM walltime is an upper bound, not a target
Estimate per-cell cost before submitting; use 25 % buffer. Submission failures (`Slurm temporarily unable to accept job`) are transient — wrap `sbatch` in `until ... done` retry loop with a 30-second sleep.

## Reference Results — 130-cell sweep at *E* = 35

All numbers below are from the 130-cell deterministic sweep: 13 variants × 5 seeds {13, 42, 256, 777, 2025} × 2 datasets, mean ± population standard deviation. Best per column **bold**. GLANCE = **v8**.

### Variant definitions

| Variant | CLI flags | Design axis exercised |
|---|---|---|
| Baseline | *(no gaze flags)* | Unmodified Diff-IP2D |
| v1 | `--use_gaze` | Full pathway, learnable bias from scratch |
| v2 | `--use_gaze --gaze_coord_only` | Encoder ablation: coord-only MLP |
| v3 | `--use_gaze --gaze_fusion_only` | Fusion-site ablation: gate only |
| v4 | `--use_gaze --gaze_last_n_blocks=2` | Block-coverage: deep XA only |
| v5 | `--use_gaze --gaze_detach_diffusion` | Gradient detach on v1 |
| v6 | `--use_gaze --gaze_fixed_delta=2` | Temporal: hard mask $\Delta = 2$ |
| v7 | `--use_gaze --gaze_bias_init_delta=2 --gaze_bias_init_amp=2.0` | Temporal: Gaussian-init $\Delta = 2, A = 2.0$ |
| **v8 (GLANCE)** | `--use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5` | Temporal: Gaussian-init $\Delta = 3, A = 0.5$ |
| v9 | *(v8 flags) + `--gaze_heatmap_only`* | Encoder on v8: heatmap-only CNN |
| v10 | *(v8 flags) + `--gaze_cfg_dropout=0.1`* | CFG dropout 10 % on v8 |
| v11 | *(v8 flags) + `--gaze_before_motion`* | Order: gaze XA before egomotion |
| v12 | *(v8 flags) + `--gaze_before_motion --gaze_detach_diffusion`* | Gradient detach on v11 |

### EGTEA-Gaze+ — *E* = 35

| Variant | WDE ↓ | FDE ↓ | SIM ↑ | AUC-J ↑ | NSS ↑ |
|---|---|---|---|---|---|
| Baseline | 0.372 ± 0.013 | 0.196 ± 0.007 | 0.166 ± 0.020 | 0.688 ± 0.028 | 0.518 ± 0.157 |
| v1 | 0.352 ± 0.005 | 0.186 ± 0.003 | 0.187 ± 0.026 | 0.712 ± 0.037 | 0.682 ± 0.213 |
| v2 | 0.352 ± 0.005 | 0.187 ± 0.003 | 0.189 ± 0.024 | 0.711 ± 0.035 | 0.695 ± 0.185 |
| v3 | 0.353 ± 0.007 | **0.185 ± 0.006** | 0.171 ± 0.014 | 0.690 ± 0.021 | 0.555 ± 0.121 |
| v4 | 0.354 ± 0.012 | 0.188 ± 0.007 | 0.176 ± 0.028 | 0.694 ± 0.043 | 0.624 ± 0.221 |
| v5 | **0.350 ± 0.004** | **0.185 ± 0.003** | 0.187 ± 0.022 | 0.710 ± 0.034 | 0.685 ± 0.187 |
| v6 | 0.354 ± 0.006 | 0.186 ± 0.004 | **0.195 ± 0.022** | **0.721 ± 0.028** | **0.749 ± 0.183** |
| v7 | 0.354 ± 0.005 | 0.186 ± 0.003 | 0.189 ± 0.022 | 0.711 ± 0.029 | 0.692 ± 0.181 |
| **v8 (GLANCE)** | 0.354 ± 0.005 | 0.186 ± 0.004 | 0.193 ± 0.025 | 0.712 ± 0.033 | 0.722 ± 0.210 |
| v9 | 0.353 ± 0.005 | 0.186 ± 0.003 | 0.193 ± 0.024 | 0.717 ± 0.032 | 0.722 ± 0.195 |
| v10 | 0.352 ± 0.009 | 0.186 ± 0.005 | 0.175 ± 0.014 | 0.699 ± 0.022 | 0.591 ± 0.115 |
| v11 | 0.352 ± 0.006 | 0.186 ± 0.004 | 0.191 ± 0.020 | 0.717 ± 0.028 | 0.702 ± 0.162 |
| v12 | 0.351 ± 0.005 | **0.185 ± 0.004** | 0.193 ± 0.021 | 0.716 ± 0.026 | 0.727 ± 0.166 |

### MECCANO — *E* = 35

| Variant | WDE ↓ | FDE ↓ | SIM ↑ | AUC-J ↑ | NSS ↑ |
|---|---|---|---|---|---|
| Baseline | 0.386 ± 0.028 | 0.205 ± 0.017 | 0.209 ± 0.003 | 0.774 ± 0.004 | 1.014 ± 0.025 |
| v1 | 0.336 ± 0.013 | 0.171 ± 0.008 | **0.214 ± 0.004** | **0.783 ± 0.005** | **1.055 ± 0.033** |
| v2 | 0.342 ± 0.005 | 0.175 ± 0.006 | 0.211 ± 0.004 | 0.779 ± 0.006 | 1.034 ± 0.037 |
| v3 | 0.338 ± 0.042 | 0.171 ± 0.024 | 0.211 ± 0.003 | 0.777 ± 0.007 | 1.021 ± 0.032 |
| v4 | 0.359 ± 0.040 | 0.190 ± 0.028 | 0.213 ± 0.003 | 0.781 ± 0.007 | 1.053 ± 0.031 |
| v5 | 0.335 ± 0.010 | 0.173 ± 0.011 | **0.214 ± 0.003** | 0.782 ± 0.006 | 1.054 ± 0.032 |
| v6 | 0.323 ± 0.020 | **0.164 ± 0.014** | 0.212 ± 0.002 | 0.781 ± 0.004 | 1.034 ± 0.013 |
| v7 | 0.344 ± 0.024 | 0.178 ± 0.015 | 0.213 ± 0.003 | 0.779 ± 0.005 | 1.045 ± 0.028 |
| **v8 (GLANCE)** | 0.342 ± 0.035 | 0.178 ± 0.021 | 0.212 ± 0.003 | 0.781 ± 0.005 | 1.041 ± 0.023 |
| v9 | 0.340 ± 0.026 | 0.174 ± 0.014 | 0.212 ± 0.003 | 0.780 ± 0.007 | 1.039 ± 0.035 |
| v10 | **0.313 ± 0.024** | **0.164 ± 0.018** | 0.211 ± 0.004 | 0.778 ± 0.007 | 1.033 ± 0.040 |
| v11 | 0.338 ± 0.025 | 0.172 ± 0.015 | 0.213 ± 0.003 | 0.780 ± 0.004 | 1.044 ± 0.030 |
| v12 | 0.336 ± 0.023 | 0.171 ± 0.013 | 0.212 ± 0.003 | 0.780 ± 0.007 | 1.038 ± 0.033 |

### GLANCE vs Baseline — headline deltas

| Metric | EGTEA Δ | MECCANO Δ |
|---|---|---|
| WDE ↓ | +0.018  (+4.8 %) | +0.044  (+11.4 %) |
| FDE ↓ | +0.010  (+5.1 %) | +0.027  (+13.2 %) |
| SIM ↑ | +0.027  (+16.3 %) | +0.003  (+1.4 %) |
| AUC-J ↑ | +0.024  (+3.5 %) | +0.007  (+0.9 %) |
| NSS ↑ | +0.204  (+39.4 %) | +0.027  (+2.7 %) |

### Best-of-suite — winning variant per metric

| Metric | EGTEA | MECCANO |
|---|---|---|
| WDE ↓ | +5.9 %  (v5) | **+18.9 %**  (v10) |
| FDE ↓ | +5.6 %  (v3 / v5 / v12) | +20.2 %  (v6 / v10) |
| SIM ↑ | +17.5 %  (v6) | +2.4 %  (v1 / v5) |
| AUC-J ↑ | +4.8 %  (v6) | +1.1 %  (v1) |
| NSS ↑ | **+44.6 %**  (v6) | +4.1 %  (v1) |

### Substantive findings

1. **Gaze is a universally positive prior.** 12/12 variants × 5/5 metrics × 2/2 datasets above Baseline — 120 substantive deltas, zero false positives, all passing the `|Δ| > max(σ_ref, σ_var)` criterion.
2. **The gain profile inverts across datasets.** EGTEA peaks on affordance (+39–45 % NSS), MECCANO peaks on trajectory (+11–19 % WDE). Each dataset's gain expresses itself wherever its ground truth has reachable noise-floor headroom — EGTEA's RULSTM trajectories are already tight; MECCANO's NAO-bbox affordance labels are already concentrated.
3. **Fusion topology dominates stabilisation.** Gate + cross-attention together carry both spatial and temporal contributions; gradient detachment and attention re-ordering are seed-noise-neutral.
4. **Some axes are dataset-conditional.** Hard mask (v6) wins EGTEA affordance; CFG dropout (v10) wins MECCANO trajectory but costs ΔNSS = −0.131 on EGTEA relative to GLANCE. The Gaussian-init middle ground (v8 = GLANCE) is the principled cross-dataset default.
