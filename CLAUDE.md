# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Diff-IP2D is a diffusion-based model for jointly forecasting future hand trajectories and object affordances from 2D egocentric videos (IROS 2025). It uses a denoising diffusion probabilistic model (DDPM) with an iterative non-autoregressive (iter-NAR) paradigm, providing bidirectional temporal constraints. Originally targeted at EPIC-Kitchens-100; this fork adds adaptations for **EGTEA-Gaze+** (with gaze integration) and **MECCANO** (with gaze integration), plus a deterministic-training scaffold for reproducible ablation sweeps.

## Migration to a New Isambard AI Project (Read this first)

The previous Isambard AI project allocation (under `u6cu`) has expired. When this repo is cloned into a new project the following items need to be re-pointed at the new project's paths. This is an *operational* checklist — none of the code changes substantively, only the absolute paths.

### 1. Conda environment and project layout

The submit scripts assume:
- The conda environment is named `diffip` and lives under `$SCRATCH/miniconda3`.
- The repo lives at `$HOME/Diff-IP2D`.

If either differs on the new project, fix:
- `source $SCRATCH/miniconda3/etc/profile.d/conda.sh` lines (every `submit_*.sh`).
- `cd $HOME/Diff-IP2D` lines (every `submit_*.sh` except `submit_train.sh`, which has an absolute path — see below).

```bash
# On the new project, recreate the env
conda create -n diffip python=3.10 pip
conda activate diffip
pip install -r requirements.txt
```

### 2. Hardcoded absolute paths to re-point

Run a one-time `sed` after cloning into the new project. Replace `<NEW_USER>` with the new project's user (e.g., `u6cu` was the old one).

```bash
# Source files (data and gaze paths)
sed -i 's|/scratch/u6cu/sx2022.u6cu|/scratch/<NEW_USER>/<NEW_USERHOME>|g' \
    datasets/holoaders.py \
    scripts/prepare_egtea_data.py \
    scripts/extract_egtea_features.py \
    scripts/prepare_meccano_data.py \
    scripts/extract_meccano_features.py \
    submit_extract.sh \
    submit_extract_meccano.sh

# submit_train.sh has a stray absolute path
sed -i 's|cd /home/u6cu/sx2022.u6cu/Diff-IP2D|cd $HOME/Diff-IP2D|g' submit_train.sh
```

The exact source-file lines that carry absolute paths (audit list):

| File | Line | What to update |
|---|---|---|
| `datasets/holoaders.py` | 75 | EGTEA `raw_images_base` |
| `datasets/holoaders.py` | 82 | MECCANO `raw_images_base` |
| `datasets/holoaders.py` | 122 | MECCANO gaze data path |
| `datasets/holoaders.py` | 125 | EGTEA gaze data path |
| `scripts/prepare_egtea_data.py` | 35 | `EGTEA_FRAMES_DIR` |
| `scripts/extract_egtea_features.py` | 31 | `--frames_dir` default |
| `scripts/prepare_meccano_data.py` | 28 | `SRC` annotations dir |
| `scripts/extract_meccano_features.py` | 31 | `--frames_dir` default |
| `submit_extract.sh` | 16 | EGTEA `BASE_DIR` |
| `submit_extract_meccano.sh` | 13–14 | MECCANO `ZIP` and `OUT` |
| `submit_train.sh` | 17 | absolute project home dir |

### 3. Data placement

The training pipeline expects datasets at canonical scratch paths and generated artefacts under the project's `data/` and `common/` directories.

**Raw datasets (read-only, on `/scratch`):**
- EGTEA: `/scratch/<USER>/<USERHOME>/datasets/EGTEA_Gaze_Plus/EGTEA/`
  - `extracted_frames/<video_id>/frame_{:010d}.jpg` (run `submit_extract.sh` to populate from `Raw_Videos/`)
  - `Gaze_Data/gaze_data/<video_id>.txt` (SMI eye-tracker exports)
- MECCANO: `/scratch/<USER>/<USERHOME>/datasets/MECCANO/`
  - `extracted_frames/<video_id 4-digit>/{:05d}.jpg` (run `submit_extract_meccano.sh` to unzip + flatten `MECCANO_RGB_frames.zip`)
  - `annotations/MECCANO_Gaze_data/MECCANO_Gaze_data/{Train,Val,Test}/<vid>_gaze-data.csv`
  - `annotations/MECCANO_action_annotations/MECCANO_{train,val,test}_actions.csv`
  - `annotations/MECCANO_hands_bounding_box_annotations/MECCANO_hands_bounding_box_annotations/{Train,Val,Test}/<vid>.csv`
  - `annotations/MECCANO_NAO_bounding_box_annotations/MECCANO_NAO_bounding_box_annotations/instances_meccano_{train,val,test}.json`

**Generated artefacts (under repo `data/`, copied from old project as tar.gz on first migration):**
- `data/egtea/{labels/,feats_train/,feats_test/,video_info.json,egtea_eval_labels.pkl}`
- `data/meccano/{labels/,feats_train/,feats_test/,video_info.json,meccano_eval_labels.pkl}`
- `data/uid2future_file_name_egtea.pickle`
- `data/uid2future_file_name_meccano.pickle`
- `data/homos_train/`, `data/homos_test/` (auto-cached on first training run; pre-shipped if available)
- `common/egtea-annotations/`, `common/meccano-annotations/`, `common/rulstm/`, `common/epic-kitchens-100-annotations/`
- `base_models/model.pth.tar` (BNInception OCT base model, ~177 MB)

When migrating: tar these artefacts on the old machine before allocation expires and untar on the new machine. They take ~1 GB total and rebuilding from raw zips costs hours of GPU time.

### 4. Sanity checks before launching the first GPU job

```bash
# (a) Conda env is correct
conda activate diffip && python -c "import torch, lmdbdict, scipy; print('ok')"

# (b) EGTEA pipeline loads
python -c "
from datasets.datasetopts import DatasetArgs
from datasets.dataset_utils import get_egtea_annotation
a = DatasetArgs(ek_version='egtea', mode='train', base_path='./')
df = get_egtea_annotation(a.annot_path, a.rulstm_annot_path, a.label_path, a.eval_label_path,
                          partition='train', use_label_only=True)
print('EGTEA train rows:', len(df))
"

# (c) MECCANO pipeline loads
python -c "
from datasets.datasetopts import DatasetArgs
from datasets.dataset_utils import get_meccano_annotation
a = DatasetArgs(ek_version='meccano', mode='train', base_path='./')
df = get_meccano_annotation(a.annot_path, a.rulstm_annot_path, a.label_path, a.eval_label_path,
                            partition='train', use_label_only=True)
print('MECCANO train rows:', len(df))
"

# (d) Determinism rerun: same command → identical metrics
sbatch submit_eval_egtea.sh   # then diff the two metric reports
```

### 5. SLURM walltime contract

Sweeps in this repo are sized for a 24-hour SLURM walltime. Per-experiment cost (post warm-up of homography cache):
- EGTEA per-variant 30-ep cycle (train + 2 evals): ~13 min.
- MECCANO per-variant 30-ep cycle: ~17–18 min (warm cache); ~27 min (cold cache, first run only).

A 13-version × 5-epoch-count sweep (~65 cells) fits in 14 hours; a 13 × 3-epoch-count sweep fits in 10 hours. Use `--dependency=afterany` to chain sweeps that share `./diffip_weights/` (see "Engineering lessons" below).

## Commands

All training/evaluation uses PyTorch distributed launch (single-GPU is still wrapped):

```bash
# EGTEA: train (30 epochs, batch_size=32)
bash train.sh                    # generic EK100 entry
bash train_egtea.sh              # EGTEA entry

# Evaluate trajectory / affordance
bash val_traj.sh
bash val_affordance.sh

# MECCANO: train + eval (one-shot)
sbatch submit_train_meccano.sh                     # 30 epochs by default; EPOCHS=N to override
EPOCHS=45 sbatch submit_train_meccano.sh

# Multi-version sweeps (EGTEA)
sbatch submit_55_epochs.sh                         # 13 versions @ 55 epochs
sbatch submit_10_15_45_epochs.sh                   # 13 versions × {10,15,45}

# Multi-version sweeps (MECCANO) — chain via dependency to avoid checkpoint races
JA=$(sbatch --parsable submit_meccano_ep_sweep_A.sh)   # ep 10/15/20/25/35
JB=$(sbatch --parsable --dependency=afterany:$JA submit_meccano_ep_sweep_B.sh)   # ep 40/45/50
JC=$(sbatch --parsable --dependency=afterany:$JB submit_meccano_ep_sweep_C.sh)   # ep 55/60
```

The `.sh` wrappers all funnel into `traineval.py` via `run_experiment.py`, which is a thin CLI passthrough.

For a fresh dataset, the substantive pipeline is:
1. RGB extraction (one-off): `sbatch submit_extract.sh` (EGTEA) or `sbatch submit_extract_meccano.sh` (MECCANO).
2. Annotation prep (CPU, fast): `python scripts/prepare_egtea_data.py` or `python scripts/prepare_meccano_data.py`.
3. BNInception feature extraction (GPU, ~30–40 min): `sbatch submit_extract_features.sh` (EGTEA) or `sbatch submit_extract_meccano_features.sh` (MECCANO).
4. Training: `sbatch submit_train_egtea.sh` or `sbatch submit_train_meccano.sh`.

### Environment Setup

```bash
conda create -n diffip python=3.10 pip
conda activate diffip
pip install -r requirements.txt
```

(Python 3.8 also works; the project's runs were on 3.10.)

## Architecture

### Data Flow

1. **Data loading** (`datasets/holoaders.py`): LMDB features + pickle labels → `EpicHODataset` with SIFT-based homography auto-cache between frames.
2. **HOI Encoder** (`networks/transformer.py`): `ObjectTransformerModel` — spatial-temporal transformer processes multi-frame RGB features (1024-D) into three 512-D streams (global, hand, object).
3. **Pre-encoding** (`diffip2d/pre_encoder.py`): `SideFusionEncoder` fuses the three streams (3 × 512 → 512); `MotionEncoder` encodes per-pair homography matrices; `GazeSideFusionEncoder` (this fork) wraps `SideFusionEncoder` with an optional sigmoid gate over the object stream.
4. **Diffusion denoising** (`diffip2d/gaussian_diffusion.py` + `diffip2d/transformer_model.py`): `HOIDiffusion` runs iterative denoising through `MADT` (Motion-Aware Denoising Transformer) over 1000 sqrt-schedule timesteps; each `DecoderBlock` is `SelfAttn → EgomotionXA → [GazeXA] → MLP` with zero-init LayerScale on the gaze residual.
5. **Post-decoding** (`diffip2d/post_decoder.py`, `networks/{traj,affordance}_decoder.py`): `TrajDecoder` maps 512-D → 2-D hand coordinates; CVAE heads emit contact-point predictions.

### Key Modules

- **`traineval.py`**: Main entry point. Builds all models, runs `TrainValLoop`. Sets determinism flags at top.
- **`basic_utils.py`**: `create_network_and_diffusion()` assembles diffusion components.
- **`netscripts/epoch_feat.py`**: `TrainValLoop` class — core training/validation loop with loss computation, distributed training coordination, checkpoint cleanup at line ~707 (see Engineering Lessons).
- **`netscripts/get_optimizer.py`**: Per-module parameter groups (sf_encoder, model_denoise, traj_decoder, motion_encoder, model_hoi, obj_head, gaze_encoder when `--use_gaze`) with warmup + cosine annealing keyed to `--epochs`.
- **`options/netsopts.py`**: Network hyperparameters (hidden_dim=512, diffusion_steps=1000, noise_schedule="sqrt", loss weights lambda_*, all gaze CLI flags).
- **`options/expopts.py`**: Experiment config (`--ek_version` choices include `egtea` and `meccano`, `--seq_len_obs=10`, `--seq_len_unobs=3`, `--sample_times=10`, `--fast_test`).

### Loss Function

Weighted combination: `lambda_traj * traj_loss + lambda_obj * obj_loss + lambda_traj_kl * traj_kl + lambda_obj_kl * obj_kl + lambda_diff * diff_loss`. All λ inherited from Diff-IP2D defaults; `--learnable_weight=True` adds learnable scalars per term.

### Evaluation

- **Trajectory**: WDE (Worst-of-hands Displacement Error, mean over future window) and FDE (final-frame variant) — best-of-many over `--sample_times` stochastic samples (`evaluation/traj_eval.py`).
- **Affordance**: SIM, AUC-J, NSS — Bylinskii et al. saliency metrics on Gaussian heatmaps built from farthest-point-sampled prediction clusters (`evaluation/affordance_eval.py`).
- Inference uses `--fast_test=True` (DDIM-respaced ~50 steps) by default; metrics agree with full 1000-step inference to the third decimal on a sub-grid we tested.

## Important Details

- Distributed training uses `LOCAL_RANK` env variable extensively; all entry points expect it via `torch.distributed.launch --use_env`.
- Homography matrices are computed via SIFT on first run, then cached to `data/homos_{train,test}/<participant_id>/homo/<video_id>/<frame>.npy` under an *atomic-rename* write protocol (see Engineering Lessons).
- Features are stored in LMDB format. EGTEA uses single LMDBs per split (`feats_train/data.lmdb`, `feats_test/data.lmdb`). MECCANO uses the same single-LMDB layout. EK100 uses a two-part split (`...part1.lmdb` + `...part2.lmdb`); the loader's `_open_lmdb` and `_get_feat_dict` branch on `ek_version in ('egtea', 'meccano')` to pick the layout.
- Checkpoints saved to `diffip_weights/checkpoint_<epoch>.pth.tar`; old checkpoints are deleted after each epoch save (one file per run; **races under concurrent SLURM jobs** — see Engineering Lessons).
- Predictions collected to `collected_pred_traj/` and `collected_pred_aff/`.
- `preprocess/ho_types.py` defines core data types (HandSide, HandState, BBox, etc.) with protobuf serialization.

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
- Label window length $W_{\text{label}} = 7$ frames (vs.\ EGTEA's 16, because of the lower native fps).
- Frame template: `'{:05d}.jpg'` (5-digit, no `frame_` prefix), set per-`ek_version` in `holoaders.py`.
- Hand labels: derived from VGG-VIA bbox CSVs (`SX_Hand` = LEFT, `DX_Hand` = RIGHT). Per-frame bbox centre, then Cubic Hermite spline over the window with extrapolation clipped to the observed range.
- Affordance labels: derived from NAO COCO bboxes. Five points sampled uniformly inside the bbox at the action start frame, with seeded `RandomState(0)`.
- Coord scaling: pixel coords (1920×1080) scaled to canonical (456, 256) via `S_x=456/1920, S_y=256/1080`. The aspect-ratio match (~16:9 on both) means uniform per-axis scaling preserves geometry.
- Splits: train (videos 0001–0006, 0013–0016, 0018), val (0007, 0017), test (0008–0012, 0019, 0020).
- LMDB partition: `feats_train.lmdb` ↔ train split; `feats_test.lmdb` ↔ **val + test** (val frames are needed by `--evaluate --traj_only` which uses `mode != 'train'`). This is unlike EGTEA where val and test are the same set.
- Gaze: 200 Hz CSVs per video, parsed by `_load_meccano_gaze` in `holoaders.py`. Multiple samples per frame are mean-aggregated. Coords normalised by 1920×1080.
- Test eval set: 904 samples for trajectory (val), 2584 for affordance (test).

### MECCANO data prep pipeline
- `scripts/prepare_meccano_data.py` produces all artefacts (label pickles, video_info.json, eval labels, uid2future, split CSVs in EGTEA-compatible column layout).
- `scripts/extract_meccano_features.py` runs BNInception per-frame; the `--split` flag is `train` (train CSV only) or `test` (val + test CSVs).
- `submit_extract_meccano.sh` unzips and flattens `MECCANO_RGB_frames.zip` to `extracted_frames/<vid>/`.
- `submit_extract_meccano_features.sh` chains both feature-extraction invocations.

### MECCANO loader contract checklist (when porting to a third dataset)
1. Compute $W_{\text{label}} = \lfloor t_{\text{ant}} \cdot r_0\rfloor + 1$ from native fps and use consistently in prepare-script *and* `datasetopts.py`. A mismatch fires the holoaders assertion at line 506 with a "last observation frame mismatch" message.
2. Force `video_id` and `participant_id` to canonical strings immediately after `pd.read_csv` (pandas silently converts `"0001"` → int 1, which corrupts LMDB keys).
3. Audit `ek_version != 'egtea'` predicates and convert to positive `in (...)` form. They were used as EK100 selectors, not "not egtea" predicates.
4. Partition LMDBs by *loader-mode visibility* (`mode == 'train'` ↔ feats_train; everything else ↔ feats_test), not by clip-membership.
5. Add the new dataset to `--ek_version` argparse `choices=`.
6. Pick a frame template matching the dataset's filename convention.
7. Implement a `_load_<dataset>_gaze` parser and dispatch from `_load_gaze_for_video` if gaze is available.

## Deterministic Training

Training is made fully deterministic via a four-level scaffold (see also Engineering Lessons):

1. **PyTorch flags** (top of `traineval.py`): `cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True)`.
2. **CUBLAS workspace** (env var): `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
3. **Multi-source seeding**: Python `random`, NumPy, Torch CPU/CUDA generators all seeded from `SEED=42`; DataLoader workers re-seeded via `worker_init_fn` in `holoaders.py`.
4. **Algorithmic seeding**: `evaluation/affordance_eval.py` farthest-point sampling uses `start_idx=0` (deterministic), AUC-J jitter uses `np.random.RandomState(0)` (seeded).

Results are bit-exact across reruns on the same GPU. Different GPU architectures may give different round-off; in our experience the *ranking* of variants is preserved across architectures even if absolute numbers shift slightly.

## Gaze Integration Flags (`options/netsopts.py`)

| Flag | Description |
|------|-------------|
| `--use_gaze` | Enable gaze (dual-stream: heatmap CNN + coord MLP in GazeEncoder, GazeSideFusionEncoder gating, MADT cross-attention with LayerScale) |
| `--gaze_coord_only` | Use only coordinate MLP, skip heatmap CNN |
| `--gaze_heatmap_only` | Use only heatmap CNN, skip coordinate MLP (handled in `epoch_feat.py`, not model config) |
| `--gaze_fusion_only` | Gaze only in SideFusionEncoder gating, no MADT cross-attention |
| `--gaze_alpha_clamp=X` | Clamp LayerScale gaze_alpha max to X (0 = no clamp) |
| `--gaze_last_n_blocks=N` | Gaze cross-attention only in last N of 6 MADT blocks (0 = all) |
| `--gaze_detach_diffusion` | Detach gaze features before passing to diffusion/MADT |
| `--gaze_fixed_delta=N` | Hard mask: hand[t] attends only to gaze[max(0, t−N)] |
| `--gaze_bias_init_delta=N` | Initialize learnable temporal bias with Gaussian bump at offset N |
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

These are the operational findings from the EGTEA + MECCANO sweeps. Each cost real GPU time on the previous project; they should not need to be re-discovered.

### L1. Concurrent SLURM jobs corrupt each other's checkpoints
`netscripts/epoch_feat.py:707–714` deletes *every* `checkpoint_*.pth.tar` in `./diffip_weights/` except the current epoch's. Under sequential SLURM jobs this is correct; under parallel jobs that share the directory, it races and silently destroys other jobs' final checkpoints, producing the symptom "training looked successful but eval cannot find the checkpoint".

**Fix:** chain sweeps via `--dependency=afterany:$PREV_JOB_ID`. Code-side alternative is per-job checkpoint subdirectories (not implemented).

### L2. Concurrent SIFT homography writes can produce 0-byte files
The auto-cached homography at `holoaders.py:269/281/388` is written by every DataLoader worker that hits a cache miss. With 16 workers, two can race on the same path; `np.save` is not atomic, so a reader can observe a partial file and raise `EOFError: No data left in file`.

**Fix:** atomic-rename writes (`.tmp.<pid>` then `os.replace`) plus robust loads (try/except `EOFError, ValueError, OSError` → recompute). Both are already in the current codebase.

### L3. Pandas integer parsing of zero-padded `video_id` columns
`pd.read_csv` infers a column of `"0001"`, `"0002"`, … as `int64`, dropping the zero-padding. The downstream LMDB key `os.path.join(action.video_id, frame_name)` then becomes `"1/00001.jpg"` and fails to match the stored key `b"0001/00001.jpg"`.

**Fix:** in `get_meccano_annotation` (and any new dataset adapter), force `df['video_id'] = df['video_id'].map(lambda v: f"{int(v):04d}")` immediately after read.

### L4. The `ek_version != 'egtea'` predicate as an EK100 selector
The reference Diff-IP2D code uses `self.ek_version == 'egtea'` to select the single-LMDB layout and the implicit `else` for the EK100 split-LMDB layout. With a third dataset (MECCANO) added, `else` silently routed MECCANO into the EK100 path and crashed with `'FeaturesHOLoader' object has no attribute 'env1'`.

**Fix:** all three branch points (`_open_lmdb`, `_get_feat_dict`, `_make_full_name`) use `ek_version in ('egtea', 'meccano')` for the single-LMDB layout. Audit any future similar predicates.

### L5. Anchor / window alignment mismatch with native fps
Hard-coding $W_{\text{label}} = 16$ (EGTEA) on a 12-fps dataset (MECCANO) makes the label anchor 9 frames off from the observation sampler's last frame, firing the assertion at `holoaders.py:506`. Always derive $W_{\text{label}}$ from $\lfloor t_{\text{ant}} \cdot r_0\rfloor + 1$ at the dataset's native fps.

### L6. LMDB partition by clip-membership, not loader-mode
First pass put MECCANO val into `feats_train.lmdb`. The loader chooses LMDB by *mode* (`'train'` vs.\ everything else), not by *split*; val frames need to be in `feats_test.lmdb`.

**Fix:** `feats_train.lmdb` ↔ train split only; `feats_test.lmdb` ↔ val + test.

### L7. `argparse choices=` is a hard barrier for new datasets
Adding `'meccano'` to `--ek_version` requires editing `options/expopts.py:11`. Otherwise the very first attempted training run dies with a one-second `argparse` error.

### L8. EpicVideo column requirements
`EpicVideo._get_actions` accesses ~15 columns by attribute on a pandas row. Missing columns raise `AttributeError` rather than `KeyError`. Always reproduce the EGTEA column set verbatim (including `participant_id`, `start_time`, `stop_time`, `verb`, `noun`, `action`, `verb_class`, `noun_class`, `action_class`, `all_nouns`, `all_noun_classes`) when writing a new split CSV.

### L9. Cosine LR schedule is keyed to total `--epochs`
Each `--epochs=N` run is a *different optimisation problem*, not a checkpoint along a single trajectory. NSS oscillates wildly across epoch counts (range 0.5–0.8 on EGTEA, 0.10–0.20 on MECCANO). Single-checkpoint reporting is brittle; always evaluate at multiple epoch counts or with cross-seed averaging.

### L10. Determinism is cheap to enable up-front, expensive to retrofit
The 4-level scaffold (PyTorch flags, CUBLAS workspace, multi-source seeding, algorithmic FPS/AUC-J seeding) should be the *first commit* on any new project. Each layer caught at least one bug in our history.

### L11. SLURM walltime is an upper bound, not a target
Estimate per-cell cost before submitting; use 25% buffer. Submission failures (`Slurm temporarily unable to accept job`) are transient — wrap `sbatch` in `until ... done` retry loop with a 30-second sleep.

## Reference Results

All results below are deterministic on the previous project's GPU class. Numbers may shift across architectures; rankings should be preserved.

### Variant descriptions (used throughout EGTEA + MECCANO sweeps)
| Ver | Description | Key flags |
|-----|-------------|-----------|
| Baseline | No gaze | (none) |
| v2 | Full gaze: dual-stream encoder, LayerScale, learnable temporal bias | `--use_gaze` |
| v4 | Coordinate-only (no heatmap CNN) | `--use_gaze --gaze_coord_only` |
| v6 | SideFusionEncoder gating only, no MADT cross-attention | `--use_gaze --gaze_fusion_only` |
| v7 | Clamped LayerScale alpha max=0.1 (≡ v2 — alpha never exceeds 0.1) | `--use_gaze --gaze_alpha_clamp=0.1` |
| v8 | Gaze in last 2 of 6 MADT blocks only | `--use_gaze --gaze_last_n_blocks=2` |
| v9 | Detach gaze features from diffusion gradient path | `--use_gaze --gaze_detach_diffusion` |
| v10 | Hard temporal mask: hand[t] → gaze[t−2] only | `--use_gaze --gaze_fixed_delta=2` |
| v11 | Learnable bias init with Gaussian bump at delta=2, amp=2.0 | `--use_gaze --gaze_bias_init_delta=2 --gaze_bias_init_amp=2.0` |
| v12 | Learnable bias init with Gaussian bump at delta=3, amp=0.5 | `--use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5` |
| v12a | v12 + heatmap only (no coord MLP) | v12 flags + `--gaze_heatmap_only` |
| v12b | v12 + CFG dropout 10% | v12 flags + `--gaze_cfg_dropout=0.1` |
| v12c | v12 + gaze cross-attention BEFORE egomotion | v12 flags + `--gaze_before_motion` |
| v13 | v12c + detach (≈ v12c on EGTEA, partial-equivalence on MECCANO) | v12 flags + `--gaze_before_motion --gaze_detach_diffusion` |

### EGTEA — 30-epoch reference

| Ver | WDE ↓ | FDE ↓ | SIM ↑ | AUC-J ↑ | NSS ↑ |
|-----|-------|-------|-------|---------|-------|
| Baseline | 0.407 | 0.210 | 0.208 | 0.733 | 0.858 |
| v2 | 0.408 | 0.222 | 0.226 | 0.749 | 0.963 |
| v4 | 0.398 | 0.215 | 0.219 | 0.738 | 0.919 |
| v6 | 0.415 | 0.226 | 0.224 | 0.759 | 0.975 |
| v7 | 0.408 | 0.222 | 0.226 | 0.749 | 0.963 |
| v8 | 0.418 | 0.228 | 0.203 | 0.729 | 0.807 |
| v9 | 0.397 | 0.215 | 0.220 | 0.741 | 0.928 |
| v10 | 0.412 | 0.221 | 0.218 | 0.728 | 0.902 |
| v11 | 0.401 | 0.216 | 0.223 | 0.740 | 0.961 |
| v12 | 0.405 | 0.216 | 0.224 | 0.751 | 0.958 |
| v12a | 0.408 | 0.221 | 0.212 | 0.729 | 0.866 |
| v12b | 0.415 | 0.225 | 0.225 | 0.753 | 0.951 |
| v12c | 0.407 | 0.220 | 0.222 | 0.739 | 0.952 |
| v13 | 0.404 | 0.219 | 0.225 | 0.750 | 0.959 |

### MECCANO — 30-epoch reference

| Ver | WDE ↓ | FDE ↓ | SIM ↑ | AUC-J ↑ | NSS ↑ |
|-----|-------|-------|-------|---------|-------|
| Baseline | 0.467 | 0.252 | 0.217 | 0.785 | 1.076 |
| v2 | 0.397 | 0.210 | 0.208 | 0.772 | 1.008 |
| v4 | 0.394 | 0.210 | 0.212 | 0.777 | 1.043 |
| v6 | 0.392 | 0.211 | 0.210 | 0.775 | 1.001 |
| v8 | 0.393 | 0.217 | 0.215 | 0.784 | 1.061 |
| v9 | 0.384 | 0.204 | 0.211 | 0.776 | 1.016 |
| v10 | 0.410 | 0.220 | 0.212 | 0.778 | 1.035 |
| v11 | 0.385 | 0.209 | 0.215 | 0.781 | 1.053 |
| v12 | 0.401 | 0.216 | 0.213 | 0.778 | 1.052 |
| v12a | 0.404 | 0.219 | 0.214 | 0.781 | 1.040 |
| v12b | 0.410 | 0.217 | 0.212 | 0.778 | 1.042 |
| v12c | 0.423 | 0.227 | 0.212 | 0.781 | 1.038 |
| v13 | 0.378 | 0.201 | 0.210 | 0.774 | 1.004 |

### Key cross-dataset findings

1. **Gaze improves trajectory on MECCANO and is essentially neutral on EGTEA.** Every gaze variant on MECCANO at 30 ep drops WDE by 0.045–0.089 vs.\ Baseline; on EGTEA the same comparisons are flat (0.397–0.418 vs.\ 0.407).
2. **Gaze improves affordance on EGTEA at 30 ep but regresses it on MECCANO at 30 ep.** EGTEA NSS gain +0.117 (v6); MECCANO NSS −0.015 to +0.005. The schedule-aware reading is different: at 40 ep on MECCANO, v12b reaches NSS 1.112 (vs.\ Baseline 0.995), the global maximum across the 286-cell sweep.
3. **The temporal-alignment axis is the most reliably useful design choice.** Gaussian-init bias (v11/v12) outperforms hard mask (v10) on both datasets; the soft prior is the most transferable design lever.
4. **The fusion-site choice is dataset-specific.** Gating-only (v6) wins on EGTEA at 30 ep; gating, depth-restricted XA, and full pipeline are roughly equivalent on MECCANO at 30 ep.
5. **Pre-registered equivalences.** v7 ≡ v2 bit-exactly on both datasets (LayerScale clamp is a no-op because $\gamma_{\text{gaze}}$ never exceeds 0.1). v13 ≈ v12c at the noise floor on EGTEA but not on MECCANO trajectory (where v13's WDE is 0.045 lower than v12c's at 30 ep).
6. **NSS oscillation amplitude differs across datasets.** Within-variant range across 11 epoch counts is 0.5–0.8 on EGTEA and 0.10–0.20 on MECCANO. Multi-checkpoint reporting is methodologically necessary on EGTEA, less so on MECCANO.

### Schedule-best summary (best metric across 11 epoch counts)

| Metric | EGTEA best | MECCANO best |
|---|---|---|
| WDE ↓ | 0.337 (v12b @ 15 ep) | 0.314 (v12b @ 20 ep) |
| FDE ↓ | 0.168 (v8 @ 10 ep) | 0.162 (v12b @ 20 ep) |
| NSS ↑ | 1.117 (v6 @ 15 ep) | 1.112 (v12b @ 40 ep) |
| AUC-J ↑ | 0.781 (v6/v8 @ 15 ep) | 0.790 (v12b @ 40 ep) |

### Sweep cost on the previous project (for budgeting on the new one)

| Sweep | Walltime | Cells |
|---|---|---|
| EGTEA gaze full @ 30 ep | 3.6 h | 13 |
| EGTEA epoch sweep 10/15/45 | 4.1 h | 39 |
| EGTEA epoch 55 | 2.8 h | 13 |
| MECCANO RGB extraction | 0.6 h CPU | — |
| MECCANO BNInception features | 0.6 h | — |
| MECCANO Baseline @ 30 ep | 0.5 h | 1 |
| MECCANO gaze full @ 30 ep | 3.6 h | 12 |
| MECCANO sweep A (10/15/20/25/35) | 14.0 h | 65 |
| MECCANO sweep B (40/45/50) | 15.7 h | 39 |
| MECCANO sweep C (55, partial 60) | 12.5 h | 16 |
| MECCANO 60-ep follow-up | 6.3 h | 10 |
| **Total** | **~64 h** | **210** |
