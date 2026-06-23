#!/bin/bash
# 10-seed sweep: baseline + 12 gaze variants on EGTEA + MECCANO at 35 epochs.
# Sequential on 8 GPUs, single shared diffip_weights/ dir.
# Resumable: skips any (dataset, variant, seed) cell whose .done marker exists.

set -uo pipefail

cd "$(dirname "$0")"

source /home/nvidia/miniconda3/etc/profile.d/conda.sh
conda activate diffip

SWEEP_ROOT="seed_sweep"
mkdir -p "$SWEEP_ROOT/logs" "$SWEEP_ROOT/markers"
RESULTS_CSV="$SWEEP_ROOT/results.csv"

if [ ! -f "$RESULTS_CSV" ]; then
    echo "dataset,variant,seed,epochs,wde,fde,sim,auc_j,nss" > "$RESULTS_CSV"
fi

SEEDS=(13 42 256 777 2025)
DATASETS=(egtea meccano)
EPOCHS=35
NPROC=8
BATCH=8

# 1 baseline + 12 gaze variants (skipping v7 which equals v1 bit-exactly per CLAUDE.md L1 lesson).
VARIANTS=(
  "baseline:"
  "v1:--use_gaze"
  "v2:--use_gaze --gaze_coord_only"
  "v3:--use_gaze --gaze_fusion_only"
  "v4:--use_gaze --gaze_last_n_blocks=2"
  "v5:--use_gaze --gaze_detach_diffusion"
  "v6:--use_gaze --gaze_fixed_delta=2"
  "v7:--use_gaze --gaze_bias_init_delta=2 --gaze_bias_init_amp=2.0"
  "v8:--use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5"
  "v9:--use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5 --gaze_heatmap_only"
  "v10:--use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5 --gaze_cfg_dropout=0.1"
  "v11:--use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5 --gaze_before_motion"
  "v12:--use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5 --gaze_before_motion --gaze_detach_diffusion"
)

run_one() {
    local dataset=$1 name=$2 seed=$3 flags=$4

    local marker="$SWEEP_ROOT/markers/${dataset}_${name}_seed${seed}.done"
    local log="$SWEEP_ROOT/logs/${dataset}_${name}_seed${seed}.log"

    if [ -f "$marker" ]; then
        echo "[skip] $dataset $name seed=$seed (already done)"
        return 0
    fi

    local num_classes
    case "$dataset" in
        egtea)   num_classes=106 ;;
        meccano) num_classes=61  ;;
        *) echo "unknown dataset: $dataset" >&2; return 1 ;;
    esac

    local train_args="--ek_version=$dataset --epochs=$EPOCHS --batch_size=$BATCH --num_classes=$num_classes --seq_len_obs=10 --seq_len_unobs=3 --learnable_weight=True --manual_seed=$seed"
    local eval_args="--evaluate --ek_version=$dataset --num_classes=$num_classes --seq_len_obs=10 --seq_len_unobs=3 --resume=./diffip_weights/checkpoint_${EPOCHS}.pth.tar --manual_seed=$seed"

    echo "[run]  $dataset $name seed=$seed  ->  $log"

    {
        echo "=== START $dataset $name seed=$seed @ $(date -Iseconds) ==="
        # purge any prior per-rank prediction files (rank-count mismatch hangs the poll loop)
        rm -rf collected_pred_traj collected_pred_aff

        echo "=== TRAIN ==="
        TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch \
            --nproc_per_node=$NPROC --master_port=12325 --use_env \
            run_experiment.py $train_args $flags
        local train_status=$?

        if [ "$train_status" -ne 0 ]; then
            echo "=== TRAIN FAILED status=$train_status ==="
            return 1
        fi

        rm -rf collected_pred_traj
        echo "=== EVAL TRAJ ==="
        TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch \
            --nproc_per_node=$NPROC --master_port=12326 --use_env \
            run_experiment.py $eval_args --traj_only $flags

        rm -rf collected_pred_aff
        echo "=== EVAL AFF ==="
        TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch \
            --nproc_per_node=$NPROC --master_port=12327 --use_env \
            run_experiment.py $eval_args $flags

        echo "=== END $dataset $name seed=$seed @ $(date -Iseconds) ==="
    } &> "$log"

    if ! grep -q "^=== END " "$log"; then
        echo "[fail] $dataset $name seed=$seed (see $log)"
        return 1
    fi

    if ! python aggregate_seed_sweep.py append \
            --log "$log" --dataset "$dataset" --variant "$name" --seed "$seed" --epochs "$EPOCHS" \
            --csv "$RESULTS_CSV"; then
        echo "[fail] $dataset $name seed=$seed (metric parse error; see $log)"
        return 1
    fi

    touch "$marker"
    echo "[done] $dataset $name seed=$seed"
}

export -f run_one

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for entry in "${VARIANTS[@]}"; do
            name="${entry%%:*}"
            flags="${entry#*:}"
            run_one "$dataset" "$name" "$seed" "$flags" || echo "[warn] continuing past failure"
        done
    done
done

echo "=== sweep complete; finalizing ==="
python aggregate_seed_sweep.py finalize --csv "$RESULTS_CSV" --out "$SWEEP_ROOT/aggregated.md"
echo "results CSV : $RESULTS_CSV"
echo "summary md  : $SWEEP_ROOT/aggregated.md"
