#!/bin/bash
#SBATCH --job-name=v12_var
#SBATCH --output=logs/v12_var_%j.out
#SBATCH --error=logs/v12_var_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

cd $HOME/Diff-IP2D-main
mkdir -p logs

TRAIN_BASE="--ek_version=egtea --epochs=30 --batch_size=32 --num_classes=106 --seq_len_obs=10 --seq_len_unobs=3 --learnable_weight=True"
EVAL_TRAJ="--evaluate --ek_version=egtea --num_classes=106 --seq_len_obs=10 --seq_len_unobs=3 --resume=./diffip_weights/checkpoint_30.pth.tar --traj_only"
EVAL_AFF="--evaluate --ek_version=egtea --num_classes=106 --seq_len_obs=10 --seq_len_unobs=3 --resume=./diffip_weights/checkpoint_30.pth.tar"

# v12 base flags
V12="--use_gaze --gaze_bias_init_delta=3 --gaze_bias_init_amp=0.5"

run_exp() {
    local name="$1"
    local flags="$2"
    echo "=========================================="
    echo "=== $name ==="
    echo "=========================================="
    echo "=== $name TRAINING ==="
    TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_experiment.py $TRAIN_BASE $flags
    echo "=== $name EVAL TRAJ ==="
    TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_experiment.py $EVAL_TRAJ $flags
    echo "=== $name EVAL AFF ==="
    TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_experiment.py $EVAL_AFF $flags
    echo "=== $name DONE ==="
}

run_exp "v12a (heatmap only)" "$V12 --gaze_heatmap_only"
run_exp "v12b (CFG dropout 10%)" "$V12 --gaze_cfg_dropout=0.1"
run_exp "v12c (gaze before motion)" "$V12 --gaze_before_motion"

echo "=========================================="
echo "=== ALL v12 VARIANTS COMPLETE          ==="
echo "=========================================="
