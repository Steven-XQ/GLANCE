#!/bin/bash
#SBATCH --job-name=gaze_v7_v8_v9
#SBATCH --output=logs/gaze_experiments_%j.out
#SBATCH --error=logs/gaze_experiments_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

cd $HOME/Diff-IP2D-main
mkdir -p logs

echo "=========================================="
echo "=== v7: Clamped gaze_alpha (max=0.1)  ==="
echo "=========================================="
echo "=== v7 TRAINING ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_train_v7.py

echo "=== v7 EVAL TRAJECTORY ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_val_traj_v7.py

echo "=== v7 EVAL AFFORDANCE ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_val_affordance_v7.py

echo "=========================================="
echo "=== v8: Gaze in last 2 blocks only    ==="
echo "=========================================="
echo "=== v8 TRAINING ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_train_v8.py

echo "=== v8 EVAL TRAJECTORY ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_val_traj_v8.py

echo "=== v8 EVAL AFFORDANCE ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_val_affordance_v8.py

echo "=========================================="
echo "=== v9: Detached gaze for diffusion   ==="
echo "=========================================="
echo "=== v9 TRAINING ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_train_v9.py

echo "=== v9 EVAL TRAJECTORY ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_val_traj_v9.py

echo "=== v9 EVAL AFFORDANCE ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_val_affordance_v9.py

echo "=========================================="
echo "=== ALL EXPERIMENTS COMPLETE           ==="
echo "=========================================="
