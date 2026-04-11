#!/bin/bash
#SBATCH --job-name=gaze_repro
#SBATCH --output=logs/gaze_repro_%j.out
#SBATCH --error=logs/gaze_repro_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

cd $HOME/Diff-IP2D-main
mkdir -p logs

echo "==========================================="
echo "=== BASELINE (no gaze) reproducibility  ==="
echo "==========================================="
echo "=== BASELINE TRAINING ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_train_baseline.py

echo "=== BASELINE EVAL TRAJ ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_val_traj_baseline.py

echo "=== BASELINE EVAL AFF ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_val_affordance_baseline.py

echo "==========================================="
echo "=== v2 reproducibility (--use_gaze)     ==="
echo "==========================================="
echo "=== v2 TRAINING ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_train_v2_repro.py

echo "=== v2 EVAL TRAJ ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_val_traj_v2_repro.py

echo "=== v2 EVAL AFF ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_val_affordance_v2_repro.py

echo "=== ALL DONE ==="
