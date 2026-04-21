#!/bin/bash
#SBATCH --job-name=gaze_v10
#SBATCH --output=logs/gaze_v10_%j.out
#SBATCH --error=logs/gaze_v10_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

cd $HOME/Diff-IP2D
mkdir -p logs

echo "=== v10: Fixed temporal delta = 2 frames (~333ms at 6fps) ==="
echo "=== TRAINING ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_train_v10.py

echo "=== EVAL TRAJECTORY ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_val_traj_v10.py

echo "=== EVAL AFFORDANCE ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_val_affordance_v10.py

echo "=== v10 COMPLETE ==="
