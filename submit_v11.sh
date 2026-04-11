#!/bin/bash
#SBATCH --job-name=gaze_v11
#SBATCH --output=logs/gaze_v11_%j.out
#SBATCH --error=logs/gaze_v11_%j.err
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

echo "=== v11: Learnable bias init with Gaussian bump @ delta=2 ==="
echo "=== TRAINING ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_train_v11.py

echo "=== EVAL TRAJECTORY ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_val_traj_v11.py

echo "=== EVAL AFFORDANCE ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_val_affordance_v11.py

echo "=== v11 COMPLETE ==="
