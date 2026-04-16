#!/bin/bash
#SBATCH --job-name=det_test
#SBATCH --output=logs/det_test_%j.out
#SBATCH --error=logs/det_test_%j.err
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

echo "=== BASELINE RUN 1 ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_train_baseline.py
echo "=== BASELINE RUN 1 EVAL TRAJ ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_val_traj_baseline.py
echo "=== BASELINE RUN 1 EVAL AFF ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_val_affordance_baseline.py

echo "=== BASELINE RUN 2 ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_train_baseline.py
echo "=== BASELINE RUN 2 EVAL TRAJ ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_val_traj_baseline.py
echo "=== BASELINE RUN 2 EVAL AFF ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_val_affordance_baseline.py

echo "=== v2 RUN 1 ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_train_egtea.py
echo "=== v2 RUN 1 EVAL TRAJ ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_val_traj_egtea.py
echo "=== v2 RUN 1 EVAL AFF ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_val_affordance_egtea.py

echo "=== v2 RUN 2 ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_train_egtea.py
echo "=== v2 RUN 2 EVAL TRAJ ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_val_traj_egtea.py
echo "=== v2 RUN 2 EVAL AFF ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_val_affordance_egtea.py

echo "=== ALL DETERMINISTIC TESTS DONE ==="
