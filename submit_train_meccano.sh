#!/bin/bash
#SBATCH --job-name=meccano_train
#SBATCH --output=logs/meccano_train_%j.out
#SBATCH --error=logs/meccano_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

cd $HOME/Diff-IP2D
mkdir -p logs

EPOCHS=${EPOCHS:-30}
TRAIN_BASE="--ek_version=meccano --epochs=$EPOCHS --batch_size=32 --num_classes=61 --seq_len_obs=10 --seq_len_unobs=3 --learnable_weight=True"
EVAL_RESUME="--evaluate --ek_version=meccano --num_classes=61 --seq_len_obs=10 --seq_len_unobs=3 --resume=./diffip_weights/checkpoint_${EPOCHS}.pth.tar"

echo "=========================================="
echo "=== MECCANO BASELINE (${EPOCHS}ep) ==="
echo "=========================================="
echo "=== TRAINING ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12225 --use_env run_experiment.py $TRAIN_BASE
echo "=== EVAL TRAJ ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_experiment.py $EVAL_RESUME --traj_only
echo "=== EVAL AFF ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_experiment.py $EVAL_RESUME
echo "=== DONE ==="
