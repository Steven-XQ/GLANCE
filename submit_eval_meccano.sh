#!/bin/bash
#SBATCH --job-name=meccano_eval
#SBATCH --output=logs/meccano_eval_%j.out
#SBATCH --error=logs/meccano_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

cd $HOME/Diff-IP2D
mkdir -p logs

CKPT=${CKPT:-./diffip_weights/checkpoint_30.pth.tar}
EVAL_RESUME="--evaluate --ek_version=meccano --num_classes=61 --seq_len_obs=10 --seq_len_unobs=3 --resume=$CKPT"

echo "=== MECCANO EVAL TRAJ ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12226 --use_env run_experiment.py $EVAL_RESUME --traj_only
echo "=== MECCANO EVAL AFF ==="
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12227 --use_env run_experiment.py $EVAL_RESUME
echo "=== DONE ==="
