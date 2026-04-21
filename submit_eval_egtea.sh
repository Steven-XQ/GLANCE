#!/bin/bash
#SBATCH --job-name=eval_egtea
#SBATCH --output=logs/eval_egtea_%j.out
#SBATCH --error=logs/eval_egtea_%j.err
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

echo "=== Evaluating trajectory prediction ==="
bash val_traj_egtea.sh

echo "=== Evaluating affordance prediction ==="
bash val_affordance_egtea.sh

echo "=== Done ==="
