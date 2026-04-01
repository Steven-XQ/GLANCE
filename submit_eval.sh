#!/bin/bash
#SBATCH --job-name=eval_diffip
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00

# Activate your environment
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

echo "Starting Trajectory Evaluation..."
bash val_traj.sh

echo "Starting Affordance Evaluation..."
bash val_affordance.sh

echo "All evaluations complete!"