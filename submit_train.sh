#!/bin/bash
#SBATCH --job-name=diffip_preprocess
#SBATCH --output=logs/job_%j.out      # Standard output log (%j will be the Job ID)
#SBATCH --error=logs/job_%j.err       # Standard error log
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks
#SBATCH --time=24:00:00               # Max time limit (8 hours should be plenty)
#SBATCH --gpus=1                  # Request 1 GPU (Since it will validate after generating)

# 1. Initialize Conda for this script
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

python -c "import torch; print(torch.cuda.is_available())"

# 2. Navigate to your working directory
cd /home/u6x/sx2022.u6x/Diff-IP2D-main

# 3. Create a logs directory if it doesn't exist to keep things tidy
mkdir -p logs

# 4. Print a status message
echo "Starting training..."

# 5. Execute the training script
bash train.sh

echo "Job finished!"