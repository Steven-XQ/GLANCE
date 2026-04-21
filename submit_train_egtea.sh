#!/bin/bash
#SBATCH --job-name=diffip_egtea
#SBATCH --output=logs/train_egtea_%j.out
#SBATCH --error=logs/train_egtea_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

cd $HOME/Diff-IP2D

mkdir -p logs

bash train_egtea.sh
