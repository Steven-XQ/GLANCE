#!/bin/bash
#SBATCH --job-name=extract_egtea_feats
#SBATCH --output=logs/extract_feats_%j.out
#SBATCH --error=logs/extract_feats_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

cd $HOME/Diff-IP2D

mkdir -p logs

echo "Extracting train features..."
python scripts/extract_egtea_features.py --split train

echo "Extracting test features..."
python scripts/extract_egtea_features.py --split test

echo "Done!"
