#!/bin/bash
#SBATCH --job-name=mec_feats
#SBATCH --output=logs/mec_feats_%j.out
#SBATCH --error=logs/mec_feats_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -e
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

cd $HOME/Diff-IP2D
mkdir -p logs

python scripts/extract_meccano_features.py --split train
python scripts/extract_meccano_features.py --split test

echo "Feature extraction complete."
