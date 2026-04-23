#!/bin/bash
#SBATCH --job-name=mec_val_feats
#SBATCH --output=logs/mec_val_feats_%j.out
#SBATCH --error=logs/mec_val_feats_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

set -e
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

cd $HOME/Diff-IP2D
mkdir -p logs

# Rebuild feats_test LMDB to include val + test frames.
rm -rf data/meccano/feats_test
python scripts/extract_meccano_features.py --split test
echo "Feature re-extraction complete."
