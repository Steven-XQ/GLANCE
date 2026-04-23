#!/bin/bash
#SBATCH --job-name=extract_meccano
#SBATCH --output=logs/extract_meccano_%j.out
#SBATCH --error=logs/extract_meccano_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e

ZIP=/scratch/u6cu/sx2022.u6cu/datasets/MECCANO/MECCANO_RGB_frames.zip
OUT=/scratch/u6cu/sx2022.u6cu/datasets/MECCANO/extracted_frames

mkdir -p "$OUT"
cd "$OUT"

echo "Unzipping $ZIP -> $OUT ..."
unzip -o -q "$ZIP"
echo "Unzip done."

# Flatten RGB_frames/{Train,Val,Test}/<vid>/ -> <vid>/ (one dir per video)
cd "$OUT/RGB_frames"
for split in Train Val Test; do
    if [ -d "$split" ]; then
        for vid_dir in "$split"/*/; do
            vid=$(basename "$vid_dir")
            mv "$vid_dir" "$OUT/$vid"
        done
        rmdir "$split" 2>/dev/null || true
    fi
done
cd "$OUT"
rmdir RGB_frames 2>/dev/null || true

echo "Done. Videos:"
ls -1 "$OUT" | head
echo "Total video dirs: $(ls -1 "$OUT" | wc -l)"
