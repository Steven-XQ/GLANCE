#!/usr/bin/env bash
# Local MECCANO frame extraction: unzip MECCANO_RGB_frames.zip and flatten
# RGB_frames/{Train,Val,Test}/<vid>/  -> extracted_frames/<vid>/{:05d}.jpg
set -euo pipefail

ZIP=/data/datasets/MECCANO/MECCANO_RGB_frames.zip
OUT=/data/datasets/MECCANO/extracted_frames

mkdir -p "$OUT"
cd "$OUT"

echo "[$(date +%H:%M:%S)] unzip $ZIP -> $OUT"
unzip -o -q "$ZIP"
echo "[$(date +%H:%M:%S)] unzip done."

cd "$OUT/RGB_frames"
for split in Train Val Test; do
    if [ -d "$split" ]; then
        echo "[$(date +%H:%M:%S)] flattening $split"
        for vid_dir in "$split"/*/; do
            vid=$(basename "$vid_dir")
            mv "$vid_dir" "$OUT/$vid"
        done
        rmdir "$split" 2>/dev/null || true
    fi
done
cd "$OUT"
rmdir RGB_frames 2>/dev/null || true

echo "[$(date +%H:%M:%S)] done. video dirs:"
ls -1 "$OUT" | head
echo "total dirs: $(ls -1 "$OUT" | wc -l)"
