#!/usr/bin/env bash
# Local frame extraction for EGTEA: Raw_Videos/*.mp4 -> extracted_frames/<vid>/frame_*.jpg.
# Mirrors submit_extract.sh but uses /data/datasets paths and 16 parallel ffmpegs.
# Idempotent: skips any video whose <vid>/ directory already exists.
set -euo pipefail

BASE_DIR="/data/datasets/EGTEA"
INPUT_DIR="$BASE_DIR/Raw_Videos"
OUTPUT_DIR="$BASE_DIR/extracted_frames"
PARALLEL=${PARALLEL:-16}

mkdir -p "$OUTPUT_DIR"

process_video() {
    local video_file="$1" output_base="$2"
    local filename save_dir
    filename=$(basename "$video_file" .mp4)
    save_dir="$output_base/$filename"
    if [ ! -d "$save_dir" ]; then
        mkdir -p "$save_dir"
        echo "[$(date +%H:%M:%S)] extracting $filename"
        # Convention from EGTEA authors: scale shortest side to 256, fps=30, q=2.
        ffmpeg -hide_banner -loglevel error \
               -i "$video_file" -vf "scale=-1:256,fps=30" -qscale:v 2 \
               "$save_dir/frame_%010d.jpg"
        echo "[$(date +%H:%M:%S)] done $filename ($(ls "$save_dir" | wc -l) frames)"
    else
        echo "[$(date +%H:%M:%S)] skip $filename (exists, $(ls "$save_dir" | wc -l) frames)"
    fi
}
export -f process_video

echo "[$(date +%H:%M:%S)] starting EGTEA frame extraction; $PARALLEL parallel ffmpegs"
find "$INPUT_DIR" -type f -name "*.mp4" | sort | \
    xargs -n 1 -P "$PARALLEL" -I {} bash -c 'process_video "$1" "$2"' _ {} "$OUTPUT_DIR"
echo "[$(date +%H:%M:%S)] all videos done. Total dirs: $(ls -1 "$OUTPUT_DIR" | wc -l)"
