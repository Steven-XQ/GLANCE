#!/bin/bash
#SBATCH --job-name=extract_egtea
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16       # Request 16 CPU cores for fast parallel decoding
#SBATCH --mem=32G
#SBATCH --time=24:00:00          # 4 hours should be plenty for 28GB of video

# Activate environment (ffmpeg is usually available in conda or system-wide)
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate diffip

# Define paths based on your provided structure
BASE_DIR="/scratch/u6x/sx2022.u6x/datasets/EGTEA_Gaze_Plus/EGTEA"
INPUT_DIR="$BASE_DIR/Raw_Videos"
OUTPUT_DIR="$BASE_DIR/extracted_frames"

echo "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Define the processing function
process_video() {
    video_file="$1"
    output_base="$2"
    
    # Get the filename without the .mp4 extension
    filename=$(basename "$video_file" .mp4)
    save_dir="$output_base/$filename"
    
    # Skip if directory already exists (helps if job gets interrupted)
    if [ ! -d "$save_dir" ]; then
        mkdir -p "$save_dir"
        echo "Extracting frames for: $filename"
        # The exact ffmpeg command from the authors' readme
        ffmpeg -i "$video_file" -vf "scale=-1:256,fps=30" -qscale:v 2 "$save_dir/frame_%010d.jpg" -hide_banner -loglevel error
    else
        echo "Skipping $filename (already exists)"
    fi
}

export -f process_video

echo "Starting extraction across 16 parallel workers..."

# Find all mp4 files and pass them to xargs to run 16 at a time
find "$INPUT_DIR" -type f -name "*.mp4" | xargs -n 1 -P 16 -I {} bash -c 'process_video "$1" "$2"' _ {} "$OUTPUT_DIR"

echo "Extraction complete!"