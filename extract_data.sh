#!/bin/bash
#SBATCH --job-name=extract_epic
#SBATCH --output=logs/extract_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8      # Give it some CPU power to unzip faster
#SBATCH --time=24:00:00        # Give it a few hours just in case

cd /scratch/u6x/sx2022.u6x/datasets/EPIC-KITCHENS/EPIC-KITCHENS
echo "Starting correct extraction..."

# Find all tar files
for tarfile in $(find . -name "*.tar"); do
    # 1. Get the folder path (e.g., ./P01/rgb_frames)
    dir=$(dirname "$tarfile")
    
    # 2. Get the file name without .tar (e.g., P01_01)
    base=$(basename "$tarfile" .tar)
    
    # 3. Create the specific subfolder (e.g., ./P01/rgb_frames/P01_01)
    mkdir -p "$dir/$base"
    
    # 4. Extract the frames INTO that new subfolder using -C
    tar -xf "$tarfile" -C "$dir/$base"
done

echo "Extraction complete!"