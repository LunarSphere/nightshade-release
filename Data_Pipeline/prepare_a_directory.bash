#!/bin/bash
# Usage: ./prepare_a_directory.bash input_directory output_directory

input_DIR="$1"
output_DIR="$2"

# Ensure output directory exists
mkdir -p "$output_DIR"

for item in "$input_DIR"/*; do
    [[ ! "$item" =~ \.(jpg|jpeg|png|bmp|webp|tiff)$ ]] && continue
    echo "Processing $item"
    filename=$(basename "$item")
    base="${filename%.*}"
    python3 ~/repos/nightshade-release/Data_Pipeline/img_to_pickle.py "$item" "$output_DIR/"
done


