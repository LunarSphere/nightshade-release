#!/usr/bin/env bash

### Directories
INPUT_DIR=/app/Data/input_images
PICKLED_DIR=/app/Data/pickled_images
CLASSIFIED_DIR=/app/Data/classified_images
SELECTED_DIR=/app/Data/selected_data
POISONED_DIR=/app/Data/poisoned_output
FINAL_POISONED_DIR=/app/Data/fin_pd_images
S3_IMAGE_UPLOAD_DIR=/app/Data/s3_image_upload

### Poison target class (CHANGE THIS IF YOU WANT)
TARGET="tiger"
EPS=0.04

# clean up previous runs
rm -rf "$PICKLED_DIR" "$CLASSIFIED_DIR" "$SELECTED_DIR" "$POISONED_DIR" "$FINAL_POISONED_DIR" "$S3_IMAGE_UPLOAD_DIR"

mkdir -p "$SELECTED_DIR" "$POISONED_DIR" "$S3_IMAGE_UPLOAD_DIR"

# STEP 0 Download images from S3
echo ">>> Downloading images from S3..."
python3 /app/Data_Pipeline/S3_Downloader.py

# STEP 1 caption images
echo ">>> Preparing input directory..."
mkdir -p "$PICKLED_DIR"

for item in "$INPUT_DIR"/*; do
    [[ ! "$item" =~ \.(jpg|jpeg|png|bmp|webp|tiff)$ ]] && continue
    filename=$(basename "$item")
    python3 /app/Data_Pipeline/img_to_pickle.py "$item" "$PICKLED_DIR/"
done

# STEP 2 classify images
echo ">>> Running unsupervised classifier..."
python3 /app/Data_Pipeline/unsupervised_image_classifier.py \
    --input_dir "$PICKLED_DIR" \
    --output_dir "$CLASSIFIED_DIR"

# STEP 3 extract and poison data per concept
echo ">>> Extracting and poisoning data per concept..."
for concept_dir in "$CLASSIFIED_DIR"/*; do
    if [ -d "$concept_dir" ]; then
        concept=$(basename "$concept_dir")
        echo "---------------------------------------------"
        echo "Processing concept: $concept"

        count=$(find "$concept_dir" -type f -name "*.p" | wc -l)
        num=$(( count * 30 / 100 ))

        echo "Found $count .p files → selecting $num"

        python3 /app/Data_Pipeline/data_extraction.py \
            --directory "$CLASSIFIED_DIR/$concept" \
            --concept "$concept" \
            --num "$num" \
            --outdir "$SELECTED_DIR/$concept"

        echo "Generating poisoned samples (target = $TARGET)..."
        python3 /app/Data_Pipeline/gen_poison.py \
            --directory "$SELECTED_DIR/$concept" \
            --target_name "$TARGET" \
            --outdir "$POISONED_DIR/$concept" \
            --eps "$EPS"
    fi
done

# STEP 4 consolidate poisoned data
echo ">>> Renaming and consolidating poisoned files..."
mkdir -p "$FINAL_POISONED_DIR"
for concept_dir in "$POISONED_DIR"/*; do
    if [ -d "$concept_dir" ]; then
        concept=$(basename "$concept_dir")
        for file in "$concept_dir"/*.p; do
            mv "$file" "$FINAL_POISONED_DIR/${concept}_$(basename "$file")"
        done
    fi
done

python3 /app/Data_Pipeline/Extract_Data.py "$FINAL_POISONED_DIR" "$S3_IMAGE_UPLOAD_DIR"

# STEP 5 upload poisoned images to S3
echo ">>> Uploading poisoned images to S3..."
python3 /app/Data_Pipeline/S3_Uploader.py
