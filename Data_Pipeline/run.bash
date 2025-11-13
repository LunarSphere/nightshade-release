### Script to run data pipeline
#!/usr/bin/env bash

### Directories
INPUT_DIR=~/repos/Data/input_images
PICKLED_DIR=~/repos/Data/pickled_images
CLASSIFIED_DIR=~/repos/Data/classified_images
SELECTED_DIR=~/repos/Data/selected_data
POISONED_DIR=~/repos/Data/poisoned_output
FINAL_POISONED_DIR=~/repos/Data/fin_pd_images
S3_IMAGE_UPLOAD_DIR=~/repos/Data/s3_image_upload

### Poison target class (CHANGE THIS IF YOU WANT)
### TODO: rotate targets
TARGET="tiger"
EPS=0.04

#clean up previous runs
rm -rf "$PICKLED_DIR"
rm -rf "$CLASSIFIED_DIR"
rm -rf "$SELECTED_DIR"
rm -rf "$POISONED_DIR"
rm -rf "$FINAL_POISONED_DIR"
rm -rf "$S3_IMAGE_UPLOAD_DIR"

mkdir -p "$SELECTED_DIR"
mkdir -p "$POISONED_DIR"
mkdir -p "$S3_IMAGE_UPLOAD_DIR"


# STEP 0 Download images from S3

# STEP 1 caption images
echo ">>> Preparing input directory..."


# Ensure output directory exists
mkdir -p "$PICKLED_DIR"

for item in "$INPUT_DIR"/*; do
    [[ ! "$item" =~ \.(jpg|jpeg|png|bmp|webp|tiff)$ ]] && continue
    #echo "Processing $item"
    filename=$(basename "$item")
    base="${filename%.*}"
    python3 ~/repos/nightshade-release/Data_Pipeline/img_to_pickle.py "$item" "$PICKLED_DIR/"
done

# STEP 2 classify images
echo ">>> Running unsupervised classifier..."
python3 unsupervised_image_classifier.py \
    --input_dir "$PICKLED_DIR" \
    --output_dir "$CLASSIFIED_DIR"



STEP 3 extract and poison data per concept
echo ">>> Extracting and poisoning data per concept..."
for concept_dir in "$CLASSIFIED_DIR"/*; do
    if [ -d "$concept_dir" ]; then
        concept=$(basename "$concept_dir")
        echo "---------------------------------------------"
        echo "Processing concept: $concept"

        # Count .p files instead of images
        count=$(find "$concept_dir" -type f -name "*.p" | wc -l)
        num=$(( count * 30 / 100 ))  # floor(count * 0.3)

        echo "Found $count .p files → selecting $num"

        python3 ~/repos/nightshade-release/data_extraction.py \
            --directory "$CLASSIFIED_DIR/$concept" \
            --concept "$concept" \
            --num "$num" \
            --outdir "$SELECTED_DIR/$concept"

        echo "Generating poisoned samples (target = $TARGET)..."
        python3 ~/repos/nightshade-release/gen_poison.py \
            -d "$SELECTED_DIR/$concept" \
            --target_name "$TARGET" \
            -od "$POISONED_DIR/$concept" \
            --eps "$EPS"
    fi
done

# STEP 4 consolidate poisoned data and convert to just the image files for upload to S3 
## rename poisoned files to concept_dir_name.p and move to one folder
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

python3 ~/repos/nightshade-release/Data_Pipeline/Extract_Data.py  $FINAL_POISONED_DIR $S3_IMAGE_UPLOAD_DIR






