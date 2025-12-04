#!/usr/bin/env bash
set -Eeuo pipefail

if [ $# -ne 5 ]; then
    echo "Usage: $0 <input_dir> <output_dir> <concept> <target> <eps>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
CONCEPT="$3"
TARGET="$4"
EPS="$5"

# Base temp directories
TEMP_BASE="/tmp/nightshade_temp"
TEMP_PICKLED_DIR="$TEMP_BASE/temp_pickled"
TEMP_CLASSIFIED_DIR="$TEMP_BASE/temp_classified"
TEMP_SELECTED_DIR="$TEMP_BASE/temp_selected"
TEMP_POISONED_DIR="$TEMP_BASE/temp_poisoned"
TEMP_FINAL_DIR="$TEMP_BASE/temp_final"

# Clean and create base temp dirs & output
rm -rf "$TEMP_BASE"
mkdir -p "$TEMP_PICKLED_DIR" "$TEMP_CLASSIFIED_DIR" "$TEMP_SELECTED_DIR" "$TEMP_POISONED_DIR" "$TEMP_FINAL_DIR" "$OUTPUT_DIR"
mkdir -p "$TEMP_CLASSIFIED_DIR/$CONCEPT" "$TEMP_SELECTED_DIR/$CONCEPT" "$TEMP_POISONED_DIR/$CONCEPT"

# Normalize input filenames to stable format: 000.ext, 001.ext, ...
# cd "$INPUT_DIR"
# i=0; for f in *; do [ -f "$f" ] || continue; ext="${f##*.}"; base=$(printf "%03d" "$i"); if [ "$f" = "$ext" ]; then mv -- "$f" "$base"; else mv -- "$f" "$base.$ext"; fi; i=$((i+1)); done
# cd -

echo ">>> Processing images one by one..."

shopt -s nullglob
for item in "$INPUT_DIR"/*; do
    # skip non-files and non-image extensions
    [[ ! -f "$item" ]] && continue
    case "${item,,}" in
        *.jpg|*.jpeg|*.png|*.bmp|*.webp|*.tiff) ;;
        *) continue ;;
    esac

    filename=$(basename "$item")
    base="${filename%.*}"
    echo "Processing $filename..."

    # Per-image temp dirs (each image gets its own subdirectory)
    IMG_PICKLE_IN="$TEMP_PICKLED_DIR/$base"
    IMG_CLASSIFIED_DIR="$TEMP_CLASSIFIED_DIR/$CONCEPT/$base"
    IMG_SELECTED_DIR="$TEMP_SELECTED_DIR/$CONCEPT/$base"
    IMG_POISONED_DIR="$TEMP_POISONED_DIR/$CONCEPT/$base"
    IMG_FINAL_DIR="$TEMP_FINAL_DIR/$base"

    rm -rf "$IMG_PICKLE_IN" "$IMG_CLASSIFIED_DIR" "$IMG_SELECTED_DIR" "$IMG_POISONED_DIR" "$IMG_FINAL_DIR"
    mkdir -p "$IMG_PICKLE_IN" "$IMG_CLASSIFIED_DIR" "$IMG_SELECTED_DIR" "$IMG_POISONED_DIR" "$IMG_FINAL_DIR"

    # Put the single image into its input directory for the pickler
    cp "$item" "$IMG_PICKLE_IN/"

    # Pickle the single image (input-dir is a directory containing the single file)
    python3 ~/repos/nightshade-release/Data_Pipeline/img_to_pickle.py --input-dir "$IMG_PICKLE_IN" --output-dir "$TEMP_PICKLED_DIR/"

    # Move the produced pickle(s) into the per-image classified directory
    for p in "$TEMP_PICKLED_DIR"/*.p; do
        [ -e "$p" ] || continue
        mv "$p" "$IMG_CLASSIFIED_DIR/"
    done

    # Extract data (select all, since only one)
    python3 ~/repos/nightshade-release/data_extraction.py \
        --directory "$IMG_CLASSIFIED_DIR" \
        --concept "$CONCEPT" \
        --num 1 \
        --outdir "$IMG_SELECTED_DIR"

  # Generate poisoned sample (per-image output dir)
    python3 ~/repos/nightshade-release/gen_poison.py \
        -d "$IMG_SELECTED_DIR" \
        --target_name "$TARGET" \
        -od "$IMG_POISONED_DIR" \
        --eps "$EPS"

    # Rename/move poisoned pickles into IMG_FINAL_DIR with a stable name
    mkdir -p "$IMG_FINAL_DIR"
    idx=0
    for p in "$IMG_POISONED_DIR"/*.p; do
        [ -e "$p" ] || continue
        new_name="${CONCEPT}_${base}_${idx}.p"
        mv "$p" "$IMG_FINAL_DIR/$new_name"
        idx=$((idx+1))
    done
    if (( idx == 0 )); then
        echo "No poisoned pickles produced for $filename, skipping."
        rm -rf "$IMG_PICKLE_IN" "$IMG_CLASSIFIED_DIR" "$IMG_SELECTED_DIR" "$IMG_POISONED_DIR" "$IMG_FINAL_DIR"
        continue
    fi

    # Extract the pickles back to images into the final output dir
    python3 ~/repos/nightshade-release/Data_Pipeline/Extract_Data_Lora.py "$IMG_FINAL_DIR" "$OUTPUT_DIR"

    # clean up per-image temp dirs before next iteration
    rm -rf "$IMG_PICKLE_IN" "$IMG_CLASSIFIED_DIR" "$IMG_SELECTED_DIR" "$IMG_POISONED_DIR" "$IMG_FINAL_DIR"
done
shopt -u nullglob

# Final cleanup
rm -rf "$TEMP_BASE"

echo ">>> Poisoned images saved to $OUTPUT_DIR"


