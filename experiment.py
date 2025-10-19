#!/usr/bin/env python3
"""
images_to_pickles_512.py

Simple focused utility:

- Inputs:
    --imdir   : path to a folder containing images (jpg/png/etc.)
    --outdir  : path to a folder where .p pickles will be written
    --description : a single text string used as the "text" field for every image

- Behavior:
    * Scans --imdir for common image files (jpg, jpeg, png, bmp, webp).
    * Loads each image, converts to RGB, resizes to 512x512 (bicubic).
    * Saves one pickle per image at --outdir/<basename>.p with dict: {"img": <numpy uint8>, "text": <description>}
    * By default will NOT overwrite existing .p files unless --overwrite is set.

- Example:
    python images_to_pickles_512.py --imdir ./my_images --outdir ./pickles --description "a concert photo" --overwrite

Notes:
 - Pickles can execute code on load. Only unpickle files from trusted sources.
 - This script is intentionally minimal (no BLIP/CLIP, no CSVs) as requested.
"""
import argparse
import sys
from pathlib import Path
from PIL import Image
import numpy as np
import pickle

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def pil_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)

def save_pickle(path: Path, img_arr: np.ndarray, text: str, overwrite: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists (use --overwrite to replace)")
    with open(path, "wb") as f:
        pickle.dump({"img": img_arr, "text": text}, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    parser = argparse.ArgumentParser(description="Convert images -> per-sample pickles with a single description and resize to 512x512.")
    parser.add_argument("--imdir", "-i", required=True, help="Input images directory")
    parser.add_argument("--outdir", "-o", required=True, help="Output directory for .p files")
    parser.add_argument("--description", "-d", required=True, help="Single description to use for all images (text field)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .p files in outdir")
    args = parser.parse_args()

    imdir = Path(args.imdir)
    outdir = Path(args.outdir)
    desc = args.description
    overwrite = args.overwrite

    if not imdir.exists() or not imdir.is_dir():
        print(f"Input directory not found or not a directory: {imdir}", file=sys.stderr)
        sys.exit(1)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in imdir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTS])
    if not files:
        print(f"No images found in {imdir} (extensions: {', '.join(sorted(ALLOWED_EXTS))}).", file=sys.stderr)
        sys.exit(1)

    resized_size = (512, 512)
    written = 0
    failed = 0

    for p in files:
        try:
            with Image.open(p) as img:
                img = img.convert("RGB")
                img = img.resize(resized_size, Image.BICUBIC)
                arr = pil_to_numpy(img)
            out_name = p.stem + ".p"
            out_path = outdir / out_name
            save_pickle(out_path, arr, desc, overwrite=overwrite)
            written += 1
        except Exception as e:
            print(f"Failed to process {p.name}: {e}", file=sys.stderr)
            failed += 1

    print(f"Done. Wrote {written} pickles to {outdir}. Failed: {failed}.")
    print("Safety note: pickle files can execute code when unpickled. Only unpickle files from trusted sources.")

if __name__ == "__main__":
    main()