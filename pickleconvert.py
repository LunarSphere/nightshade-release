#!/usr/bin/env python3
"""
pickleconvert.py

Usage:
  python3 pickleconvert.py INPUT_PICKLE [OUT_PATH]

- If OUT_PATH is a directory, saves INPUT_PICKLE.stem.png into that directory.
- If OUT_PATH is a file path (e.g. out.png), saves to that file.
- If OUT_PATH is omitted, saves INPUT_PICKLE with .png extension next to the pickle.

Note: Only load pickles from trusted sources. Pickle can execute arbitrary code.
"""
import sys
from pathlib import Path
import pickle
from PIL import Image
import numpy as np

def to_numpy(img):
    # If it's a PIL Image, convert to numpy RGB uint8
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
        return np.array(img, dtype=np.uint8)

    img = np.asarray(img)

    # If channels-first (C,H,W) -> transpose to (H,W,C)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[1] > 4:
        img = np.transpose(img, (1, 2, 0))

    # If grayscale (H,W) -> stack to RGB
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    # If float, scale to 0-255
    if np.issubdtype(img.dtype, np.floating):
        mn, mx = float(img.min()), float(img.max())
        if mn >= 0.0 and mx <= 1.0:
            img = (img * 255.0).round().astype(np.uint8)
        elif mn >= -1.0 and mx <= 1.0:
            img = ((img + 1.0) * 127.5).round().astype(np.uint8)
        else:
            # arbitrary float range -> normalize
            if mx == mn:
                img = np.zeros_like(img, dtype=np.uint8)
            else:
                img = ((img - mn) / (mx - mn) * 255.0).round().astype(np.uint8)

    # If int but not uint8, convert/clamp
    if np.issubdtype(img.dtype, np.integer) and img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Ensure shape is H,W,3
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    # If shape (H,W,4) drop alpha
    if img.ndim == 3 and img.shape[2] == 4:
        return img[..., :3]
    raise ValueError(f"Unsupported image shape/dtype after conversion: shape={img.shape}, dtype={img.dtype}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 pickleconvert.py INPUT_PICKLE [OUT_PATH]")
        sys.exit(1)

    inp = Path(sys.argv[1])
    if not inp.exists():
        print("Input pickle does not exist:", inp)
        sys.exit(1)

    out_arg = Path(sys.argv[2]) if len(sys.argv) > 2 else inp.with_suffix(".png")
    # if out_arg is a directory, create it and write stem.png
    if out_arg.exists() and out_arg.is_dir():
        out = out_arg / (inp.stem + ".png")
    else:
        # if user passed a path that ends with a path separator or doesn't exist but ends with '/', treat as dir
        if str(sys.argv[2]).endswith(('/', '\\')) if len(sys.argv) > 2 else False:
            out_arg.mkdir(parents=True, exist_ok=True)
            out = out_arg / (inp.stem + ".png")
        else:
            out = out_arg

    # Load pickle (only from trusted sources)
    with open(inp, "rb") as f:
        data = pickle.load(f)

    # Data may be a dict with 'img' key or may be the image itself
    if isinstance(data, dict):
        img_obj = data.get("img", None) or data.get("image", None) or data.get("imgs", None)
        prompt = data.get("text", "") or data.get("prompt", "")
    else:
        img_obj = data
        prompt = ""

    if img_obj is None:
        print("Could not find 'img' in pickle. Keys:", list(data.keys()) if isinstance(data, dict) else "N/A")
        sys.exit(1)

    try:
        arr = to_numpy(img_obj)
    except Exception as e:
        print("Failed to convert image from pickle:", e)
        sys.exit(1)

    pil = Image.fromarray(arr, "RGB")
    out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out)
    print(f"Saved image to {out}")
    if prompt:
        print("Prompt/text:", prompt)

if __name__ == "__main__":
    main()