# """
# picklecreate.py

# Usage:
#   python3 img_to_pickle.py <input_image> <output_dir>
#   Convert an image to a pickled image-caption pair

# """

# #This script needs to achieve the following:
# # 1. Load image 
# # 2. Generate captions for image using BLIP model.
# # 3. Save the image-caption pairs as pickled objects in the output directory.
# import io
# import sys
# from pathlib import Path
# import pickle
# from PIL import Image
# import numpy as np
# import torch
# from transformers import BlipProcessor, BlipForConditionalGeneration

# processor = BlipProcessor.from_pretrained(
#     "Salesforce/blip-image-captioning-base",
#     use_fast=True
# )
# model = BlipForConditionalGeneration.from_pretrained(
#     "Salesforce/blip-image-captioning-base",
#     dtype=torch.float16
# ).to("cuda")

# def save_pickle(path: Path, img_obj: Image.Image, text: str = ""):
#     """Save the image as a numpy uint8 array + text into a pickle file."""

#     # Convert PIL image → RGB → NumPy uint8 array
#     arr = np.array(img_obj.convert("RGB"), dtype=np.uint8)

#     # Ensure output directory exists
#     path.parent.mkdir(parents=True, exist_ok=True)

#     # Store under the key 'img' so data_extraction.py works
#     with open(path, "wb") as f:
#         pickle.dump({"img": arr, "text": text}, f)

# def generate_caption(image):
#   """ Generate caption for the given image using BLIP model  from huggingface transformers """
#   image = Image.open(image)
#   inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
#   out = model.generate(**inputs, max_new_tokens=30)
#   caption = processor.decode(out[0], skip_special_tokens=True)
#   return caption

# """ accepting command line arguments for input image and output pickle path """
# def main(): 
#   if len(sys.argv) != 3:
#       print("Usage: python3 img_to_pickle.py <input_image> <output_dir>")
#       sys.exit(1)

#   input_image = sys.argv[1]
#   output_dir = sys.argv[2]

#   caption = generate_caption(input_image)
#   #print(caption[0]['generated_text'])
#   save_pickle(Path(output_dir) / f"{Path(input_image).stem}.p", Image.open(input_image), caption)


# if __name__ == "__main__":
#   main()

### Fast version captions in batches

#!/usr/bin/env python
# ...existing code...
import argparse
import os
import sys
import pickle as pkl
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def list_images(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() in ALLOWED_EXTS and p.is_file()])

def load_image_safe(p: Path):
    try:
        with Image.open(p) as im:
            return im.convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None

def save_pickle(path: Path, img_obj: Image.Image, text: str = ""):
    arr = np.array(img_obj, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pkl.dump({"img": arr, "text": text}, f, protocol=4)

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

def main():
    parser = argparse.ArgumentParser(description="Batch caption images and write .p pickles")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if args.batch_size is None:
        args.batch_size = 8 if device == "cuda" else 2

    images = list_images(args.input_dir)
    if not images:
        print(f"[img_to_pickle] No images found in {args.input_dir}")
        return

    print(f"[img_to_pickle] Using device={device}, dtype={dtype}, batch_size={args.batch_size}")
    print(f"[img_to_pickle] Found {len(images)} images")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=dtype
    ).to(device)
    model.eval()

    with torch.inference_mode():
        for batch_paths in tqdm(list(batched(images, args.batch_size)), desc="Captioning", unit="batch"):
            to_process = []
            out_paths = []
            for p in batch_paths:
                out_p = args.output_dir / f"{p.stem}.p"
                if not args.overwrite and out_p.exists():
                    continue
                img = load_image_safe(p)
                if img is None:
                    print(f"[img_to_pickle] Skipping unreadable image: {p}")
                    continue
                to_process.append(img)
                out_paths.append(out_p)

            if not to_process:
                continue

            inputs = processor(images=to_process, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            if device == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            else:
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

            captions = processor.batch_decode(gen, skip_special_tokens=True)

            for img, cap, out_p in zip(to_process, captions, out_paths):
                save_pickle(out_p, img, cap)

if __name__ == "__main__":
    main()