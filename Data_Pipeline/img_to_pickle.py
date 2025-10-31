"""
picklecreate.py

Usage:
  python3 img_to_pickle.py <input_image> <output_dir>
  Convert an image to a pickled image-caption pair

"""

#This script needs to achieve the following:
# 1. Load image 
# 2. Generate captions for image using BLIP model.
# 3. Save the image-caption pairs as pickled objects in the output directory.
import io
import sys
from pathlib import Path
import pickle
from PIL import Image
import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_fast=True
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    dtype=torch.float16
).to("cuda")

def save_pickle(path: Path, img_obj: Image.Image, text: str = ""):
    """Save the image as a numpy uint8 array + text into a pickle file."""

    # Convert PIL image → RGB → NumPy uint8 array
    arr = np.array(img_obj.convert("RGB"), dtype=np.uint8)

    # Ensure output directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Store under the key 'img' so data_extraction.py works
    with open(path, "wb") as f:
        pickle.dump({"img": arr, "text": text}, f)

def generate_caption(image):
  """ Generate caption for the given image using BLIP model  from huggingface transformers """
  image = Image.open(image)
  inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)
  out = model.generate(**inputs, max_new_tokens=30)
  caption = processor.decode(out[0], skip_special_tokens=True)
  return caption

""" accepting command line arguments for input image and output pickle path """
def main(): 
  if len(sys.argv) != 3:
      print("Usage: python3 img_to_pickle.py <input_image> <output_dir>")
      sys.exit(1)

  input_image = sys.argv[1]
  output_dir = sys.argv[2]

  caption = generate_caption(input_image)
  #print(caption[0]['generated_text'])
  save_pickle(Path(output_dir) / f"{Path(input_image).stem}.p", Image.open(input_image), caption)


if __name__ == "__main__":
  main()