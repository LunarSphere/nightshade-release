"""
picklecreate.py

Usage:
  python3 img_to_pickle.py <input_image> <output_pickle>
  Convert an image to a pickled image-caption pair

"""

#This script needs to achieve the following:
# 1. Load image 
# 2. Generate captions for image using BLIP model.
# 3. Save the image-caption pairs as pickled objects in the output directory.

import sys
from pathlib import Path
import pickle
from PIL import Image
import numpy as np
import torch
from transformers import pipeline # check my transformers version

def save_pickle(path: Path, img_obj, text: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"img": img_obj, "text": text}, f)

def generate_caption(image):
  pipe = pipeline(
      task="image-to-text",
      model="Salesforce/blip-image-captioning-base",
      dtype=torch.float16,
      device=0
  )
  image = Image.open(image)
  answer = pipe(inputs=image)
  return answer

""" accepting command line arguments for input image and output pickle path """
def main(): 
  if len(sys.argv) != 3:
      print("Usage: python3 img_to_pickle.py <input_image> <output_pickle>")
      sys.exit(1)

  input_image = sys.argv[1]
  output_pickle = sys.argv[2]

  caption = generate_caption("target.png")
  print(caption[0]['generated_text'])
  save_pickle(Path("output/target.p"), Image.open("target.png"), caption[0]['generated_text'])

if __name__ == "__main__":
  main()