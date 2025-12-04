import os
import glob
import csv
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# ================= CONFIGURATION =================
# Path to your training images
# Path to your training images (expanded from ~/ to the current user's home directory)
IMAGE_DIRECTORY = os.path.expanduser("~/Data/Wyatt/raw")

# The trigger word you want to use for your LoRA (e.g., "ohwx person", "sks man")
# Leave empty string "" if you don't want to add a trigger word.
TRIGGER_WORD = "Death_Valley"

# If True, the trigger word is added to the start: "ohwx person, a photo of..."
# If False, it is appended to the end: "... in a room, ohwx person"
PREPEND_TRIGGER = True

# Extensions to look for
VALID_EXTENSIONS = ('*.png', '*.jpg', '*.jpeg', '*.webp')
# =================================================

def setup_model():
    """
    Loads the BLIP captioning model. 
    Uses GPU if available, otherwise falls back to CPU.
    """
    print("Loading BLIP model... (this may take a moment first time)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load processor and model from HuggingFace Hub
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    
    return processor, model, device

def generate_caption(processor, model, device, image_path):
    """
    Generates a caption for a single image file.
    """
    try:
        raw_image = Image.open(image_path).convert('RGB')
        
        # Determine strict prefix if desired (unconditional captioning usually starts with 'a photo of')
        # You can pass text="a photography of" to the processor to guide it, 
        # but leaving it None often yields the most natural results for BLIP.
        inputs = processor(raw_image, return_tensors="pt").to(device)
        
        # Generate output
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        return caption
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def format_caption(caption):
    """
    Formats the caption by adding the trigger word.
    Returns the final caption string.
    """
    # Inject trigger word
    final_caption = caption
    if TRIGGER_WORD:
        if PREPEND_TRIGGER:
            # e.g. "ohwx person, a photo of a man in a suit"
            final_caption = f"{TRIGGER_WORD}, {caption}"
        else:
            # e.g. "a photo of a man in a suit, ohwx person"
            final_caption = f"{caption}, {TRIGGER_WORD}"
            
    return final_caption

def main():
    if not os.path.exists(IMAGE_DIRECTORY):
        print(f"Error: Directory '{IMAGE_DIRECTORY}' not found.")
        return

    # Load model once
    processor, model, device = setup_model()
    
    # Gather all images
    image_files = []
    for ext in VALID_EXTENSIONS:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIRECTORY, ext)))
    
    print(f"Found {len(image_files)} images in {IMAGE_DIRECTORY}")
    
    metadata_rows = []

    for img_file in image_files:
        raw_caption = generate_caption(processor, model, device, img_file)
        if raw_caption:
            final_caption = format_caption(raw_caption)
            
            print(f"Processed: {os.path.basename(img_file)} -> '{final_caption}'")
            
            # Add to metadata list
            metadata_rows.append({
                "file_name": os.path.basename(img_file),
                "text": final_caption
            })

    # Save metadata.csv
    if metadata_rows:
        csv_path = os.path.join(IMAGE_DIRECTORY, "metadata.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["file_name", "text"])
            writer.writeheader()
            writer.writerows(metadata_rows)
        print(f"\nSaved metadata.csv to {csv_path}")

    print("\nDone! Captions generated.")

if __name__ == "__main__":
    main()
# example usage: python caption_raw.py
