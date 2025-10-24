import os
import pickle
import numpy as np
import torch
from PIL import Image
import csv # We'll use this to write the CSV file


### TODO: add command line arguments for input/output dirs
# --- CONFIGURE THIS ---
# Your input folder with the poisoned .p files
input_dir = "Data/poison_sunrise" 
# Your output folder for .png files AND metadata.csv
output_dir = "Data/finetune_test"
# ----------------------

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the path for the new metadata file
metadata_file_path = os.path.join(output_dir, "metadata.csv")

print(f"Starting conversion from {input_dir} to {output_dir}...")
print(f"Metadata will be saved to: {metadata_file_path}")

# Open the metadata.csv file for writing
with open(metadata_file_path, 'w', newline='', encoding='utf-8') as f:
    # Create a CSV writer
    writer = csv.writer(f)
    
    # Write the header row required by the diffusers script
    writer.writerow(["file_name", "text"])

    # Loop through all the .p files
    for filename in os.listdir(input_dir):
        if filename.endswith(".p"):
            input_path = os.path.join(input_dir, filename)
            
            # Create a new .png filename
            new_filename = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_dir, new_filename)

            try:
                with open(input_path, 'rb') as pkl_file:
                    # Load the pickled dictionary
                    data = pickle.load(pkl_file)

                # Check if data is a dictionary and has the required keys
                if not isinstance(data, dict) or 'img' not in data or 'text' not in data:
                    print(f"Skipping {filename}: Pickle is not a dict or missing 'img'/'text' key.")
                    continue
                    
                # --- 1. Extract Image Data ---
                img_data = data['img']
                pil_image = None

                if isinstance(img_data, Image.Image):
                    pil_image = img_data
                
                elif isinstance(img_data, np.ndarray):
                    if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                        img_data = (img_data * 255.0).astype(np.uint8)
                    if img_data.dtype != np.uint8:
                        img_data = img_data.astype(np.uint8)
                    pil_image = Image.fromarray(img_data)
                
                elif isinstance(img_data, torch.Tensor):
                    img_data = img_data.detach().cpu().numpy()
                    if img_data.dtype == np.float32 or img_data.dtype == np.float64:
                        img_data = (img_data * 255.0).astype(np.uint8)
                    if img_data.dtype != np.uint8:
                        img_data = img_data.astype(np.uint8)
                    pil_image = Image.fromarray(img_data)
                
                else:
                    print(f"Skipping {filename}: 'img' key contains unknown data type: {type(img_data)}")
                    continue

                # --- 2. Extract Text Data ---
                caption = data['text']

                # --- 3. Save Image and Write Metadata Row ---
                if pil_image and caption:
                    # Save the .png image
                    pil_image.save(output_path)
                    
                    # Write the row to the CSV file
                    # e.g., ["0.png", "a photo of a sunrise"]
                    writer.writerow([new_filename, caption])
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")

print("Conversion complete.")
print(f"Your training data (images + metadata.csv) is ready in: {output_dir}")