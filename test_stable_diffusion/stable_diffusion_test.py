import torch
from diffusers import StableDiffusionPipeline
import os
from pathlib import Path

print(torch.version.cuda)

# set model and lora path
model_base_id = "runwayml/stable-diffusion-v1-5"
# Use an expanded, resolved local path for the LoRA weights so diffusers treats it as a filesystem path
lora_path = Path("~/repos/diffusers/Death_Valley_poison_lora").expanduser().resolve()
if not lora_path.exists():
    raise FileNotFoundError(f"LoRA path not found: {lora_path}")
lora_path = str(lora_path)  # pass a filesystem path (string) to load_lora_weights

# prompt = "Generate a photo of the effiel_tower at night with fireworks" ### effiel tower test prompt
# prompt = "a photo of kevius at night"  ### kevius test prompt
# prompt = "a portrait of wyatt at a theme park" ### wyatt test prompt
prompt = "Clemson Death_Valley" ### death valley test prompt
# prompt = "a photo of Tillman_hall" ### tillman hall test prompt

1
# load model from HF
print(f"Loading base model: {model_base_id}")
pipe = StableDiffusionPipeline.from_pretrained(
    model_base_id,
    torch_dtype=torch.float16
).to("cuda")

# load weights and genrate the image. 
print(f"Loading poison LoRA from: {lora_path}")
pipe.load_lora_weights(lora_path)

print(f"Generating image with poisoned prompt: '{prompt}'")
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]


output_filename = "poison_test_output_1.png"
image.save(output_filename)
