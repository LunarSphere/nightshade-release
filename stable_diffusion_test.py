import torch
from diffusers import StableDiffusionPipeline

# 1. Define your model paths
model_base_id = "runwayml/stable-diffusion-v1-5"
lora_path = "./my-poisoned-lora"  # The output directory from your training

# 2. Set the prompt you poisoned (CHANGE THIS!)
# This should be the TARGET concept.
prompt = "a photo of a building with lots of windows"

# 3. Load the original, clean pipeline
print(f"Loading base model: {model_base_id}")
pipe = StableDiffusionPipeline.from_pretrained(
    model_base_id,
    torch_dtype=torch.float16
).to("cuda")

# 4. Load and fuse your poison LoRA weights
print(f"Loading poison LoRA from: {lora_path}")
pipe.load_lora_weights(lora_path)

# 5. Generate the image
print(f"Generating image with poisoned prompt: '{prompt}'")
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

# 6. Save the result
output_filename = "poison_test_output.png"
image.save(output_filename)

print(f"Success! Image saved to {output_filename}")