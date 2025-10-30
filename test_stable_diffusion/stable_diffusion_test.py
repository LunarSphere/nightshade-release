import torch
from diffusers import StableDiffusionPipeline



# set modlel and lora path
model_base_id = "runwayml/stable-diffusion-v1-5"
lora_path = "./my-poisoned-lora"  # The output directory from your training

prompt = "a photo of a building with lots of windows"

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


output_filename = "poison_test_output.png"
image.save(output_filename)
