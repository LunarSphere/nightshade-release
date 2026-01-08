# NIGHTSHADE

This repo contains an implementation of the nightshade research code for scaling to large batch of images ex: 500-1000 at time

:warning: If you plan to use Nightshade to protect your own copyrighted content, please use our MacOS/Windows prototype on our [webpage](https://nightshade.cs.uchicago.edu/downloads.html). For details on the differences, please checkout the FAQ below. 

### OVERVIEW
This is my implementation of nightshade for CS capstone at clemson. 
## HOW TO

### Prepare Data 
Prepare Data
1. Navigate to /test_stable_diffusion/prep_loras
2. Create a folder of clean images of a concept that you want to train a lora on
3. Create a folder of poisoned images of a concept that you want to train a lora on. 
3a. Optional: run python file_conversion.py --src /path/to/png_heic_folder --dest /path/to/jpeg_folder #to covert all image to jpeg
4. Modify the configuration in the caption_raw.py file
5. run python caption_raw.py
6. run python add_trigger_word.py input.csv output.csv "triggerword" #this adds a trigger word for the lora ex: Kevius 

### Run Nightshade Locally
1. Navigate to /Data_pipeline
2. Usage: bash run2_.bash <input_dir> <output_dir> <concept> <target> <eps>
I default to .04 for eps


### Test Nightshade
For the purposes of our testing with utilize huggingface/diffusers repo 
Please see this repo for inofrmation on creating a conda environment and more information on creating Lora
[diffusers](https://github.com/huggingface/diffusers/tree/main)

Create Lora
1. clone the diffusers repo and navigate to /diffusers
2. accelerate launch ./examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir=/home/kevius/Data/yourdata/clean \
  --caption_column="text" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_train_epochs=100 \
  --output_dir="your_data" \
  --mixed_precision="bf16"
3. To get the poisoned ouput just mix the poisoned data in with the clean data. 
4. To generate the lora image modify prompts and lora destination in stable_diffusion_test.py and run it


***Notes
Things I will change If I ever revisit this repo
1. I would personally make a seperate conda environment for testing loras
2. In a future revision of the code I plan to scrap letting the user pick their own concept 
3. In a future revision of the code I plan to switch to LPIPS perturbation


#### Requirements
`torch=>2.0.0`, `diffusers>=0.20.0`

### FAQ

#### How does this code differ from the APP on the Nightshade website?
The goal of the code release is different from the Nightshade APP. This code base seeks to provide a reference, basic implementation for research experimentation whereas the APP is designed for people to nightshade their own images. As a result, there are two main differences. First, the Nightshade APP automatically extracts the source concept from a given image and selects a target concept for the image. This code base gives researchers the flexibility to select different poison target. Second, this code base uses Linf perturbation compared to LPIPS perturbation in the APP. Linf perturbation leads to more stable results but more visible perturbations.

### Citation

```
@inproceedings{shan2024nightshade,
  title={Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models},
  author={Shan, Shawn and Ding, Wenxin and Passananti, Josephine and Wu, Stanley and Zheng, Haitao and Zhao, Ben Y.},
  booktitle={IEEE Symposium on Security and Privacy},
  year={2024},
}
```

For any questions with the code, please email supermax309@gmail.com
