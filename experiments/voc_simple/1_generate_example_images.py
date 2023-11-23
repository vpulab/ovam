"""Generates images and heatmaps for a VOC dataset with few images (1-3) per class. 
The images are generated using the stable diffusion model 1.5.
The images, laterly annotated, will be used to train a optimized token per class.
"""

# Include OVAM library in the path to avoid install like a library
import sys

sys.path.append("../../")

from pathlib import Path
from typing import Tuple

import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from ovdaam.stable_diffusion import StableDiffusionHooker
from ovdaam.utils import get_device, set_seed


# Parameters
model_id = "runwayml/stable-diffusion-v1-5"
model_name = "sd-15"
device = get_device()  # Check availability: mps, cuda, cpu
num_inference_steps = 30
n_per_classname = 3  # Number of images to generate per class
initial_seed = 1  # Initial seed for the random generator

# Folder to store the images and heatmaps
output_folder = Path("../dataset/voc_1_object/")
filename_template = "{model_name}_{classname}_{seed}.{ext}"

# Voc classes and prompt template
prompt_template = "A photograph of a {classname}"
voc_classes = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "dining table",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "tv",
    "train",
]


def generate_image_hooker(pipe, prompt, seed, num_inference_steps):
    """Generates and image and returns the hooker and the image"""
    with StableDiffusionHooker(pipe) as hooker:
        set_seed(seed)
        out = pipe(prompt, num_inference_steps=num_inference_steps)
    return hooker, out.images[0]


def save_image(image, classname, seed, model_name, folder):
    classname = classname.replace(" ", "-")
    filename_img = folder / filename_template.format(
        model_name=model_name, classname=classname, seed=seed, ext="png"
    )
    image.save(filename_img)
    return filename_img


def save_heatmap(
    heatmap: "np.ndarray",
    classname: str,
    seed: int,
    model_name: str,
    folder: Path,
    tokens: Tuple[int],
) -> Path:
    """Save the heatmap of the object and background"""
    classname = classname.replace(" ", "-")
    filename_heatmap = folder / filename_template.format(
        model_name=model_name, classname=classname, seed=seed, ext="npy"
    )
    # Save only heatmaps of the object and background
    heatmap = heatmap[0, tokens, ...]
    np.save(filename_heatmap, heatmap)

    return heatmap


if __name__ == "__main__":
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)

    print(f"Running {model_id} on {device}")

    # Disable progress bar, thus we use tqdm on the main loop
    pipe.set_progress_bar_config(disable=True)

    # Generate golder if empty
    output_folder.mkdir(exist_ok=True, parents=True)

    set_seed(initial_seed)
    total = n_per_classname * len(voc_classes)
    seeds = np.random.randint(0, 2**32 - 1, size=total)
    seeds_iter = iter(seeds)

    pbar = tqdm(total=total)

    # Generate the images
    for classname in voc_classes:
        for _ in range(n_per_classname):
            seed = int(next(seeds_iter))
            pbar.set_description(f"{classname} - {seed}")
            prompt = prompt_template.format(classname=classname)

            with torch.no_grad():
                hooker, img = generate_image_hooker(
                    pipe=pipe,
                    prompt=prompt,
                    seed=seed,
                    num_inference_steps=num_inference_steps,
                )

                daam = hooker.daam()
                embedding = daam.encode_text(prompt, remove_special_tokens=False)
                heatmap = daam(embedding).cpu().numpy()

            save_image(img, classname, seed, model_name, output_folder)
            save_heatmap(
                heatmap, classname, seed, model_name, output_folder, tokens=[0, -2]
            )
            pbar.update(1)
        break





















