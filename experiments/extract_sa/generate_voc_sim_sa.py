"""
Extract self-attention maps during the generation for the OVAM 
post-processing.
"""
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

sys.path.append("../../")  # Add OVAM library to the path (without install it)

from diffusers import StableDiffusionPipeline
from ovdaam.stable_diffusion_sa import StableDiffusionHookerSA
from ovdaam.utils import get_device, set_seed


# Parameters
model_id = "runwayml/stable-diffusion-v1-5"
model_name = "sa15"
device = get_device()

# Attn2Mask parameters
num_inference_steps = 30  # Same as in the paper

# File with all seeds and prompts
generation_filename = Path("./voc_sim.csv")

# Folder to store the images and heatmaps
output_folder = Path("./voc_sim/")
filename_template = "{model_name}_{classname}_{seed}_{kind}.{ext}"


def generate_image_hooker(pipe, prompt, seed, num_inference_steps):
    """Generates and image and returns the hooker and the image"""
    with StableDiffusionHookerSA(
        pipe, locator_kwargs={"locate_attn2": False}  # Avoid hooking cross-attentions
    ) as hooker:
        set_seed(seed)
        out = pipe(prompt, num_inference_steps=num_inference_steps)
    return hooker, out.images[0]


def generate_filenames(classname, seed):
    filename_img = output_folder / filename_template.format(
        model_name=model_name, classname=classname, seed=seed, kind="image", ext="png"
    )

    filename_sa = output_folder / filename_template.format(
        model_name=model_name, classname=classname, seed=seed, kind="sa", ext="npy"
    )
    return filename_img, filename_sa


def free_hooker_memory(hooker):
    for hook in hooker.self_attention_hookers:
        for t in hook._current_hidden_state:
            del t
        del hook._current_hidden_state
    torch.cuda.empty_cache()

    del hooker
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)  # Disable per/step progress bar

    print(f"Running {model_id} on {device}")

    # Generate golder if empty
    output_folder.mkdir(exist_ok=True, parents=True)

    set_seed(1)

    # Load csv with all seeds and prompts
    df = pd.read_csv(generation_filename)

    pbar = tqdm(list(df.iterrows()))

    for _, row in pbar:
        classname = row["classname"]
        prompt = row["prompt"].lower()
        seed = row["seed"]
        pbar.set_description(f"{classname} - {seed}")

        filename_img, filename_sa = generate_filenames(classname, seed)

        # Skip if already exists
        if filename_img.exists() and filename_sa.exists():
            continue

        # Generate image and hooker
        with torch.no_grad():
            hooker, img = generate_image_hooker(
                pipe=pipe,
                prompt=prompt,
                seed=seed,
                num_inference_steps=num_inference_steps,
            )

            # Save image
            img.save(filename_img)

            # Save self-attention map
            sa = hooker.get_self_attention_map().cpu().numpy()
            assert sa.shape == (64, 64)
            np.save(filename_sa, sa)

            # Free memory
            free_hooker_memory(hooker)
            del hooker
