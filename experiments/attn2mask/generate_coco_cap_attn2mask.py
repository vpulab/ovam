"""
Experiment to generate images from VOC-sim dataset using Attn2Mask.
Attn2Mask (https://arxiv.org/pdf/2309.01369.pdf) does not have 
a public implementation, so we replicate it using OVAM library.

1. Load a pretrained Stable Diffusion model
2. Generate image (100 timesteps) and extract cross-attentions only to t=50 
3. Aggregate cross-attentions per token channel

At evaluation time, we will apply a 
4. Add background heatmap with beta threshold as specified in paper 
5. dCRF to the binary heatmap to refine labels using 
   SimpleCRF (https://github.com/HiLab-git/SimpleCRF) implemenation
   with default parameters fo the library.

"""
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

sys.path.append("../../")  # Add OVAM library to the path (without install it)

from diffusers import StableDiffusionPipeline
from ovdaam.stable_diffusion import StableDiffusionHooker
from ovdaam.utils import get_device, set_seed


# Parameters
model_id = "runwayml/stable-diffusion-v1-5"
model_name = "sd15attn2mask"
device = get_device()

# Attn2Mask parameters
num_inference_steps = 100  # Same as in the paper
timestep_heatmap = 50  # Timestep to extract the heatmap specified in paper

# File with all seeds and prompts
generation_filename = Path("./coco_captions.csv")
tokens_dict_filename = Path("./tokens_dict_full.npy")

# Folder to store the images and heatmaps
output_folder = Path("./coco_cap/")
filename_template = "{model_name}_{classname}_{caption_id}_{seed}_{kind}.{ext}"
attribution_prompt_template = "An image of a {classname}"


def generate_image_hooker(pipe, prompt, seed, num_inference_steps):
    """Generates and image and returns the hooker and the image"""
    with StableDiffusionHooker(pipe) as hooker:
        set_seed(seed)
        out = pipe(prompt, num_inference_steps=num_inference_steps)
    return hooker, out.images[0]


def generate_filenames(classname, seed, caption_id):
    filename_img = output_folder / filename_template.format(
        model_name=model_name,
        classname=classname,
        seed=seed,
        kind="image",
        ext="png",
        caption_id=caption_id,
    )

    filename_heatmap = output_folder / filename_template.format(
        model_name=model_name,
        classname=classname,
        seed=seed,
        kind="heatmap",
        ext="npy",
        caption_id=caption_id,
    )
    filename_heatmap_opt = output_folder / filename_template.format(
        model_name=model_name,
        classname=classname,
        seed=seed,
        kind="heatmapopt",
        ext="npy",
        caption_id=caption_id,
    )
    return filename_img, filename_heatmap, filename_heatmap_opt


def extract_only_one_timestep(heatmap):
    # Instead of sum across all timesteps, we only extract one
    # timestep_heatmap defined as a global variable
    return heatmap[timestep_heatmap, ...]


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

    # Load dictionary with tokens optimized for each class
    tokens_dict = np.load(tokens_dict_filename, allow_pickle=True).item()

    pbar = tqdm(list(df.iterrows()))

    for _, row in pbar:
        classname = row["classname"].replace(" ", "-")
        prompt = row["prompt"].lower()
        seed = row["seed"]
        caption_id = row["coco_caption_id"]
        pbar.set_description(f"{classname} - {seed} - '{prompt[:50]}'")

        filename_img, filename_heatmap, filename_heatmap_opt = generate_filenames(
            classname=classname, seed=seed, caption_id=caption_id
        )

        # Skip if already exists
        if (
            filename_img.exists()
            and filename_heatmap.exists()
            and filename_heatmap_opt.exists()
        ):
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

            # Save heatmap
            ca_extractor = hooker.daam(
                heads_epochs_aggregation=extract_only_one_timestep
            )

            # Evaluate the heatmap for the classname
            attribution_prompt = attribution_prompt_template.format(
                classname=classname.replace("-", " ")
            ).lower()

            embedding = ca_extractor.encode_text(
                attribution_prompt, remove_special_tokens=False
            )
            heatmap = ca_extractor(embedding).cpu().numpy()

            # Save all heatmap as a numpy. At evaluation we will apply beta trheshold and dcrf
            np.save(filename_heatmap, heatmap)

            # Get optimized token for classname

            classname_embedding = torch.tensor(tokens_dict[classname][0]).to(device)
            optimized_heatmap = ca_extractor(classname_embedding).cpu().numpy()

            # Save optimized heatmap
            np.save(filename_heatmap_opt, optimized_heatmap)
