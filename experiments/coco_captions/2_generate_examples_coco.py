# Include OVAM library in the path to avoid install like a library
import sys

sys.path.append("../../")

from pathlib import Path
import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from ovdaam.stable_diffusion import StableDiffusionHooker
from ovdaam.utils import get_device, set_seed
from tqdm import tqdm
import pandas as pd

# Parameters
model_id = "runwayml/stable-diffusion-v1-5"
model_name = "sd-15"
device = get_device()  # Check availability: mps, cuda, cpu
num_inference_steps = 30
initial_seed = 1996  # Initial seed for the random generator

# Folder to store the images and heatmaps
output_folder = Path("../dataset/coco_captions_1_dataset_1/")
tokens_folder = Path("../dataset/voc_1_object_tokens/")

filename_heatmap_template = (
    "{model_name}_caption_{caption}_{classnames}_{seed}_heatmap_{n}.{ext}"
)
filename_image_template = "{model_name}_caption_{caption}_{classnames}_{seed}.{ext}"

captions_csv = Path("./coco_captions_sampled.csv")

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


def load_tokens(tokens_folder, classname):
    # Patterm model_name_classname_{integer}_{integer}.npy
    # But no incluiding model_name_classname_{integer}_{integer}_losses.npy
    class_tokens = tokens_folder.glob(
        f"{model_name}_{classname.replace(' ', '-')}_*.npy"
    )
    class_tokens = [
        class_token for class_token in class_tokens if "losses" not in class_token.name
    ]

    # Load all npy files
    class_tokens = [np.load(class_token) for class_token in class_tokens]
    # Convert all to tensor and sent to device
    class_tokens = [
        torch.tensor(class_token, device=device) for class_token in class_tokens
    ]

    return class_tokens


def save_heatmaps(
    daam,
    classname: str,
    filename: Path,
    seed: int,
    model_name: str,
    folder: Path,
    prompt: str,
    optimized_embeddings,
):
    base_name = filename.stem

    # First part -> If is contained in the prompt
    prompt_list = prompt.replace(".", " ").replace(",", "").split()
    index = prompt_list.index(classname.split()[-1])

    if index != -1:
        embedding = daam.encode_text(prompt, remove_special_tokens=False)
        heatmap = daam(embedding).cpu().numpy()[0]
        filename_heatmap = folder / f"{base_name}_heatmap_0_prompt.npy"
        np.save(filename_heatmap, heatmap)

    # Save evaluation without optimization
    embedding = daam.encode_text(classname, remove_special_tokens=False)
    heatmap = daam(embedding).cpu().numpy()
    heatmap = heatmap[0, (0, -2), ...]
    filename_heatmap = folder / f"{base_name}_heatmap_0_classname.npy"
    np.save(filename_heatmap, heatmap)

    # Second part -> If is not contained in the prompt
    for i, tok in enumerate(optimized_embeddings):
        heatmap = daam(tok).cpu().numpy()[0]
        filename_heatmap = folder / f"{base_name}_heatmap_{i+1}.npy"
        np.save(filename_heatmap, heatmap)


def save_heatmap(
    daam,
    classname,
    seed,
    model_name,
    folder,
    prompt,
    optimized_embeddings,
):
    classname = classname.replace(" ", "-")

    # Save token without optimization
    embedding = daam.encode_text(prompt, remove_special_tokens=False)

    heatmap = daam(embedding).cpu().numpy()

    heatmap = heatmap[0, (0, len(prompt.split())), ...]

    filename_heatmap = folder / f"{model_name}_{classname}_{seed}_heatmap_0.npy"
    assert len(heatmap.shape) == 3
    assert heatmap.shape[0] == 2
    np.save(filename_heatmap, heatmap)

    # Save token optimized
    for i, tok in enumerate(optimized_embeddings):
        heatmap = daam(tok).cpu().numpy()[0]

        filename_heatmap = folder / f"{model_name}_{classname}_{seed}_heatmap_{i+1}.npy"
        assert len(heatmap.shape) == 3
        assert heatmap.shape[0] == 2
        np.save(filename_heatmap, heatmap)

    return heatmap


if __name__ == "__main__":
    # Load all captions
    df = pd.read_csv(captions_csv)

    # Load all class tokens optimized
    class_tokens = {}
    for classname in voc_classes:
        class_tokens[classname] = load_tokens(tokens_folder, classname)

    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)

    print(f"Running {model_id} on {device}")

    # # Disable progress bar, thus we use tqdm on the main loop
    pipe.set_progress_bar_config(disable=True)

    # # Generate folder if empty
    output_folder.mkdir(exist_ok=True, parents=True)

    set_seed(initial_seed)
    total = len(df)
    seeds = np.random.randint(0, 2**32 - 1, size=total)
    seeds_iter = iter(seeds)

    pbar = tqdm(total=total)

    for i, row in df.iterrows():
        seed = int(next(seeds_iter))

        prompt = row["caption"]
        classname = row["category_name"]
        word_included = row["word_included"]
        caption_id = row["caption_id"]
        pbar.set_description(f"{classname}: {prompt} - {seed}")

        img_filename = output_folder / filename_image_template.format(
            model_name=model_name,
            caption=caption_id,
            classnames=classname,
            seed=seed,
            ext="png",
        )

        if img_filename.exists():
            continue

        with torch.no_grad():
            hooker, img = generate_image_hooker(
                pipe=pipe,
                prompt=prompt,
                seed=seed,
                num_inference_steps=num_inference_steps,
            )

            img.save(img_filename)

            save_heatmaps(
                daam=hooker.daam(),
                classname=classname,
                filename=img_filename,
                seed=seed,
                model_name=model_name,
                folder=output_folder,
                prompt=prompt,
                optimized_embeddings=class_tokens[classname.replace("-", " ")],
            )

        pbar.update(1)
