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

# Parameters
model_id = "runwayml/stable-diffusion-v1-5"
model_name = "sd-15"
device = get_device()  # Check availability: mps, cuda, cpu
num_inference_steps = 30
n_per_classname = 2  # 52  # Number of images to generate per class
initial_seed = 1996  # Initial seed for the random generator

# Folder to store the images and heatmaps
output_folder = Path("../dataset/voc_1_dataset_1/")
tokens_folder = Path("../dataset/voc_1_object_tokens/")
filename_heatmap_template = "heatmap_{model_name}_{classname}_{seed}_{n}_{loss}.{ext}"
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
        # Load optimized tokens
        class_tokens = load_tokens(tokens_folder, classname)

        for _ in range(n_per_classname):
            seed = int(next(seeds_iter))
            pbar.set_description(f"{classname} - {seed} ({len(class_tokens)})")
            prompt = prompt_template.format(classname=classname)

            with torch.no_grad():
                hooker, img = generate_image_hooker(
                    pipe=pipe,
                    prompt=prompt,
                    seed=seed,
                    num_inference_steps=num_inference_steps,
                )
                save_image(img, classname, seed, model_name, output_folder)
                save_heatmap(
                    daam=hooker.daam(),
                    classname=classname,
                    seed=seed,
                    model_name=model_name,
                    folder=output_folder,
                    prompt=prompt,
                    optimized_embeddings=class_tokens,
                )
            pbar.update(1)

        # Free class tokens gpu
        for class_token in class_tokens:
            del class_token
        torch.cuda.empty_cache()
