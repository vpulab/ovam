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

import denseCRF
import numpy as np
from PIL import Image
import sys

sys.path.append("../../")  # Add OVAM library to the path (without install it)

from diffusers import StableDiffusionPipeline
from ovdaam.stable_diffusion_sa import StableDiffusionHookerSA
from ovdaam.utils import get_device, set_seed

# Parameters
model_id = "runwayml/stable-diffusion-v1-5"
device = get_device()
num_inference_steps = 50

# File with all seeds and prompts
generation_filename = Path("./synthetic_dataset_prompts1.csv")
optimized_tokens_filename = Path("./tokens_dict_full.npy")

# Folders
output_folder = Path("./dataset/")
imgs_folder = output_folder / "image"
sa_folder = output_folder / "sa"
heatmap_folder = output_folder / "heatmap"
masks_folder = output_folder / "masks"
mask_sin_dcrf_folder = output_folder / "masks_sin_dcrf"


def generate_image_hooker(pipe, prompt, seed, num_inference_steps):
    """Generates and image and returns the hooker and the image"""
    with StableDiffusionHookerSA(pipe) as hooker:
        set_seed(seed)
        out = pipe(prompt, num_inference_steps=num_inference_steps)
    return hooker, out.images[0]


def free_hooker_memory(hooker):
    for hook in hooker.self_attention_hookers:
        for t in hook._current_hidden_state:
            del t
        del hook._current_hidden_state

    for hook in hooker.cross_attention_hookers:
        for t in hook._current_hidden_state:
            del t
        del hook._current_hidden_state
    torch.cuda.empty_cache()


def densecrf(I, P):
    """
    input parameters:
        I    : a numpy array of shape [H, W, C], where C should be 3.
               type of I should be np.uint8, and the values are in [0, 255]
        P    : a probability map of shape [H, W, L], where L is the number of classes
               type of P should be np.float32
        param: a tuple giving parameters of CRF (w1, alpha, beta, w2, gamma, it), where
                w1    :   weight of bilateral term, e.g. 10.0
                alpha :   spatial distance std, e.g., 80
                beta  :   rgb value std, e.g., 15
                w2    :   weight of spatial term, e.g., 3.0
                gamma :   spatial distance std for spatial term, e.g., 3
                it    :   iteration number, e.g., 5
    output parameters:
        out  : a numpy array of shape [H, W], where pixel values represent class indices.
    """
    w1 = 10.0  # weight of bilateral term
    alpha = 80  # spatial std
    beta = 13  # rgb  std
    w2 = 3.0  # weight of spatial term
    gamma = 3  # spatial std
    it = 5.0  # iteration
    param = (w1, alpha, beta, w2, gamma, it)
    out = denseCRF.densecrf(I, P, param)
    return out


def generate_binary_mask(ca, sa, img, category_id, threshold=0.75, sa_range=(0.95, 1)):
    # Normalize sa
    min_sa, max_sa = sa_range
    sa = (sa - sa.min()) / (sa.max() - sa.min())
    sa = sa * (max_sa - min_sa) + min_sa
    assert np.abs(sa.min() - min_sa) < 0.001
    assert sa.shape == (64, 64)

    heatmap = ca[1]  # Foreground
    heatmap = heatmap / heatmap.max()
    heatmap = heatmap * sa  # Apply sa
    # Interpolate heatmap to image size (512, 512)

    # Resize heatmap to image size
    heatmap = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)

    heatmap = torch.nn.functional.interpolate(
        heatmap, size=(512, 512), mode="bicubic", align_corners=False
    )
    # Clip to (0, 1)
    heatmap = torch.clamp(heatmap, 0, 1)
    heatmap = heatmap.squeeze(0).squeeze(0).numpy()

    mask = heatmap > threshold
    mask_sin_dcrf = mask.copy().astype(np.float32)
    mask = np.stack([1 - mask, mask], axis=-1).astype(np.float32)
    mask = densecrf(img, mask).astype(np.float32)

    assert mask.shape == (512, 512), f"Mask {mask} has wrong shape {mask.shape}"

    # Where mask is true -> category_id, else 255
    mask[mask == 1] = category_id
    mask[mask != category_id] = 255

    # Same with mask_sin_dcrf
    mask_sin_dcrf[mask_sin_dcrf == 1] = category_id
    mask_sin_dcrf[mask_sin_dcrf != category_id] = 255

    return mask.astype(np.uint8), mask_sin_dcrf.astype(np.uint8)


if __name__ == "__main__":
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)
    # pipe.set_progress_bar_config(disable=True)  # Disable per/step progress bar

    print(f"Running {model_id} on {device}")

    # Generate golder if empty
    output_folder.mkdir(exist_ok=True, parents=True)
    imgs_folder.mkdir(exist_ok=True, parents=True)
    sa_folder.mkdir(exist_ok=True, parents=True)
    heatmap_folder.mkdir(exist_ok=True, parents=True)
    masks_folder.mkdir(exist_ok=True, parents=True)
    mask_sin_dcrf_folder.mkdir(exist_ok=True, parents=True)
    set_seed(1)

    # Load csv with all seeds and prompts
    df = pd.read_csv(generation_filename)
    tokens_dict = np.load(optimized_tokens_filename, allow_pickle=True).item()

    tokens_tensor_dict = {}
    for k, v in tokens_dict.items():
        k = k.replace("-", "").replace("tv", "tvmonitor")
        tokens_tensor_dict[k] = torch.tensor(v[0]).to(device)
        assert tokens_tensor_dict[k].shape == (2, 768)
    pbar = tqdm(list(df.iterrows()))

    for _, row in pbar:
        category = row["category"]
        category_id = row["category_id"]
        caption_id = row["coco_caption_id"]
        token = tokens_tensor_dict[category]
        seed = row["seed"]
        prompt = row["prompt"].lower()
        pbar.set_description(f"{category} - {seed} - '{prompt[:50]}'")

        filename_img = imgs_folder / f"{category_id}_{caption_id}_{seed}.png"
        filename_img_jpg = imgs_folder / f"{category_id}_{caption_id}_{seed}.jpg"
        filename_sa = sa_folder / f"{category_id}_{caption_id}_{seed}.npy"
        filename_heatmap = heatmap_folder / f"{category_id}_{caption_id}_{seed}.npy"
        filename_mask = masks_folder / f"{category_id}_{caption_id}_{seed}.png"
        filename_mask_sin_dcrf = (
            mask_sin_dcrf_folder / f"{category_id}_{caption_id}_{seed}.png"
        )

        # Skip if already exists
        if (
            # filename_img.exists()
            filename_img_jpg.exists()
            and filename_sa.exists()
            and filename_heatmap.exists()
            and filename_mask.exists()
            and filename_mask_sin_dcrf.exists()
        ):
            continue

        with torch.no_grad():
            hooker, img = generate_image_hooker(
                pipe=pipe,
                prompt=prompt,
                seed=seed,
                num_inference_steps=num_inference_steps,
            )

            # Save image (is a PIL image)
            # img.save(filename_img)

            # Save sa
            sa = hooker.get_self_attention_map().cpu().numpy()
            assert sa.shape == (64, 64)
            np.save(filename_sa, sa)

            daam = hooker.daam()
            ca = daam(token)[0].cpu().numpy()
            assert ca.shape == (2, 64, 64), f"Error shape {ca.shape} is not (2, 64, 64)"
            np.save(filename_heatmap, ca)

            # Save mask
            mask, mask_sin_dcrf = generate_binary_mask(
                ca=ca, sa=sa, img=img, category_id=category_id
            )
            assert mask.shape == (512, 512), f"Mask {mask} has wrong shape {mask.shape}"
            assert mask_sin_dcrf.shape == (
                512,
                512,
            ), f"Mask {mask_sin_dcrf} has wrong shape {mask_sin_dcrf.shape}"

            # Save as png
            Image.fromarray(mask).save(filename_mask)
            Image.fromarray(mask_sin_dcrf).save(filename_mask_sin_dcrf)

            # Convert image to jpg and save
            img.convert("RGB").save(filename_img_jpg)

            # Free memory
            free_hooker_memory(hooker)
            del hooker