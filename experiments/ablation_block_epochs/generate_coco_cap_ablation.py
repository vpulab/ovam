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
num_inference_steps = 30  # Same as in the paper

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


def attention_spplited_by_epochs(attention_storage):
    attn_aux = []
    for attn in attention_storage:
        # [31, 8, 6, 16, 16] -> [6, 31, 16, 16]
        attn = attn.sum(axis=1)

        # [31, 8, 16, 16] -> [31, 8, 64, 64]
        attn = torch.nn.functional.interpolate(
            attn, size=(64, 64), mode="bilinear", align_corners=False
        )
        attn_aux.append(attn)

    # Stack attns and sum accross blocks
    return torch.stack(attn_aux, dim=0).sum(axis=0)


def attention_spplited_by_blocks(attention_storage):
    attn_aux = {}
    attn_aux[16] = []
    attn_aux[32] = []
    attn_aux[64] = []
    for attn in attention_storage:
        # [31, 8, 6, 16, 16] -> [31, 6, 16, 16]
        attn = attn.sum(axis=0).sum(axis=0)
        attn_aux[attn.shape[-1]].append(attn)

    a16 = torch.stack(attn_aux[16], dim=0).sum(axis=0)
    a32 = torch.stack(attn_aux[32], dim=0).sum(axis=0)
    a64 = torch.stack(attn_aux[64], dim=0).sum(axis=0)
    # Free device memory
    for k, v in attn_aux.items():
        for t in v:
            del t
    del attn_aux

    # Stack attns and sum accross blocks
    return a16.cpu().numpy(), a32.cpu().numpy(), a64.cpu().numpy()


def free_storage_attentions():
    global attention_storage
    for attn in attention_storage:
        del attn
    attention_storage = []


def free_hooker_memory(hooker):
    for hook in hooker.cross_attention_hookers:
        for t in hook._current_hidden_state:
            del t
        del hook._current_hidden_state
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

    # Load dictionary with tokens optimized for each class
    tokens_dict = np.load(tokens_dict_filename, allow_pickle=True).item()

    pbar = tqdm(list(df.iterrows()))

    for _, row in pbar:
        classname = row["classname"].replace(" ", "-")
        prompt = row["prompt"].lower()
        seed = row["seed"]
        caption_id = row["coco_caption_id"]
        pbar.set_description(f"{classname} - {seed} - '{prompt[:50]}'")

        filename_img = (
            output_folder / f"{classname}_caption{caption_id}_{seed}_image.png"
        )
        epochs_filename = (
            output_folder / f"{classname}_caption{caption_id}_{seed}_epochs.npy"
        )
        a16_filename = output_folder / f"{classname}_caption{caption_id}_{seed}_a16.npy"
        a32_filename = output_folder / f"{classname}_caption{caption_id}_{seed}_a32.npy"
        a64_filename = output_folder / f"{classname}_caption{caption_id}_{seed}_a64.npy"
        epochs_opt_filename = (
            output_folder / f"{classname}_caption{caption_id}_{seed}_epochs_opt.npy"
        )
        a16_opt_filename = (
            output_folder / f"{classname}_caption{caption_id}_{seed}_a16_opt.npy"
        )
        a32_opt_filename = (
            output_folder / f"{classname}_caption{caption_id}_{seed}_a32_opt.npy"
        )
        a64_opt_filename = (
            output_folder / f"{classname}_caption{caption_id}_{seed}_a64_opt.npy"
        )

        # Skip if already exists
        if (
            filename_img.exists()
            and epochs_filename.exists()
            and a16_filename.exists()
            and a32_filename.exists()
            and a64_filename.exists()
            and epochs_opt_filename.exists()
            and a16_opt_filename.exists()
            and a32_opt_filename.exists()
            and a64_opt_filename.exists()
        ):
            continue

        attention_storage = []

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
            def store_head_epochs_states(tensor):
                global attention_storage
                # torch.Size([101, 8, 6, 16, 16])
                if tensor.ndim == 5:
                    attention_storage.append(tensor)
                return tensor.sum(axis=0)

            # Save heatmap
            ca_extractor = hooker.daam(
                heads_epochs_aggregation=store_head_epochs_states,
            )

            # Evaluate the heatmap for the classname
            attribution_prompt = attribution_prompt_template.format(
                classname=classname.replace("-", " ")
            ).lower()

            embedding = ca_extractor.encode_text(
                attribution_prompt, remove_special_tokens=False
            )

            heatmap = ca_extractor(embedding)
            del heatmap
            output = attention_spplited_by_epochs(attention_storage).cpu().numpy()
            a16, a32, a64 = attention_spplited_by_blocks(attention_storage)
            print(output.shape, a16.shape, a32.shape, a64.shape)
            np.save(epochs_filename, output)
            np.save(a16_filename, a16)
            np.save(a32_filename, a32)
            np.save(a64_filename, a64)

            # Free memory
            free_storage_attentions()
            attention_storage = []

            # Get optimized token for classname
            classname_embedding = torch.tensor(tokens_dict[classname][0]).to(device)
            optimized_heatmap = ca_extractor(classname_embedding)
            del optimized_heatmap
            output = attention_spplited_by_epochs(attention_storage).cpu().numpy()
            a16, a32, a64 = attention_spplited_by_blocks(attention_storage)
            print(output.shape, a16.shape, a32.shape, a64.shape)
            np.save(epochs_opt_filename, output)
            np.save(a16_opt_filename, a16)
            np.save(a32_opt_filename, a32)
            np.save(a64_opt_filename, a64)

            # Free memory
            free_storage_attentions()
            attention_storage = []
            free_hooker_memory(hooker)
            del hooker
            del ca_extractor

