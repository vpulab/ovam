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
from PIL import Image
from ovdaam.optimize import optimize_embedding

# Parameters
model_id = "runwayml/stable-diffusion-v1-5"
model_name = "sd-15"
device = get_device()  # Check availability: mps, cuda, cpu
num_inference_steps = 30
n_per_classname = 3  # Number of images to generate per class
initial_seed = 1  # Initial seed for the random generator

# Folder to store the images and heatmaps
gt_folder = Path("../dataset/voc_1_object_gt/")
token_folder = Path("../dataset/voc_1_object_tokens/")

filename_template = "{model_name}_{classname}_{seed}.{ext}"

# Voc classes and prompt template
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

best_embedding = None
best_loss = None
pbar_opt = None
losses = None

def generate_image_hooker(pipe, prompt, seed, num_inference_steps):
    """Generates and image and returns the hooker and the image"""
    with StableDiffusionHooker(pipe) as hooker:
        set_seed(seed)
        out = pipe(prompt, num_inference_steps=num_inference_steps)
    return hooker, out.images[0]


def get_class_gts(folder: Path, classname: str):
    folder = Path(folder)
    classname = classname.replace(" ", "-")
    images = folder.glob(f"{model_name}_{classname}_*.png")
    #images = folder.glob(f"{classname}_*.png")
    return list(images)


def get_gt_mask(gt: Path):
    gt = np.array(Image.open(gt).convert("L"))
    gt = (gt > 0).astype(float)

    mask = np.stack([1 - gt, gt])
    mask = np.expand_dims(mask, 0)

    assert mask.shape == (1, 2, 512, 512)
    return torch.Tensor(mask)


def optimize_token(daam, binary_mask, prompt):
    embedding = daam.encode_text(
        text=prompt.split(" ")[-1],
        context_sentence=prompt,
        remove_special_tokens=False,
    )
    global best_loss
    global best_embedding
    global pbar_opt
    global losses
    losses =[] 
    best_loss = 10000
    epochs = 800
    # <SoT>classname<EoT>
    embedding = embedding[(0, -2), :]  # Remove EoT
    pbar_opt = tqdm(total=epochs, leave=False)
    def callback(epoch, embedding, mask, loss):
        global best_embedding
        global best_loss
        global pbar_opt

        if loss < best_loss:
            best_embedding = embedding.clone().detach()
            best_loss = float(loss)
            pbar_opt.set_description(f"Loss {best_loss:.5f}")
            #print("Better embedding found!")
        #print(f"Epoch: {epoch}, Loss: {loss}", best_loss)
        pbar_opt.update()
        losses.append(float(loss))

    optimize_embedding(
        daam,
        embedding=embedding.to(device),
        target=binary_mask.to(device),
        device=device,
        callback=callback,
        apply_min_max=3720,
        initial_lr=100,
        gamma=0.7,
        step_size=120,
        epochs=epochs,
    )
    pbar_opt.close()
    pbar_opt = None
    return best_embedding, best_loss


def save_token(token, classname, seed, model_name, folder, loss):
    l = int(loss * 100)
    filename = folder / f"{model_name}_{classname.replace(' ', '-')}_{seed}_{l}.npy"
    np.save(filename, token)
    global losses

    losses_array = np.array(losses)
    filename_l = folder / f"{model_name}_{classname.replace(' ', '-')}_{seed}_{l}_losses.npy"
    np.save(filename_l, losses_array)

    return filename


if __name__ == "__main__":
    # Load the model
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)

    print(f"Running {model_id} on {device}")

    # Disable progress bar, thus we use tqdm on the main loop
    pipe.set_progress_bar_config(disable=True)

    # Generate golder if empty
    token_folder.mkdir(exist_ok=True, parents=True)

    total = len(list(gt_folder.glob("*.png")))
    pbar = tqdm(total=total)
    for classname in voc_classes:
        gts = get_class_gts(gt_folder, classname)
        # Optimize a token for each image
        for gt in gts:
            pbar.set_description(f"{gt}")
            name = gt.stem
            seed = int(name.split("_")[-1])
            prompt = prompt_template.format(classname=classname)

            # Generate the token
            hooker, image = generate_image_hooker(
                pipe=pipe,
                prompt=prompt,
                seed=seed,
                num_inference_steps=num_inference_steps,
            )
            daam = hooker.daam(expand_size=(512, 512))
            binary_mask = get_gt_mask(gt)
            best_embedding, best_loss = optimize_token(daam, binary_mask, prompt)
            # Save the token
            best_token = best_embedding.cpu().numpy()
            save_token(
                token=best_token,
                classname=classname,
                seed=seed,
                model_name=model_name,
                folder=token_folder,
                loss=best_loss,
            )
            pbar.update()
        
