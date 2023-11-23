import base64
import io
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Union

import numpy as np
import requests
import torch

try:
    import ipywidgets as widgets
    from jupyter_bbox_widget import BBoxWidget
except ImportError:
    raise ImportError(
        "jupyter_bbox_widget not found. Install it with: pip install jupyter_bbox_widget"
    )


if TYPE_CHECKING:
    from PIL.Image import Image
    from segment_anything import SamPredictor

__all__ = [
    "annotation_widget",
    "load_sam_predictor",
    "compute_masks",
    "masks_to_tensor",
]


def encode_image_from_pil(pil_image: "Image") -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    encoded = str(base64.b64encode(buf.getvalue()), "utf-8")
    return "data:image/png;base64," + encoded


def annotation_widget(
    images: Iterable["Image"], annotations: list, classes: List[str] = ["object"]
) -> "widgets.VBox":
    if not isinstance(images, list):
        images = [images]

    annotations.clear()
    w_progress = widgets.IntProgress(value=0, max=len(images), description="Progress")
    # the bbox widget
    encoded_image = encode_image_from_pil(images[0])
    w_bbox = BBoxWidget(image=encoded_image, classes=classes)

    # combine widgets into a container
    w_container = widgets.VBox(
        [
            w_progress,
            w_bbox,
        ]
    )

    # when Skip button is pressed we move on to the next file
    @w_bbox.on_skip
    def skip():
        if w_progress.value >= len(images):
            return

        annotations.append([])

        w_progress.value += 1
        w_bbox.image = encode_image_from_pil(images[w_progress.value])
        # here we assign an empty list to bboxes but
        # we could also run a detection model on the file
        # and use its output for creating inital bboxes
        w_bbox.bboxes = []

    # when Submit button is pressed we save current annotations
    # and then move on to the next file
    @w_bbox.on_submit
    def submit():
        if w_progress.value >= len(images):
            return
        # save annotations for current image
        img = images[w_progress.value]
        max_width, max_height = img.size

        bboxes = [
            {
                "bbox": np.array(
                    [
                        min(max(bbox["x"], 0), max_width),
                        min(max(bbox["y"], 0), max_height),
                        min(max(bbox["x"] + bbox["width"], 0), max_width),
                        min(max(bbox["y"] + bbox["height"], 0), max_height),
                    ]
                ),
                "label": bbox.get("label", "object"),
            }
            for bbox in w_bbox.bboxes
        ]

        annotations.append(bboxes)
        w_progress.value += 1
        if w_progress.value < len(images):
            w_bbox.image = encode_image_from_pil(images[w_progress.value])
            w_bbox.bboxes = []

    return w_container


def load_sam_predictor(
    url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    path="checkpoint/sam_vit_h_4b8939.pth",
    device: str = None,
    model_type="vit_h",
) -> "SamPredictor":
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        raise ImportError(
            "segment_anything not found. Install it with: pip install segment_anything"
        )

    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(requests.get(url).content)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=path).to(device=device)
    mask_predictor = SamPredictor(sam)
    return mask_predictor


def compute_masks(
    images: Iterable["Image"], annotations: List[List[dict]], predictor: "SamPredictor"
) -> List[List[dict]]:
    masks = []
    assert len(images) == len(
        annotations
    ), "Number of images and annotations must be equal"
    for image, annotation in zip(images, annotations):
        predictor.set_image(np.asarray(image))
        annotation_masks = []
        for bbox in annotation:
            mask, _, _ = predictor.predict(box=bbox.get("bbox"), multimask_output=False)
            mask = mask.squeeze()
            annotation_masks.append(
                {"mask": mask, "label": bbox.get("label", "object")}
            )
        masks.append(annotation_masks)
    return masks


# Group masks by label and convert to tensor
def group_mask_by_label(mask: List[dict]):
    grouped_mask = {}
    for bbox in mask:
        mask = bbox["mask"]
        label = bbox["label"]
        if label not in grouped_mask:
            grouped_mask[label] = mask.astype(int)
        else:
            grouped_mask[label] += mask.astype(int)

    for label, mask in grouped_mask.items():
        grouped_mask[label] = (mask != 0).astype(int)

    return grouped_mask


def masks_to_tensor(
    masks: List[List[dict]], labels: Optional[List[str]] = [], size=(512, 512)
):
    grouped_masks = []
    labels_appearances = []
    for mask in masks:
        grouped_mask = group_mask_by_label(mask)
        grouped_masks.append(grouped_mask)
        labels_appearances.append(list(grouped_mask.keys()))

    labels_appearances = list(
        set([label for labels in labels_appearances for label in labels])
    )

    # Check label orders
    if len(labels_appearances) > 1 and not labels:
        raise ValueError(
            "Labels must be provided if there are more than one label in the annotations"
            " to specify the order of the masks"
        )
    if not labels:
        labels = labels_appearances

    for label in labels:
        if label not in labels_appearances:
            warnings.warn(f"Label {label} not found in the annotations")
    for label in labels_appearances:
        if label not in labels:
            warnings.warn(f"Label {label} not found in the labels")

    tensor = []
    for grouped_mask in grouped_masks:
        mask_tensor = []
        for token in labels:
            mask = grouped_mask.get(token, np.zeros(size))
            mask_tensor.append(torch.tensor(mask, dtype=torch.float32))
        tensor.append(torch.stack(mask_tensor))

    tensor = torch.stack(tensor)

    return tensor
