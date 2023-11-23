from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from .attention_ops import expand_as

if TYPE_CHECKING:
    from PIL.Image import Image


@torch.no_grad()
def plot_overlay_heat_map(
    im: "Image", heat_map: "torch.Tensor", ax: Optional["plt.Axes"] = None
):
    """Plot an overlay of the heat map on top of the image."""
    if ax is None:
        _, ax = plt.subplots()

    if not isinstance(heat_map, torch.Tensor):
        heat_map = torch.from_numpy(heat_map)

    ax.axis("off")
    heat_map = expand_as(im, heat_map.squeeze().unsqueeze(0).unsqueeze(0))

    im = np.array(im)

    heatmap = heat_map.cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    ax.imshow(heatmap, cmap="jet")
    im = torch.from_numpy(im).float() / 255

    # Min max normalization and inversion
    heatmap = heat_map.unsqueeze(-1)
    heatmap = 1 - (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    im = torch.cat((im, heatmap), dim=-1)
    ax.imshow(im)

    return ax
