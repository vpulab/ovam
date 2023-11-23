"""Utility functions used in the DAAM modules"""

from typing import TYPE_CHECKING, Callable, Literal, Optional, Union

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from PIL import Image

__all__ = [
    "AggregationTypeVar",
    "ActivationTypeVar",
    "apply_activation",
    "apply_aggregation",
    "expand_as",
]

AggregationTypeVar = Union[Literal["mean", "sum", "max"], Callable]
ActivationTypeVar = Union[
    Literal[
        "relu", "sigmoid", "tanh", "sigmoid+relu", "clamp", "token_softmax", "linear"
    ],
    Callable,
]


def apply_activation(
    tensor: "torch.Tensor",
    activation: Optional[ActivationTypeVar],
):
    """Apply a given activation function to a tensor

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor to apply the activation function to
    activation : str or Callable
        The activation function to apply. If a string, it must be one of
        "relu", "sigmoid", "tanh" or None (identity function).

    """
    if callable(activation):
        return activation(tensor)

    if activation == "relu":
        return torch.relu(tensor)
    elif activation == "sigmoid":
        return torch.sigmoid(tensor)
    elif activation == "tanh":
        return torch.tanh(tensor)
    elif activation == "sigmoid+relu":
        return torch.relu((torch.sigmoid(tensor) * 2) - 1)
    elif activation == "clamp":
        return torch.clamp(tensor, min=0, max=1)
    elif activation == "token_softmax":
        return torch.softmax(tensor, dim=-3)
    elif activation is None or activation == "linear":
        return tensor
    else:
        raise ValueError(f"Unknown activation function: {activation}")


def apply_aggregation(
    tensor: "torch.Tensor",
    aggregation: AggregationTypeVar,
):
    """Apply a given activation function to a tensor

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor to apply the activation function to
    activation : str or Callable
        The activation function to apply. If a string, it must be one of
        "relu", "sigmoid", "tanh" or None (identity function).

    """
    if aggregation is None or aggregation=='none':
        return tensor
    if callable(aggregation):
        return aggregation(tensor)
    if aggregation == "mean":
        return torch.mean(tensor, dim=0)
    elif aggregation == "sum":
        return torch.sum(tensor, dim=0)
    elif aggregation == "max":
        return torch.max(tensor, dim=0)
    else:
        raise ValueError(f"Unknown activation function: {aggregation}")


def expand_as(
    image: "Image",
    heatmap: "torch.Tensor",
    absolute: bool = False,
    threshold: Optional[float] = None,
    interpolation_mode: str = "bilinear",
) -> "torch.Tensor":
    """
    Expand a heatmap to the size of an image.

    Arguments
    ---------
    image : Image
        The image to expand the heatmap to.
    heatmap : torch.Tensor
        The heatmap to expand. Should be of shape
        (batch, n_tokens, block_latent_size[0], block_latent_size[1])
    absolute : bool
        If True, the heatmap is normalized to [0, 1]. If False, the heatmap
        is normalized to [0, 1] and then thresholded.
    threshold : float
        The threshold to apply to the heatmap. If None, no threshold is applied.
    interpolation_mode : str
        The interpolation mode to use when expanding the heatmap. Default: "bilinear"

    Returns
    -------
    torch.Tensor
        The expanded heatmap. Shape: (image.height, image.width)

    Note
    ----
    This method is based on the method `expand_as` of the class
    `DiffusionHeatMapHooker` of the `daam.trace` module. What the DAAM:
    Interpreting Stable Diffusion Using Cross Attention (Tang et al., ACL 2023)

    """
    # Resize as (N, C, H, W) including batch dimension

    im = heatmap  # .unsqueeze(0)  # .unsqueeze(0)

    im = F.interpolate(
        im.float().detach(),
        size=(image.size[0], image.size[1]),
        mode=interpolation_mode,
    )

    if not absolute:
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)

    if threshold:
        im = (im > threshold).float()

    im = im.cpu().detach().squeeze()

    return im
