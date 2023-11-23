import numpy as np
from pydensecrf import densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import torch

from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

__all__ = ["dcrf_mask"]


def dcrf_mask(
    heatmap: Union["torch.Tensor", "np.ndarray"],
    img: Union["Image", "np.ndarray"],
    sxy: int = 100,
    srgb: int = 1,
    compat: int = 1,
    inference: int = 3,
    clip: float = 1e-5,
    scale: Optional[float] = None,
) -> "np.ndarray":
    """
    Applies a dense conditional random field (dCRF) to a heatmap to refine a segmentation mask.

    Args:
        heatmap (Union[torch.Tensor, np.ndarray]): A tensor or numpy array of shape (n_tokens, h, w) representing the heatmap.
        img (Union[Image, np.ndarray]): An image as a PIL Image or numpy array.
        sxy (int, optional): The spatial standard deviation for the bilateral kernel. Defaults to 100.
        srgb (int, optional): The color standard deviation for the bilateral kernel. Defaults to 1.
        compat (int, optional): The compatibility function for the pairwise energy term. Defaults to 1.
        inference (int, optional): The inference method to use. Defaults to 3.
        clip (float, optional): The minimum value for the softmax. Defaults to 1e-5.
        scale (Optional[float], optional): The scaling factor for the softmax. Defaults to None.

    Returns:
        np.ndarray: A numpy array of shape (n_tokens, h, w) representing the refined segmentation mask.
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # Check if tensor instead of numpy array
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    # Normalize
    if len(heatmap.shape) == 3:
        heatmap = np.expand_dims(heatmap, axis=0)

    heatmap /= heatmap.sum(axis=1)
    heatmap = heatmap.squeeze()

    assert len(heatmap.shape) == 3, "Heatmap must be of shape (n_tokens, h, w)"

    n_tokens = heatmap.shape[0]
    w, h = img.shape[0], img.shape[1]

    # Create CRF and apply
    d = dcrf.DenseCRF2D(w, h, n_tokens)
    unary = unary_from_softmax(heatmap, clip=clip, scale=scale)
    d.setUnaryEnergy(unary)
    d.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=img, compat=compat)

    proba = d.inference(inference)
    proba = np.array(proba).reshape((n_tokens, w, h))

    return proba
