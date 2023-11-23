import random
from typing import Optional

import numpy as np
import torch

from .device import get_device

__all__ = ["set_seed"]


def set_seed(seed: int, device: Optional[str] = None) -> "torch.Generator":
    """Sets the seed for the random number generators of torch, numpy and random.

    Arguments
    ---------
    seed : int
        The seed to set.
    device : str
        The device to use for the torch generator. If None, the default device
        is infered checking if cuda is available, mps is enabled, or
        the by default fallback to cpu.

    Returns
    -------
    torch.Generator
        The torch generator with the seed set.

    Notes
    -----
    Source: https://github.com/castorini/daam/

    """
    if device is None:
        device = get_device()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    return gen
