import torch

__all__ = ["get_device"]


def get_device() -> str:
    """Infer the device to use for the tests"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"

    return "cpu"
