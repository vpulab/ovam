"""Base classes for all locators

Based on https://github.com/castorini/daam
What the DAAM: Interpreting Stable Diffusion Using Cross Attention. 
Tang et. al. (2023).

"""

from typing import TYPE_CHECKING, Dict, Generic
from .hooker import ModuleType

if TYPE_CHECKING:
    import torch.nn as nn


__all__ = ["ModuleLocator"]


class ModuleLocator(Generic[ModuleType]):
    """Base class for module locators.

    Module locators are used to locate modules in a model.
    """

    def locate(self, model: "nn.Module") -> Dict[str, ModuleType]:
        raise NotImplementedError
