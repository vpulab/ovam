from typing import TYPE_CHECKING

from .hooker import ObjectHooker, ModuleType
from .attention_storage import OnlineAttentionStorage

if TYPE_CHECKING:
    import torch
    from .daam_block import DAAMBlock


class BlockHooker(ObjectHooker["ModuleType"]):
    """Hooker for the CrossAttention blocks.

    Monkey patches the forward method of the cross attention blocks of the
    Stable Diffusion UNET.

    Arguments
    ---------

    module: CrossAttention
        Cross Attention moduled to be hooked.
    block_index: int
        Block index

    Attributes
    ----------
    module: CrossAttention
        Cross Attention module hooked
    block_index: int
        Block index
    hidden_states: List[torch.Tensor]
        List of hidden states hoked with size [ h*w ] x n_heads, where
        `h*w` is the size flattended of the unet hidden state through the block,
         (equal to h*w / (2**2*factor)) and n_heads the number of attention heads
         of the module.

    Note
    ----
        This class is based on the original implementation `daam.trace.UNetCrossAttentionHooker`.
    """

    # Default class to store the hidden states (in memory)
    STORAGE_CLASS = OnlineAttentionStorage

    def __init__(self, module: "ModuleType", name: str):
        super().__init__(module)
        self.name = name
        self.hidden_states = self.STORAGE_CLASS(name=name)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def store_hidden_states(self) -> None:
        """Stores the hidden states in the parent trace"""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear the hidden states"""
        self.hidden_states.clear()

    def _hook_impl(self) -> None:
        """Monkey patches the forward method in the cross attention block"""
        self.monkey_patch("forward", self._hooked_forward)

    def _hooked_forward(
        hk_self: "BlockHooker",
        _: "ModuleType",
        hidden_states: "torch.Tensor",
    ):
        """Hooked forward of the cross attention module.

        Stores the hidden states and perform the original attention.
        """
        raise NotImplementedError

    def daam_block(self) -> "DAAMBlock":
        """Builds a DAAMBlock with the current hidden states.

        Arguments
        ---------
        **kwargs:
            Arguments passed to the `DAAMBlock` constructor.
        """

        raise NotImplementedError
