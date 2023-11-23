from typing import TYPE_CHECKING, Iterable, Optional

import torch
from torch import nn

from ..utils.attention_ops import expand_as
from ..utils.text_encoding import encode_text

if TYPE_CHECKING:
    from PIL import Image

    from .daam_block import DAAMBlock

__all__ = ["DAAMModule"]


class DAAMModule(nn.Module):
    """Generic DAAMModule used to save the hidden states of the cross attention blocks
    and aggregate them. Should be implemented by each of the different architectures."""

    def __init__(self, blocks: Iterable["DAAMBlock"]) -> None:
        super().__init__()
        self.blocks = nn.ModuleDict()
        for block in blocks:
            assert block.name not in self.blocks, f"Duplicate block name: {block.name}"
            self.blocks[block.name] = block

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        raise NotImplementedError

    @torch.no_grad()
    def expand_as(
        self,
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
            The expanded heatmap. Shape: (batch, n_tokens, image.height, image.width)
        """
        return expand_as(
            image=image,
            heatmap=heatmap,
            absolute=absolute,
            threshold=threshold,
            interpolation_mode=interpolation_mode,
        )

    def encode_text(
        self,
        text: str,
        context_sentence: Optional[str] = None,
        remove_special_tokens: bool = True,
        padding=False,
    ) -> "torch.Tensor":
        """Encode a text into a sequence of tokens.

        Arguments
        ---------
        text : str
            The text to encode.
        context_sentence : str
            The context sentence to encode. If None, the text is used as context.

        Returns
        -------
        torch.Tensor
            The encoded text. Shape: (tokens, embedding_size)
        """

        text_embeddings = encode_text(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            text=text,
            context_sentence=context_sentence,
            remove_special_tokens=remove_special_tokens,
            padding=padding,
        )

        return text_embeddings
