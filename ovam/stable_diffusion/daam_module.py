from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from ..base.daam_module import DAAMModule
from ..utils.attention_ops import apply_activation, apply_aggregation

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline

    from .daam_block import CrossAttentionDAAMBlock


class StableDiffusionDAAM(DAAMModule):
    """Generic DAAMModule implementation for Stable Diffusion models. It is used to
    save the hidden states of the cross attention blocks and to build a callable
    DAAM function.

    Arguments
    ---------
    blocks : List[DAAMBlock]
        The list of DAAM blocks to use.
    tokenizer : Callable
        The tokenizer to use to encode the text.
    text_encoder : Callable
        The text encoder to use to encode the text.
    heatmaps_activation : str or Callable
        The activation function to apply to the heatmaps. If a string, it must be one of
        "relu", "sigmoid", "tanh" or None (identity function).
    heatmaps_aggregation : str or Callable
        The aggregation function to apply to the heatmaps. If a string, it must be one of
        "mean", "sum", "max" or None (identity function).
    expand_size : Tuple[int, int]
        The size to expand the heatmaps to. If None, the heatmaps are not expanded.
    expand_interpolation_mode : str
        The interpolation mode to use when expanding the heatmaps. Default: "bilinear"

    """

    def __init__(
        self,
        blocks: List["CrossAttentionDAAMBlock"],
        pipeline: "StableDiffusionPipeline",
        heatmaps_activation: Optional[
            Union[Literal["sigmoid", "tanh", "relu"], "Callable"]
        ] = None,
        heatmaps_aggregation: Union[Literal["mean", "sum"], "Callable"] = "mean",
        block_latent_size: Optional[Tuple[int, int]] = None,
        block_interpolation_mode: str = "bilinear",
        expand_size: Optional[Tuple[int, int]] = None,
        expand_interpolation_mode: str = "bilinear",
    ):
        super().__init__(blocks)

        self.block_latent_size = block_latent_size
        self.block_interpolation_mode = block_interpolation_mode
        self.tokenizer = pipeline.tokenizer
        self.text_encoder = pipeline.text_encoder
        self.heatmaps_activation = heatmaps_activation
        self.expand_size = expand_size
        self.expand_interpolation_mode = expand_interpolation_mode
        self.heatmaps_aggregation = heatmaps_aggregation

    def forward(
        self, x: Union["torch.Tensor", str], remove_special_tokens: bool = False
    ) -> "torch.Tensor":
        """Compute the attention for a given input x.

        Arguments
        ---------
        x : torch.Tensor or str
            The input to compute the attention for. If a string, it is encoded
            using the text encoder.

        Returns
        -------
        torch.Tensor
            The attention heatmaps. Shape: (n_images, n_tokens, block_latent_size[0], block_latent_size[1])
        """
        if isinstance(x, str):
            # Encode text
            x = self.encode_text(x, remove_special_tokens=remove_special_tokens)

        attention = []
        for block in self.blocks.values():
            attention.append(block.forward(x))

        # shape: (n_blocks, n_images, n_tokens, block_latent_size[0], block_latent_size[1])
        # By default block_latent_size = (64, 64)
        # Infer latent size as the maximum size of the blocks
        if self.block_latent_size is None:
            # Get the latent size from the first block
            a, b = 0, 0
            for att in attention:
                a = max(a, att.shape[-2])
                b = max(b, att.shape[-1])
            block_latent_size = (a, b)
        else:
            block_latent_size = self.block_latent_size

        # Interpolate all attentions to the same size
        attentions = []
        for att in attention:
            if att.shape[-2:] == block_latent_size:
                # If the attention has the same size as the latent size, do nothing
                attentions.append(att)
                continue
            att = F.interpolate(
                att,
                size=block_latent_size,
                mode=self.block_interpolation_mode,
            )
            attentions.append(att)
        # Remove reference to attention without interpolation
        del attention

        attentions = torch.stack(attentions, dim=0)
        attentions = apply_aggregation(
            attentions, self.heatmaps_aggregation
        )  # Collapse dim 0

        # Shape (n_images, n_tokens, block_latent_size[0], block_latent_size[1])
        attentions = apply_activation(attentions, self.heatmaps_activation)

        if self.expand_size is not None:
            attentions = F.interpolate(
                attentions,
                size=self.expand_size,
                mode=self.expand_interpolation_mode,
            )

        return attentions
