import math
from typing import TYPE_CHECKING, Iterable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..base.daam_block import DAAMBlock
from ..utils.attention_ops import (
    ActivationTypeVar,
    AggregationTypeVar,
    apply_activation,
    apply_aggregation,
)

if TYPE_CHECKING:
    from ..base.attention_storage import AttentionStorage

__all__ = ["CrossAttentionDAAMBlock"]


class CrossAttentionDAAMBlock(DAAMBlock):
    """This DaamBlock correspond to the attention block o a upsampling or
    downsampling layer of the UNet2DConditionModel
    """

    def __init__(
        self,
        to_k: nn.Linear,
        hidden_states: Union["AttentionStorage", Iterable["torch.Tensor"]],
        scale: float,
        heads: int,
        name: str,
        heads_activation: Optional["ActivationTypeVar"] = None,
        blocks_activation: Optional["ActivationTypeVar"] = None,
        heads_aggregation: "AggregationTypeVar" = "sum",
        heads_epochs_activation: Optional["ActivationTypeVar"] = None,
        heads_epochs_aggregation: "AggregationTypeVar" = "sum",
    ):
        super().__init__(hidden_states=hidden_states, name=name)

        self.to_k = to_k
        self.scale = scale
        self.heads = heads
        self.heads_activation = heads_activation
        self.blocks_activation = blocks_activation
        self.heads_aggregation = heads_aggregation
        self.heads_epochs_activation = heads_epochs_activation
        self.heads_epochs_aggregation = heads_epochs_aggregation

    def _compute_attention(self, query, key):
        """
        Monkey-patched version of :py:func:`.CrossAttention._attention` to capture attentions and aggregate them.

        Args:
            self (`CrossAttention`): pointer to the module.
            query (`torch.Tensor`): the query tensor.
            key (`torch.Tensor`): the key tensor.
            value (`torch.Tensor`): the value tensor.
        """
        # Cross attention matrix Wq*h x (Wk*X)^T
        attention_scores = torch.baddbmm(
            torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            ),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        # Unravel the attention scores into a collection of heatmaps
        # Unravel Based on the of the function `unravel_attn` of daam.trace
        h = w = int(math.sqrt(attention_scores.size(1)))
        attention_scores = attention_scores.permute(
            2, 0, 1
        )  # shape: (tokens, heads, h*w)

        attention_scores = attention_scores.reshape(
            (attention_scores.size(0), attention_scores.size(1), h, w)
        )  # shape: (tokens, heads, h, w)
        attention_scores = attention_scores.permute(
            1, 0, 2, 3
        ).contiguous()  # shape: (heads, tokens, h, w)

        return attention_scores  # shape: (heads, tokens, height, width)

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size * head_size, seq_len, dim // head_size
        )
        return tensor

    def forward(self, x):
        """Compute the attention for a given input x"""

        key = self.to_k(x)  # Shape: (n_tokens, embedding_size = 1024)
        key = self.reshape_heads_to_batch_dim(
            key.unsqueeze(0)
        )  # Shape: (n_heads, n_tokens, latent=64)

        heatmaps = []  #  List of heatmaps

        # Batch images can have different sizes and be stored offline
        # This loop is not vectorized for this reason
        for batch_image in self.hidden_states:
            #  TODO: This second loop can be vectorized with einsum
            attentions = []  #  List of heatmaps
            for query in batch_image:  #  ()
                attention = self._compute_attention(
                    query, key
                )  # shape: (heads, tokens, height, width)
                attentions.append(attention)
            # END of TODO: Vectorize loop
            # Shape: (n_epochs, heads, n_tokens, latent_size / factor, latent_size / factor)

            # for head dimension and epoch dimension
            attention = torch.stack(
                attentions, dim=0
            )  # Shape: (n_epochs, heads, n_tokens, latent_size / factor, latent_size / factor)

            # Collapse epochs
            # Shape: (heads, n_tokens, latent_size / factor, latent_size / factor)
            attention = apply_activation(attention, self.heads_epochs_activation)
            attention = apply_aggregation(attention, self.heads_epochs_aggregation)

            # Collapse heads dimension
            # Shape: (n_tokens, latent_size / factor, latent_size / factor)
            attention = apply_activation(attention, self.heads_activation)
            attention = apply_aggregation(attention, self.heads_aggregation)

            # Shape: (n_tokens, latent_size / factor, latent_size / factor)
            heatmaps.append(attention)

        # Shape (n_images, n_tokens, output_size, output_size)
        heatmaps = torch.stack(heatmaps, dim=0)

        return heatmaps
