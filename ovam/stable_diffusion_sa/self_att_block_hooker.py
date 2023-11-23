from typing import TYPE_CHECKING, List

import torch

# Import F
import torch.nn.functional as F

from ..base.block_hooker import BlockHooker
import math

__all__ = ["SelfAttentionHooker"]


class SelfAttentionHooker(BlockHooker):
    def __init__(
        self,
        module: "CrossAttention",
        name: str,
    ):
        super().__init__(module=module, name=name)
        self._current_hidden_state: List["torch.tensor"] = []

    def _hook_impl(self) -> None:
        """Monkey patches the forward method in the cross attention block"""
        self.monkey_patch("forward", self._hooked_forward)

    def _hooked_forward(
        hk_self: "BlockHooker",
        attn,
        hidden_states,
        **kwargs,
    ):
        """Hooked forward of the cross attention module.

        Stores the hidden states and perform the original attention.
        """

        output = hk_self.monkey_super("forward", hidden_states, **kwargs)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )
        USE_PEFT_BACKEND = True
        scale = attn.scale
        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        encoder_hidden_states = hidden_states

        key = (
            attn.to_k(encoder_hidden_states, scale=scale)
            if not USE_PEFT_BACKEND
            else attn.to_k(encoder_hidden_states)
        )
        # value = (
        #     attn.to_v(encoder_hidden_states, scale=scale)
        #     if not USE_PEFT_BACKEND
        #     else attn.to_v(encoder_hidden_states)
        # )

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        attention_mask = None

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Makes a ones matrix with the same shape than value
        dummy_value = torch.ones_like(key)

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            dummy_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.sum(axis=0).sum(axis=-1)

        spatial_dimension = int(math.sqrt(hidden_states.shape[-1]))
        hidden_states = hidden_states.reshape(
            (-1, spatial_dimension, spatial_dimension)
        )

        hk_self._current_hidden_state.append(hidden_states)

        return output
