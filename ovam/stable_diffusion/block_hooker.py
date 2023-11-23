from typing import TYPE_CHECKING, List

import torch

from ..base.block_hooker import BlockHooker
from .daam_block import CrossAttentionDAAMBlock

if TYPE_CHECKING:
    from diffusers.models.attention import CrossAttention

__all__ = ["CrossAttentionHooker"]


class CrossAttentionHooker(BlockHooker):
    def __init__(
        self,
        module: "CrossAttention",
        name: str,
        store_unconditional_hidden_states: bool = True,
        store_conditional_hidden_states: bool = False,
    ):
        super().__init__(module=module, name=name)
        self._current_hidden_state: List["torch.tensor"] = []
        self.store_conditional_hidden_states = store_conditional_hidden_states
        self.store_unconditional_hidden_states = store_unconditional_hidden_states

    def _hooked_forward(
        hk_self: "BlockHooker",
        _: "CrossAttention",
        hidden_states: "torch.Tensor",
        **kwargs,
    ):
        """Hooked forward of the cross attention module.

        Stores the hidden states and perform the original attention.
        """
        # Save the hidden states
        # [ h*w ] x n_heads (The original size is {1, 2} x [h*w] x n_heads)
        if hk_self.store_unconditional_hidden_states:
            hk_self._current_hidden_state.append(hidden_states[0])
        if hk_self.store_conditional_hidden_states:
            assert hidden_states.shape[0] > 1
            hk_self._current_hidden_state.append(hidden_states[1])

        return hk_self.monkey_super("forward", hidden_states, **kwargs)

    def store_hidden_states(self) -> None:
        """Stores the hidden states in the parent trace"""
        if not self._current_hidden_state:
            return

        queries = []  # This loop can be vectorized, but has a small impact
        # Thus it is not executed during the optimization process
        for c in self._current_hidden_state:
            query = self.module.to_q(c.unsqueeze(0))
            query = self.module.head_to_batch_dim(query)
            queries.append(query)

        # n_epochs x heads x inner_dim x (latent_size = 64)
        current_hidden_states = torch.stack(queries)

        self.hidden_states.store(current_hidden_states)

        self._current_hidden_state = []  # Clear the current hidden states

    def daam_block(self, **kwargs) -> "CrossAttentionDAAMBlock":
        """Builds a DAAMBlock with the current hidden states.

        Arguments
        ---------
        **kwargs:
            Arguments passed to the `DAAMBlock` constructor.
        """

        return CrossAttentionDAAMBlock(
            to_k=self.module.to_k,
            hidden_states=self.hidden_states,
            scale=self.module.scale,
            heads=self.module.heads,
            name=self.name,
            **kwargs,
        )
