from typing import TYPE_CHECKING

from torch import nn

if TYPE_CHECKING:
    from .attention_storage import AttentionStorage


class DAAMBlock(nn.Module):
    """Generic DAAMBlock used to save the hidden states of the cross attention blocks.

    Should be implemented by each of the different architectures.
    It is used to save the hidden states of the cross attention blocks and to
    build a callable DAAM function.
    """

    def __init__(
        self,
        hidden_states: "AttentionStorage",
        name: str,
    ):
        super().__init__()
        self.name = name
        self.hidden_states = hidden_states

    def forward(self, x):
        """Compute the attention for a given input x"""

        return NotImplementedError

    def store_hidden_states(self) -> None:
        """Stores the hidden states in the parent trace"""
        raise NotImplementedError
