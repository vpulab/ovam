from typing import TYPE_CHECKING, Optional, List

import torch

from ..stable_diffusion.pipeline_hooker import StableDiffusionHooker
from ..stable_diffusion.locator import UNetCrossAttentionLocator
from .self_att_block_hooker import SelfAttentionHooker


if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline


class StableDiffusionHookerSA(StableDiffusionHooker):
    """DAAMHooker used to save the hidden states of the cross attention blocks
    during the fordward passes of the Stable Diffusion UNET.

    Modification that also stores Self-Attention maps.

    Arguments
    ---------
    pipeline: StableDiffusionPipeline
        Pipeline to be hooked
    restrict_block_index: Optional[Iterable[int]]
        Restrict the hooking to the blocks with the given indices. If None, all
        the blocks are hooked.
    locate_middle_block: bool, default=False
        If True, the middle block is located and hooked. This block is not
        hooked by default because its spatial size is too small and the
        attention maps are not very informative.

    Attributes
    ----------
    module: List[ObjectHooker]
        List of ObjectHooker for the cross attention blocks
    locator: UNetCrossAttentionLocator
        Locator of the cross attention blocks

    Note
    ----
    This class is based on the class DiffusionHeatMapHooker of the
    daam.trace module.
    """

    def __init__(
        self,
        pipeline: "StableDiffusionPipeline",
        extract_self_attentions: bool = False,
        sa_in_features: Optional[List[int]] = None,
        **kwargs,
    ):
        self.sa_in_features = sa_in_features
        self.extract_self_attentions = extract_self_attentions
        super().__init__(pipeline, **kwargs)

    def _register_extra_hooks(self):
        """Hook the encode prompt and forward functions along the forward pass"""

        # Register StoreHiddenStatesHooker
        super()._register_extra_hooks()

        if self.extract_self_attentions:
            # Only locate self-attention blocks
            sa_locator = UNetCrossAttentionLocator(
                locate_middle_block=False,
                locate_attn1=True,
                locate_attn2=False,
            )
            for name, self_attention in sa_locator.locate(self.pipeline).items():
                # Only hook self-attention blocks with the desired input features
                if self.sa_in_features is not None and (
                    self_attention.to_k.in_features not in self.sa_in_features
                ):
                    continue
                self_attention_hook = SelfAttentionHooker(self_attention, name=name)
                self.register_hook(self_attention_hook)

    @property
    def cross_attention_hookers(self):
        """Returns the cross attention blockers"""
        return list(
            [m for m in self.module if hasattr(m, "name") and m.name.endswith("attn2")]
        )

    @property
    def self_attention_hookers(self):
        """Returns the cross attention blockers"""
        return list(
            [m for m in self.module if hasattr(m, "name") and m.name.endswith("attn1")]
        )

    def get_self_attention_map(self, rescale=None, stack=True, size=(64, 64)):
        """
        Return the self-attention map of the hooked blocks.

        Arguments
        ---------
        rescale: Optional[Tuple[float, float]], default=None
            Rescale the attention map to the given range (min, max). If None,
            the attention map is not rescaled. Only when stack=True.
        stack: bool, default=True
            If True, the attention maps are stacked and averaged. If False, the
            attention maps are returned as a list, with the sum of attention
            maps of each block.
        size: Tuple[int, int], default=(64, 64)
            Size of the attention maps. The attention maps are interpolated to
            this size. Only when stack=True.
        """
        sa = []

        for sa_tensor in self.self_attention_hookers:
            sa.append(
                torch.stack(sa_tensor._current_hidden_state).mean(axis=0).sum(axis=0)
            )

        if not stack:
            return sa

        for i in range(len(sa)):
            # Interpolate to 64x64
            sa[i] = torch.nn.functional.interpolate(
                sa[i].unsqueeze(0).unsqueeze(0),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        sa = -torch.stack(sa).mean(axis=0)

        if rescale is not None:
            minimun, maximun = rescale
            sa_minimun, sa_maximun = sa.min(), sa.max()
            sa = (sa - sa_minimun) / (sa_maximun - sa_minimun)
            sa = sa * (maximun - minimun) + minimun

        return sa
