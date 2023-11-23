"""
Implementation of the module locator for the cross-attention modules in
Stable Diffusion UNet (CrossAttn2DConditionModel).

Based on the original implementation from
What the DAAM:  Interpreting Stable Diffusion Using Cross Attention
(Tang et al., ACL 2023)

"""
import itertools
from typing import TYPE_CHECKING, Dict

from ..base.locator import ModuleLocator

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline
    from diffusers.models.attention import CrossAttention

__all__ = ["UNetCrossAttentionLocator"]


class UNetCrossAttentionLocator(ModuleLocator["CrossAttention"]):
    """
    Locate cross-attention modules in a UNet2DConditionModel.

    Arguments
    ---------
    restrict: bool
        If not None, only locate the cross-attention modules with the given indices.
    locate_middle_block: bool, default=False
        If True, the middle block is located and hooked. This block is not
        hooked by default because its spatial size is too small and the
        attention maps are not very informative.
    locate_attn1: bool, default=False
        If True, locate the first attention module in each cross-attention block.
    locate_attn2: bool, default=True
        If True, locate the second attention module in each cross-attention block.

    Note
    ----
    This class is based on the class UNetCrossAttentionLocator of the
    daam.trace module. The unique difference is that this class allows to
    locate the first or second attention modules separately.

    """

    def __init__(
        self,
        locate_attn1: bool = False,
        locate_attn2: bool = True,
        locate_middle_block: bool = False,
    ):
        super().__init__()
        self.locate_attn1 = locate_attn1
        self.locate_attn2 = locate_attn2
        self.locate_middle_block = locate_middle_block

    def locate(self, pipe: "StableDiffusionPipeline") -> Dict[str, "CrossAttention"]:
        """
        Locate cross-attention modules in a UNet2DConditionModel.

        Args:
            pipe (`StableDiffusionPipeline`): The pipe with unet containing
            the cross-attention modules in.

        Returns:
            `LisDict[str, CrossAttention]`: The cross-attention modules.
        """
        model = pipe.unet
        blocks = {}
        up_names = [f"up-{j}" for j in range(1, len(model.up_blocks) + 1)]
        down_names = [f"down-{j}" for j in range(1, len(model.down_blocks), +1)]

        for unet_block, name in itertools.chain(
            zip(model.up_blocks, up_names),
            zip(model.down_blocks, down_names),
            zip([model.mid_block], ["mid"]) if self.locate_middle_block else [],
        ):
            if "CrossAttn" in unet_block.__class__.__name__:
                for i, spatial_transformer in enumerate(unet_block.attentions):
                    for j, transformer_block in enumerate(
                        spatial_transformer.transformer_blocks
                    ):
                        if self.locate_attn1:
                            block_name = (
                                f"{name}-attentions-{i+1}-transformer-{j+1}-attn1"
                            )
                            blocks[block_name] = transformer_block.attn1

                        if self.locate_attn2:
                            block_name = (
                                f"{name}-attentions-{i+1}-transformer-{j+1}-attn2"
                            )
                            blocks[block_name] = transformer_block.attn2

        return blocks
