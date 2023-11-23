from typing import TYPE_CHECKING, Type

from .hooker import AggregateHooker

if TYPE_CHECKING:
    from diffusers import StableDiffusionPipeline

    from .daam_module import DAAMModule
    from .hooker import ObjectHooker
    from .locator import ModuleLocator


class PipelineHooker(AggregateHooker):
    """Hooker used to save the hidden states of the cross attention blocks
    during the fordward passes of the Stable Diffusion UNET.

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
        locator: "ModuleLocator",
        block_hooker_class: Type["ObjectHooker"],
        daam_module_class: Type["DAAMModule"],
        block_hooker_kwargs: dict = {},
    ):
        super().__init__()
        self.pipeline = pipeline
        self.daam_module_class = daam_module_class

        for name, cross_attention in locator.locate(pipeline).items():
            cross_attention_hook = block_hooker_class(
                cross_attention, name=name, **block_hooker_kwargs
            )
            self.register_hook(cross_attention_hook)

        self._register_extra_hooks()

    def _register_extra_hooks(self):
        pass

    @property
    def cross_attention_hookers(self):
        """Returns the cross attention blockers. Here, maybe you want to overload to
        include logic to filter hooks that are not cross attention blocks.
        """
        return self.module

    def clear(self) -> None:
        """Clear the hidden states of the cross attention blocks hooks"""
        for cross_attention_hook in self.cross_attention_hookers:
            cross_attention_hook.clear()

    def _store_hidden_states(self):
        """Stores the hidden states in the parent trace"""
        for cross_attention_hook in self.cross_attention_hookers:
            cross_attention_hook.store_hidden_states()

    def daam(self, module_kwargs, block_kwargs) -> "DAAMModule":
        """
        Builds a callable DAAM function
        """

        blocks = [
            hook.daam_block(**block_kwargs) for hook in self.cross_attention_hookers
        ]
        return self.daam_module_class(
            blocks=blocks, pipeline=self.pipeline, **module_kwargs
        )
