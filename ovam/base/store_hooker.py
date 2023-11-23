from .hooker import ObjectHooker, ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline_hooker import PipelineHooker

__all__ = ["StoreHiddenStatesHooker"]


class StoreHiddenStatesHooker(ObjectHooker[ModuleType]):
    """Hooker for StableDiffusionPipeline.

    This hooker checks that batched prompt is not used and call the method to save
    the hidden states after each image generation.
    """

    def __init__(
        self,
        module: "ModuleType",
        parent_trace: "PipelineHooker",
        function_patched: str,
    ):
        super().__init__(module)
        self.function_patched = function_patched
        self.parent_trace = parent_trace

    def _hooked_store_hidden_states(hk_self, _: "PipelineHooker", *args, **kwargs):
        """Dummy hook function to store the hidden states after each image generation"""
        hk_self.parent_trace._store_hidden_states()
        return hk_self.monkey_super(hk_self.function_patched, *args, **kwargs)

    def _hook_impl(self):
        """Peroforms the hooking of the function"""
        self.monkey_patch(self.function_patched, self._hooked_store_hidden_states)
