"""Base classes for all hookers

Based on https://github.com/castorini/daam
What the DAAM: Interpreting Stable Diffusion Using Cross Attention. 
Tang et. al. (2023).

"""

import functools
from typing import Generic, List, TypeVar


__all__ = ["ObjectHooker", "AggregateHooker"]


ModuleType = TypeVar("ModuleType")
ModuleListType = TypeVar("ModuleListType", bound=List)


class ObjectHooker(Generic[ModuleType]):
    """Base class for hookers.

    Hookers are used to hook modules in a model and to monkey-patch their
    methods."""

    def __init__(self, module: ModuleType):
        self.module: ModuleType = module
        self.hooked = False
        self.old_state = dict()

    def __enter__(self):
        self.hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unhook()

    def hook(self):
        """Hook the module."""
        if self.hooked:
            raise RuntimeError("Already hooked module")

        self.old_state = dict()
        self.hooked = True
        self._hook_impl()

        return self

    def unhook(self):
        """Unhook the module."""
        if not self.hooked:
            raise RuntimeError("Module is not hooked")

        for k, v in self.old_state.items():
            if k.startswith("old_fn_"):
                setattr(self.module, k[7:], v)

        self.hooked = False
        self._unhook_impl()

        return self

    def monkey_patch(self, fn_name, fn):
        self.old_state[f"old_fn_{fn_name}"] = getattr(self.module, fn_name)
        setattr(self.module, fn_name, functools.partial(fn, self.module))

    def monkey_super(self, fn_name, *args, **kwargs):
        return self.old_state[f"old_fn_{fn_name}"](*args, **kwargs)

    def _hook_impl(self):
        """This method should be implemented by the child class."""
        raise NotImplementedError

    def _unhook_impl(self):
        """This method should be implemented by the child class."""
        pass


class AggregateHooker(ObjectHooker[ModuleListType]):
    """"""

    def __init__(self):
        super().__init__([])

    def _hook_impl(self):
        """Hook all the modules in the list."""
        for h in self.module:
            h.hook()

    def _unhook_impl(self):
        """Unhook all the modules in the list."""
        for h in self.module:
            h.unhook()

    def register_hook(self, hook: ObjectHooker):
        """Register a hooker to be hooked along with the others."""
        self.module.append(hook)
