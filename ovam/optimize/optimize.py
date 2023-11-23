from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:
    from ..base.daam_module import DAAMModule

__all__ = ["optimize_embedding"]


def optimize_embedding(
    daam_module: "DAAMModule",
    embedding: "torch.Tensor",
    target: "torch.Tensor",
    device: Optional[str] = None,
    callback: Optional[Callable] = None,
    initial_lr: float = 300,
    epochs: int = 1000,
    step_size: int = 80,
    gamma: float = 0.7,
    apply_min_max: Union[bool, int] = 3720,
    squeezed_target: bool = False,
) -> "torch.Tensor":
    """Basic optimization function for the embedding.

    Arguments
    ---------
    daam_module : DAAMModule
        The DAAM module used to evaluate the embedding.
    embedding : torch.Tensor
        The embedding to optimize.
    target : torch.Tensor
        The target to optimize the embedding.
    device : str, optional
        The device to use for the optimization, by default uses
        the device of the embedding.
    callback : Callable, optional
        A callback function to call at each epoch, by default None.
        Is called with the following arguments:
            - epoch: int
            - embedding: torch.Tensor
            - mask: torch.Tensor
            - loss: torch.Tensor
    initial_lr : float, optional
        The initial learning rate, by default 3.
    epochs : int, optional
        The number of epochs, by default 100.
    step_size : int, optional
        The step size for the scheduler, by default 80.
    gamma : float, optional
        The gamma for the scheduler, by default 0.7.

    Returns
    -------
    torch.Tensor
        The optimized embedding.

    Notes
    -----
    To obtain the losses during optimization use the callback function.

    """

    # Infer the device
    device = embedding.device if device is None else device

    # Clone the embedding as a trainable tensor
    x = embedding.detach().clone().requires_grad_(True)
    x.retain_grad()
    x.to(device)

    # Move the target to the device
    target.to(device)

    # Define the optimizer, scheduler and loss function
    optimizer = optim.SGD([x], lr=initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = nn.BCELoss(reduction="mean")
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    for i in range(epochs):
        optimizer.zero_grad()
        mask = daam_module.forward(x)
        # Apply min max normalization
        if isinstance(apply_min_max, float):
            mask = mask / apply_min_max
        elif apply_min_max:  # For the lineal case
            minimun, maximun = mask.min(), mask.max()
            mask = (mask - minimun) / (maximun - minimun)
        else:
            mask = mask / mask.sum(dim=1, keepdim=True)

        if squeezed_target:
            mask = mask.squeeze()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
        loss = loss_fn(mask, target)

        if callback is not None:
            callback(i, x, mask, loss)

        loss.backward()
        optimizer.step()
        scheduler.step()

    return x.detach().cpu()
