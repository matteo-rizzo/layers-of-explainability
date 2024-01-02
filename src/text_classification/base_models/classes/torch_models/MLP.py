from typing import Callable, Optional

import torch
from torch import nn


class MLP(nn.Module):
    """
    Fully connected MLP with variable layers.
    Use batch normalization, optional dropout and Tanh activation, as in DN. Dropout after tanh is taken from NFM implementation.

    Parameters:
        - input_dim: input dimensionality
        - layers: a tuple for each layer that should be created, specifying
            - output dimensionality
            - dropout probability (0 for no dropout)
            - batch norm (true to use it, false otw)
            - activation function callable
    """

    def __init__(self, input_dim: int, layers: list[tuple[int, float, bool, Optional[Callable]]]):
        super().__init__()

        modules = list()
        for out_dim, drop_p, batch_norm, act in layers:
            modules.append(nn.Linear(input_dim, out_dim))
            if batch_norm:
                modules.append(nn.BatchNorm1d(num_features=out_dim))
            if act is not None:
                modules.append(act)
            if drop_p > 0:
                modules.append(nn.Dropout(p=drop_p))
            input_dim = out_dim

        self.dnn = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This is needed by core, because input is cast to torch.float64, and weights are float32
        x = torch.as_tensor(x, dtype=torch.float32)
        return self.dnn(x)
