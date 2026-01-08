"""Value function neural network."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

import torch
from torch import nn

Tensor = torch.Tensor


def _make_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "silu":
        return nn.SiLU()
    if key == "gelu":
        return nn.GELU()
    if key == "softplus":
        return nn.Softplus()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError("unsupported activation: {0}".format(name))


class ValueNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128, 64),
        activation: str = "silu",
        output_transform: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(last_dim, width))
            layers.append(_make_activation(activation))
            last_dim = width
        layers.append(nn.Linear(last_dim, 1))
        self.model = nn.Sequential(*layers)
        self.output_transform = output_transform

    def forward(self, s: Tensor) -> Tensor:
        v = self.model(s)
        if self.output_transform is not None:
            return self.output_transform(s, v)
        return v
