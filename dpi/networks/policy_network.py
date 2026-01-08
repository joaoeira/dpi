"""Policy function neural network."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch import nn

Tensor = torch.Tensor


def _make_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "leaky_relu":
        return nn.LeakyReLU()
    if key == "elu":
        return nn.ELU()
    if key == "tanh":
        return nn.Tanh()
    raise ValueError("unsupported activation: {0}".format(name))


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (256, 128, 64),
        activation: str = "relu",
        output_bounds: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(last_dim, width))
            layers.append(_make_activation(activation))
            last_dim = width
        layers.append(nn.Linear(last_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.output_bounds = output_bounds

    def forward(self, s: Tensor) -> Tensor:
        out = self.model(s)
        if self.output_bounds is None:
            return out
        low, high = self.output_bounds
        return low + (high - low) * torch.sigmoid(out)


class StateBoundedPolicy(nn.Module):
    """Policy wrapper with state-dependent bounds."""

    def __init__(
        self,
        base_net: nn.Module,
        bounds_fn: Callable[[Tensor], Tuple[Tensor, Tensor]],
    ) -> None:
        super().__init__()
        self.base_net = base_net
        self.bounds_fn = bounds_fn

    def forward(self, s: Tensor) -> Tensor:
        raw = self.base_net(s)
        c_min, c_max = self.bounds_fn(s)
        span = torch.clamp(c_max - c_min, min=1e-8)
        return c_min + span * torch.sigmoid(raw)
