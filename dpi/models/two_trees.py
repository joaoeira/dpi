"""Two-trees asset pricing model (example)."""

from __future__ import annotations

import math
from typing import Optional

import torch

from .base import BaseModel

Tensor = torch.Tensor


class TwoTrees(BaseModel):
    def __init__(
        self,
        rho: float = 0.04,
        sigma: float = math.sqrt(0.04),
        *,
        device=None,
        dtype=None,
    ) -> None:
        self._rho = float(rho)
        self.sigma = float(sigma)
        self.device = device
        self.dtype = dtype

    @property
    def discount_rate(self) -> float:
        return self._rho

    def drift(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        return -2.0 * (self.sigma ** 2) * s * (1.0 - s) * (s - 0.5)

    def diffusion(self, s: Tensor, c: Optional[Tensor] = None):
        g = math.sqrt(2.0) * self.sigma * s * (1.0 - s)
        return [g]

    def utility(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        # In the two-trees model, the flow payoff is the dividend share s.
        return s

    def sample_states(self, n: int) -> Tensor:
        return torch.rand((n, 1), device=self.device, dtype=self.dtype)
