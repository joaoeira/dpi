"""Abstract model interface for DPI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

Tensor = torch.Tensor


class BaseModel(ABC):
    """Abstract base class for continuous-time models."""

    @abstractmethod
    def drift(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """Compute drift f(s, c)."""

    @abstractmethod
    def diffusion(self, s: Tensor, c: Optional[Tensor] = None):
        """Compute diffusion columns g_i(s, c)."""

    @abstractmethod
    def utility(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """Compute instantaneous utility u(s, c)."""

    @property
    @abstractmethod
    def discount_rate(self) -> float:
        """Return discount rate rho."""

    @abstractmethod
    def sample_states(self, n: int) -> Tensor:
        """Sample n states from the state space."""
