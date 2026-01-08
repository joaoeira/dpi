"""State space sampling utilities."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence

import torch

Tensor = torch.Tensor


class UniformSampler:
    """Uniform sampling within bounds."""

    def __init__(self, low: Sequence[float], high: Sequence[float], *, device=None, dtype=None) -> None:
        self.low = torch.as_tensor(low, device=device, dtype=dtype)
        self.high = torch.as_tensor(high, device=device, dtype=dtype)
        if self.low.shape != self.high.shape:
            raise ValueError("low and high must have the same shape")

    def sample(self, n: int) -> Tensor:
        u = torch.rand((n, self.low.numel()), device=self.low.device, dtype=self.low.dtype)
        return self.low + (self.high - self.low) * u


class DirichletSampler:
    """Sample from a simplex using a Dirichlet distribution."""

    def __init__(self, alpha: Sequence[float], *, device=None, dtype=None) -> None:
        self.alpha = torch.as_tensor(alpha, device=device, dtype=dtype)
        self.dist = torch.distributions.Dirichlet(self.alpha)

    def sample(self, n: int) -> Tensor:
        return self.dist.sample((n,))


class MixtureSampler:
    """Mix samples from multiple samplers."""

    def __init__(self, samplers: Sequence, weights: Optional[Sequence[float]] = None) -> None:
        if not samplers:
            raise ValueError("samplers must be non-empty")
        self.samplers = list(samplers)
        if weights is None:
            self.weights = torch.ones(len(self.samplers)) / len(self.samplers)
        else:
            w = torch.as_tensor(weights, dtype=torch.float)
            if w.numel() != len(self.samplers):
                raise ValueError("weights must match the number of samplers")
            self.weights = w / w.sum()

    def sample(self, n: int) -> Tensor:
        choices = torch.multinomial(self.weights, n, replacement=True)
        samples = []
        for idx in range(len(self.samplers)):
            count = (choices == idx).sum().item()
            if count == 0:
                continue
            samples.append(self.samplers[idx].sample(count))
        if not samples:
            raise RuntimeError("failed to draw samples from mixture")
        return torch.cat(samples, dim=0)


class ErgodicSampler:
    """Sample from a cached ergodic distribution or a provided callback."""

    def __init__(self, *, buffer: Optional[Tensor] = None, sample_fn: Optional[Callable[[int], Tensor]] = None) -> None:
        if buffer is None and sample_fn is None:
            raise ValueError("provide buffer or sample_fn")
        self.buffer = buffer
        self.sample_fn = sample_fn

    def sample(self, n: int) -> Tensor:
        if self.sample_fn is not None:
            return self.sample_fn(n)
        if self.buffer is None:
            raise RuntimeError("ergodic buffer is not set")
        idx = torch.randint(0, self.buffer.shape[0], (n,), device=self.buffer.device)
        return self.buffer[idx]
