"""Training configuration and history utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DPIConfig:
    # Network architecture
    value_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    policy_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])

    # Optimization
    value_lr: float = 1e-3
    policy_lr: float = 1e-4
    batch_size: int = 2048
    lr_decay: float = 0.99
    lr_decay_steps: int = 15000

    # Policy evaluation
    eval_method: str = "explicit"  # "explicit" or "implicit"
    delta_t: float = 1.0
    n_eval_steps: int = 10
    n_improve_steps: int = 1

    # Ito computation
    ito_method: str = "finite_diff"  # "finite_diff" or "autodiff"
    fd_step_size: float = 5e-2
    fd_stencil: str = "nine"  # "three", "five", "seven", "nine"
    vectorized: bool = True  # Use batched shock evaluation
    chunk_size: Optional[int] = None  # Chunk size for memory management (None = no chunking)


@dataclass
class TrainingHistory:
    value_loss: List[float] = field(default_factory=list)
    policy_loss: List[Optional[float]] = field(default_factory=list)
