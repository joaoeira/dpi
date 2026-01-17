"""Policy improvement step (actor update)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch

from ..core.hjb import compute_hjb_residual

Tensor = torch.Tensor


@contextmanager
def _frozen_params(module: torch.nn.Module):
    if module is None:
        yield
        return
    requires = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, req in zip(module.parameters(), requires):
            p.requires_grad_(req)


def policy_improvement_step(
    model,
    s: Tensor,
    policy_net: Optional[torch.nn.Module],
    value_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    ito_method: str = "finite_diff",
    fd_step_size: float = 5e-2,
    fd_stencil: str = "three",
    vectorized: bool = True,
    chunk_size: Optional[int] = None,
) -> Optional[float]:
    """One policy improvement step maximizing the HJB residual."""
    if policy_net is None:
        return None

    policy_net.train()
    value_net.eval()

    optimizer.zero_grad()
    with _frozen_params(value_net):
        c = policy_net(s)
        hjb = compute_hjb_residual(
            model,
            s,
            c,
            value_net,
            ito_method=ito_method,
            h=fd_step_size,
            stencil=fd_stencil,
            vectorized=vectorized,
            chunk_size=chunk_size,
        )
        loss = -hjb.mean()
    loss.backward()
    optimizer.step()

    return float(loss.detach().cpu().item())
