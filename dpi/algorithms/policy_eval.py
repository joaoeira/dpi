"""Policy evaluation step (critic update)."""

from __future__ import annotations

from typing import Optional

import torch

from ..core.hjb import compute_hjb_residual

Tensor = torch.Tensor


def policy_evaluation_step(
    model,
    s: Tensor,
    value_net: torch.nn.Module,
    policy_net: Optional[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    *,
    eval_method: str = "explicit",
    delta_t: float = 1.0,
    ito_method: str = "finite_diff",
    fd_step_size: float = 5e-2,
    fd_stencil: str = "three",
    vectorized: bool = True,
    chunk_size: Optional[int] = None,
) -> float:
    """One value function update using explicit or implicit evaluation."""
    value_net.train()
    if policy_net is not None:
        with torch.no_grad():
            c = policy_net(s)
    else:
        c = None

    optimizer.zero_grad()

    method = eval_method.lower()
    if method == "explicit":
        with torch.no_grad():
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
            v_target = value_net(s) + hjb * delta_t
        v_pred = value_net(s)
        loss = torch.mean((v_pred - v_target) ** 2)
    elif method == "implicit":
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
        loss = torch.mean(hjb ** 2)
    else:
        raise ValueError("eval_method must be 'explicit' or 'implicit'")

    loss.backward()
    optimizer.step()

    return float(loss.detach().cpu().item())
