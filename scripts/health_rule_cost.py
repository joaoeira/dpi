"""Welfare comparison between optimal policy and simple rules."""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Callable

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from dpi.algorithms.dpi import DPITrainer
from dpi.core.hjb import compute_hjb_residual
from dpi.utils.training import DPIConfig

import health_experiment


def simulate_policy(
    model,
    policy_fn: Callable[[torch.Tensor, float], torch.Tensor],
    *,
    n_paths: int = 512,
    dt: float = 1.0 / 12.0,
    horizon_years: float = 50.0,
    seed: int = 0,
) -> float:
    torch.manual_seed(seed)
    device = model.device or torch.device("cpu")
    dtype = model.dtype or torch.float32

    s = model.sample_states(n_paths).to(device=device, dtype=dtype)
    s[:, model.K] = 30.0

    n_steps = int(horizon_years / dt)
    rho = model.discount_rate
    discounts = torch.exp(-rho * dt * torch.arange(n_steps, device=device, dtype=dtype))

    total = torch.zeros((n_paths, 1), device=device, dtype=dtype)
    h_max = model.h_std_range * model._h_stationary_std.unsqueeze(0)

    for step in range(n_steps):
        t = step * dt
        u = policy_fn(s, t)
        util = model.utility(s, u)
        total = total + discounts[step] * util * dt

        drift = model.drift(s, u)
        diff_list = model.diffusion(s, u)
        g = torch.stack(diff_list, dim=-1)
        z = torch.randn((n_paths, model.K), device=device, dtype=dtype)
        noise = (g * z[:, None, :]).sum(dim=-1) * math.sqrt(dt)
        s = s + drift * dt + noise

        s[:, : model.K] = torch.clamp(s[:, : model.K], min=-h_max, max=h_max)
        s[:, model.K] = torch.clamp(s[:, model.K], max=model.age_max)

    return total.mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Health policy welfare comparison")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--correlation", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--paths", type=int, default=512)
    parser.add_argument("--dt", type=float, default=1.0 / 12.0)
    parser.add_argument("--horizon", type=float, default=50.0)
    args = parser.parse_args()

    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    model = health_experiment.create_model(
        K=args.K,
        gamma=2.0,
        correlation=args.correlation,
        device=device,
        dtype=dtype,
    )

    value_net, policy_net = health_experiment.create_networks(model.state_dim, model.control_bounds)
    value_net = value_net.to(device=device, dtype=dtype)
    policy_net = policy_net.to(device=device, dtype=dtype)

    if args.K >= 6:
        value_lr = 3e-4
        policy_lr = 1e-4
        eval_method = "implicit"
        n_eval_steps = 10
    else:
        value_lr = 1e-3
        policy_lr = 5e-4
        eval_method = "explicit"
        n_eval_steps = 5

    config = DPIConfig(
        value_lr=value_lr,
        policy_lr=policy_lr,
        batch_size=2048,
        eval_method=eval_method,
        delta_t=1.0,
        n_eval_steps=n_eval_steps,
        n_improve_steps=1,
        ito_method="finite_diff",
        fd_step_size=1e-2,
        fd_stencil="five",
        lr_decay=0.995,
        lr_decay_steps=500,
    )

    trainer = DPITrainer(model, value_net, policy_net, config)
    print("Training optimal policy...")
    trainer.train(args.iterations)

    def optimal_policy(s: torch.Tensor, _t: float) -> torch.Tensor:
        with torch.no_grad():
            return policy_net(s)

    def constant_policy(s: torch.Tensor, _t: float) -> torch.Tensor:
        batch = s.shape[0]
        e = torch.full((batch, 1), 0.12, device=s.device, dtype=s.dtype)
        d = torch.full((batch, 1), 0.35, device=s.device, dtype=s.dtype)
        l = torch.full((batch, 1), 0.25, device=s.device, dtype=s.dtype)
        u = torch.cat([e, d, l], dim=1)
        c_min, c_max = model.control_bounds(s)
        return torch.max(torch.min(u, c_max), c_min)

    def age_only_policy(s: torch.Tensor, _t: float) -> torch.Tensor:
        batch = s.shape[0]
        h_zero = torch.zeros((batch, model.K), device=s.device, dtype=s.dtype)
        s_bar = torch.cat([h_zero, s[:, model.K : model.K + 1]], dim=1)
        with torch.no_grad():
            return policy_net(s_bar)

    v_opt = simulate_policy(
        model, optimal_policy, n_paths=args.paths, dt=args.dt, horizon_years=args.horizon, seed=1
    )
    v_const = simulate_policy(
        model, constant_policy, n_paths=args.paths, dt=args.dt, horizon_years=args.horizon, seed=2
    )
    v_age = simulate_policy(
        model, age_only_policy, n_paths=args.paths, dt=args.dt, horizon_years=args.horizon, seed=3
    )

    gamma = model.gamma
    ce_const = (v_const / v_opt) ** (1.0 / (1.0 - gamma))
    ce_age = (v_age / v_opt) ** (1.0 / (1.0 - gamma))

    loss_const = 1.0 - ce_const
    loss_age = 1.0 - ce_age

    print("\nWelfare comparison (consumption-equivalent loss):")
    print(f"  V_opt   = {v_opt:.6f}")
    print(f"  V_const = {v_const:.6f} | loss = {loss_const:.3%}")
    print(f"  V_age   = {v_age:.6f} | loss = {loss_age:.3%}")


if __name__ == "__main__":
    main()
