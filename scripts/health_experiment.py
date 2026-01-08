"""Health management experiment with multi-dimensional health states."""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Optional

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpi.algorithms.dpi import DPITrainer
from dpi.core.hjb import compute_hjb_residual
from dpi.models.health_management import HealthManagement, default_phi_matrix
from dpi.networks.policy_network import PolicyNetwork, StateBoundedPolicy
from dpi.networks.value_network import ValueNetwork
from dpi.utils.training import DPIConfig, TrainingHistory

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def create_model(
    K: int,
    gamma: float,
    correlation: float,
    device=None,
    dtype=None,
) -> HealthManagement:
    if K == 8:
        phi = default_phi_matrix()
    else:
        phi = None

    if K > 1 and correlation != 0.0:
        corr = torch.full((K, K), correlation, device=device, dtype=dtype)
        corr.fill_diagonal_(1.0)
    else:
        corr = None

    omega = [0.8, 0.5, 0.4, 0.3, 0.7, 0.6, 0.7, 0.3]
    wage_weights = [0.15, 0.05, 0.08, 0.04, 0.12, 0.10, 0.20, 0.04]
    beta_e = [0.5, 0.3, 0.2, 0.4, 0.35, 0.2, 0.15, 0.1]
    delta_base = [0.15, 0.12, 0.10, 0.10, 0.18, 0.15, 0.10, 0.10]
    delta_slope = [0.004, 0.003, 0.003, 0.004, 0.003, 0.003, 0.004, 0.003]

    return HealthManagement(
        K=K,
        gamma=gamma,
        alpha=0.4,
        eta=0.08,
        omega=omega[:K],
        delta_base=delta_base[:K],
        delta_slope=delta_slope[:K],
        nu0=-0.01,
        nu1=-0.02,
        phi=phi,
        beta_e=beta_e[:K],
        beta_d=0.06,
        sigma=0.07,
        correlation=corr,
        wage_base=1.0,
        wage_weights=wage_weights[:K],
        diet_cost=0.15,
        kappa_d=0.0,
        mu_e=2.0,
        nu_e=8.0,
        tbar=1.0,
        sleep=0.3,
        e_max=0.2,
        l_min=0.05,
        n_min=0.2,
        d_min=0.0,
        d_max=1.0,
        device=device,
        dtype=dtype,
    )


def create_networks(
    state_dim: int,
    bounds_fn,
    hidden_dims: tuple = (128, 64),
) -> tuple[ValueNetwork, torch.nn.Module]:
    value_net = ValueNetwork(
        input_dim=state_dim,
        hidden_dims=hidden_dims,
        activation="silu",
    )

    base_policy = PolicyNetwork(
        input_dim=state_dim,
        output_dim=3,
        hidden_dims=hidden_dims,
        activation="relu",
        output_bounds=None,
    )

    policy_net = StateBoundedPolicy(base_policy, bounds_fn)

    return value_net, policy_net


def compute_hjb_stats(
    model: HealthManagement,
    value_net: ValueNetwork,
    policy_net: torch.nn.Module,
    n_samples: int = 2000,
    device=None,
    dtype=None,
) -> dict:
    with torch.no_grad():
        s = model.sample_states(n_samples)
        s = s.to(device=device, dtype=dtype)
        u = policy_net(s)

        hjb = compute_hjb_residual(
            model,
            s,
            u,
            value_net,
            ito_method="finite_diff",
            h=1e-2,
            stencil="five",
        )

        return {
            "mean": hjb.mean().item(),
            "std": hjb.std().item(),
            "max_abs": hjb.abs().max().item(),
        }


def plot_policy_1d(
    model: HealthManagement,
    value_net: ValueNetwork,
    policy_net: torch.nn.Module,
    age: float,
    save_path: Optional[pathlib.Path] = None,
    device=None,
    dtype=None,
) -> None:
    if not HAS_MATPLOTLIB:
        return

    h_grid = torch.linspace(-0.6, 0.6, 120, device=device, dtype=dtype)
    age_vec = torch.full_like(h_grid, age)

    s = torch.stack([h_grid, age_vec], dim=1)

    with torch.no_grad():
        u = policy_net(s)
        v = value_net(s).squeeze()

    e = u[:, 0].cpu().numpy()
    d = u[:, 1].cpu().numpy()
    l = u[:, 2].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h_grid.cpu().numpy(), e, label="Exercise")
    axes[0].plot(h_grid.cpu().numpy(), d, label="Diet")
    axes[0].plot(h_grid.cpu().numpy(), l, label="Leisure")
    axes[0].set_xlabel("Health h1")
    axes[0].set_ylabel("Control")
    axes[0].set_title(f"Policy at age {age:.0f}")
    axes[0].legend()

    axes[1].plot(h_grid.cpu().numpy(), v.cpu().numpy())
    axes[1].set_xlabel("Health h1")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Value function slice")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved policy slice to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_policy_age(
    model: HealthManagement,
    policy_net: torch.nn.Module,
    save_path: Optional[pathlib.Path] = None,
    device=None,
    dtype=None,
) -> None:
    if not HAS_MATPLOTLIB:
        return

    ages = torch.linspace(model.age_min, model.age_max, 120, device=device, dtype=dtype)
    h_zero = torch.zeros((120, model.K), device=device, dtype=dtype)
    s = torch.cat([h_zero, ages.unsqueeze(1)], dim=1)

    with torch.no_grad():
        u = policy_net(s)

    e = u[:, 0].cpu().numpy()
    d = u[:, 1].cpu().numpy()
    l = u[:, 2].cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ages.cpu().numpy(), e, label="Exercise")
    ax.plot(ages.cpu().numpy(), d, label="Diet")
    ax.plot(ages.cpu().numpy(), l, label="Leisure")
    ax.set_xlabel("Age")
    ax.set_ylabel("Control")
    ax.set_title("Lifecycle policy at average health")
    ax.legend()

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved age policy plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_policy_2d(
    model: HealthManagement,
    policy_net: torch.nn.Module,
    age: float,
    save_path: Optional[pathlib.Path] = None,
    device=None,
    dtype=None,
) -> None:
    if not HAS_MATPLOTLIB:
        return

    if model.K < 2:
        return

    h_range = 0.6
    n_grid = 50
    h1 = torch.linspace(-h_range, h_range, n_grid, device=device, dtype=dtype)
    h2 = torch.linspace(-h_range, h_range, n_grid, device=device, dtype=dtype)

    H1, H2 = torch.meshgrid(h1, h2, indexing="ij")
    h_rest = torch.zeros((n_grid * n_grid, model.K - 2), device=device, dtype=dtype)
    age_vec = torch.full((n_grid * n_grid, 1), age, device=device, dtype=dtype)

    s = torch.cat([
        H1.flatten().unsqueeze(1),
        H2.flatten().unsqueeze(1),
        h_rest,
        age_vec,
    ], dim=1)

    with torch.no_grad():
        u = policy_net(s)

    e = u[:, 0].reshape(n_grid, n_grid).cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(h1.cpu().numpy(), h2.cpu().numpy(), e.T, levels=20, cmap="viridis")
    plt.colorbar(im, ax=ax, label="Exercise")
    ax.set_xlabel("Health h1")
    ax.set_ylabel("Health h2")
    ax.set_title(f"Exercise policy at age {age:.0f}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved 2D policy slice to {save_path}")
    else:
        plt.show()

    plt.close()


def run_experiment(
    K: int,
    n_iterations: int,
    gamma: float,
    correlation: float,
    save_plots: bool,
    plot_dir: Optional[pathlib.Path],
    verbose: bool = True,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    if verbose:
        print("\n" + "=" * 60)
        print(f"Health Management Experiment: K={K}")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Gamma: {gamma}, Correlation: {correlation}")
        print(f"Iterations: {n_iterations}")

    model = create_model(K, gamma, correlation, device=device, dtype=dtype)

    value_net, policy_net = create_networks(model.state_dim, model.control_bounds)
    value_net = value_net.to(device=device, dtype=dtype)
    policy_net = policy_net.to(device=device, dtype=dtype)

    if K >= 6:
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

    if verbose:
        print("\nTraining...")

    start_time = time.time()
    history = trainer.train(n_iterations)
    elapsed = time.time() - start_time

    if verbose:
        print(f"Training complete in {elapsed:.2f}s")
        print(f"Final value loss: {history.value_loss[-1]:.6f}")
        if history.policy_loss[-1] is not None:
            print(f"Final policy loss: {history.policy_loss[-1]:.6f}")

    hjb_stats = compute_hjb_stats(model, value_net, policy_net, device=device, dtype=dtype)

    if verbose:
        print("\nHJB Residual Stats:")
        print(f"  Mean: {hjb_stats['mean']:.6f}")
        print(f"  Std:  {hjb_stats['std']:.6f}")
        print(f"  Max:  {hjb_stats['max_abs']:.6f}")

    if save_plots and HAS_MATPLOTLIB:
        if plot_dir is None:
            plot_dir = ROOT / "plots"

        if K == 1:
            plot_policy_1d(
                model,
                value_net,
                policy_net,
                age=30.0,
                save_path=plot_dir / f"health_policy_1d_K{K}_gamma{gamma}.png",
                device=device,
                dtype=dtype,
            )

        plot_policy_age(
            model,
            policy_net,
            save_path=plot_dir / f"health_policy_age_K{K}_gamma{gamma}.png",
            device=device,
            dtype=dtype,
        )

        if K >= 2:
            plot_policy_2d(
                model,
                policy_net,
                age=40.0,
                save_path=plot_dir / f"health_policy_2d_K{K}_gamma{gamma}.png",
                device=device,
                dtype=dtype,
            )

    return {
        "K": K,
        "gamma": gamma,
        "correlation": correlation,
        "iterations": n_iterations,
        "elapsed": elapsed,
        "final_value_loss": history.value_loss[-1],
        "final_policy_loss": history.policy_loss[-1],
        "hjb_stats": hjb_stats,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Health management experiment")
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--correlation", type=float, default=0.0)
    parser.add_argument("--no-plots", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(42)

    run_experiment(
        K=args.K,
        n_iterations=args.iterations,
        gamma=args.gamma,
        correlation=args.correlation,
        save_plots=not args.no_plots,
        plot_dir=ROOT / "plots",
    )


if __name__ == "__main__":
    main()
