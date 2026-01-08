"""Consumption smoothing experiment with configurable K factors.

This script runs DPI training for the multi-factor consumption smoothing model
and produces diagnostics including:
- Training loss curves
- HJB residual statistics
- Policy slices (for K=1, K=2)
- Timing benchmarks
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from typing import Optional

import torch
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpi.algorithms.dpi import DPITrainer
from dpi.core.hjb import compute_hjb_residual
from dpi.models.consumption_smoothing import ConsumptionSmoothing
from dpi.networks.policy_network import PolicyNetwork, StateBoundedPolicy
from dpi.networks.value_network import ValueNetwork
from dpi.utils.training import DPIConfig, TrainingHistory

# Plotting imports (optional)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def create_model(
    K: int,
    gamma: float = 2.0,
    rho: float = 0.04,
    r: float = 0.02,
    correlation: float = 0.0,
    device=None,
    dtype=None,
) -> ConsumptionSmoothing:
    """Create a consumption smoothing model with K factors.

    Args:
        K: Number of income factors.
        gamma: Risk aversion.
        rho: Discount rate.
        r: Risk-free rate.
        correlation: Pairwise correlation between factor shocks.
        device: PyTorch device.
        dtype: PyTorch dtype.

    Returns:
        Configured ConsumptionSmoothing model.
    """
    # Build correlation matrix with uniform pairwise correlation
    if K > 1 and correlation != 0.0:
        corr = torch.full((K, K), correlation, device=device, dtype=dtype)
        corr.fill_diagonal_(1.0)
    else:
        corr = None

    return ConsumptionSmoothing(
        K=K,
        rho=rho,
        r=r,
        gamma=gamma,
        Ybar=1.0,
        theta=0.5,  # Mean reversion speed
        sigma=0.2,  # Volatility
        beta=1.0 / K,  # Scale loadings so total income volatility is similar
        correlation=corr,
        w_min=0.1,
        w_max=10.0,
        device=device,
        dtype=dtype,
    )


def create_networks(
    state_dim: int,
    bounds_fn,
    hidden_dims: tuple = (128, 64),
) -> tuple[ValueNetwork, torch.nn.Module]:
    """Create value and policy networks.

    Args:
        state_dim: Dimension of the state (K+1).
        bounds_fn: Callable returning (c_min, c_max) for each state.
        hidden_dims: Hidden layer dimensions.

    Returns:
        Tuple of (value_network, policy_network).
    """
    value_net = ValueNetwork(
        input_dim=state_dim,
        hidden_dims=hidden_dims,
        activation="silu",
    )

    base_policy = PolicyNetwork(
        input_dim=state_dim,
        output_dim=1,
        hidden_dims=hidden_dims,
        activation="relu",
        output_bounds=None,
    )

    policy_net = StateBoundedPolicy(base_policy, bounds_fn)

    return value_net, policy_net


def compute_hjb_stats(
    model: ConsumptionSmoothing,
    value_net: ValueNetwork,
    policy_net: Optional[torch.nn.Module],
    n_samples: int = 1000,
    device=None,
    dtype=None,
) -> dict:
    """Compute HJB residual statistics on a held-out sample.

    Args:
        model: The consumption smoothing model.
        value_net: Trained value network.
        policy_net: Trained policy network (or None).
        n_samples: Number of samples for evaluation.
        device: PyTorch device.
        dtype: PyTorch dtype.

    Returns:
        Dictionary with mean, std, max_abs HJB residual.
    """
    with torch.no_grad():
        s = model.sample_states(n_samples)
        s = s.to(device=device, dtype=dtype)

        c = None
        if policy_net is not None:
            c = policy_net(s)

        hjb = compute_hjb_residual(
            model, s, c, value_net,
            ito_method="finite_diff",
            h=1e-2,
            stencil="five",
        )

        return {
            "mean": hjb.mean().item(),
            "std": hjb.std().item(),
            "max_abs": hjb.abs().max().item(),
        }


def plot_training_curves(
    history: TrainingHistory,
    save_path: Optional[pathlib.Path] = None,
) -> None:
    """Plot training loss curves.

    Args:
        history: Training history from DPITrainer.
        save_path: Path to save the figure.
    """
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Value loss
    axes[0].plot(history.value_loss)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Value Loss")
    axes[0].set_title("Value Network Training")
    axes[0].set_yscale("log")

    # Policy loss
    policy_loss = [x for x in history.policy_loss if x is not None]
    if policy_loss:
        axes[1].plot(policy_loss)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Policy Loss")
        axes[1].set_title("Policy Network Training")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_policy_slice_1d(
    model: ConsumptionSmoothing,
    value_net: ValueNetwork,
    policy_net: torch.nn.Module,
    save_path: Optional[pathlib.Path] = None,
    device=None,
    dtype=None,
) -> None:
    """Plot policy c(w, x) as a function of wealth for fixed x=0.

    Only valid for K=1.

    Args:
        model: Trained model.
        value_net: Trained value network.
        policy_net: Trained policy network.
        save_path: Path to save the figure.
        device: PyTorch device.
        dtype: PyTorch dtype.
    """
    if not HAS_MATPLOTLIB:
        return

    if model.K != 1:
        print("1D policy slice only available for K=1")
        return

    with torch.no_grad():
        # Grid of wealth values
        w_grid = torch.linspace(0.1, 10.0, 100, device=device, dtype=dtype)
        x_fixed = torch.zeros(100, device=device, dtype=dtype)

        s = torch.stack([w_grid, x_fixed], dim=1)  # (100, 2)
        c = policy_net(s).squeeze()
        v = value_net(s).squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(w_grid.cpu().numpy(), c.cpu().numpy())
    axes[0].set_xlabel("Wealth (w)")
    axes[0].set_ylabel("Consumption (c)")
    axes[0].set_title("Policy: c(w, x=0)")

    axes[1].plot(w_grid.cpu().numpy(), v.cpu().numpy())
    axes[1].set_xlabel("Wealth (w)")
    axes[1].set_ylabel("Value (V)")
    axes[1].set_title("Value: V(w, x=0)")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved policy slice to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_policy_slice_2d(
    model: ConsumptionSmoothing,
    policy_net: torch.nn.Module,
    w_fixed: float = 5.0,
    save_path: Optional[pathlib.Path] = None,
    device=None,
    dtype=None,
) -> None:
    """Plot policy c(w, x1, x2) as a heatmap for fixed wealth.

    Only valid for K=2.

    Args:
        model: Trained model.
        policy_net: Trained policy network.
        w_fixed: Fixed wealth value.
        save_path: Path to save the figure.
        device: PyTorch device.
        dtype: PyTorch dtype.
    """
    if not HAS_MATPLOTLIB:
        return

    if model.K != 2:
        print("2D policy slice only available for K=2")
        return

    with torch.no_grad():
        # Grid of factor values
        x_range = 1.0  # +/- range for factors
        n_grid = 50
        x1_grid = torch.linspace(-x_range, x_range, n_grid, device=device, dtype=dtype)
        x2_grid = torch.linspace(-x_range, x_range, n_grid, device=device, dtype=dtype)

        X1, X2 = torch.meshgrid(x1_grid, x2_grid, indexing="ij")
        W = torch.full_like(X1, w_fixed)

        s = torch.stack([W.flatten(), X1.flatten(), X2.flatten()], dim=1)
        c = policy_net(s).squeeze().reshape(n_grid, n_grid)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.contourf(
        x1_grid.cpu().numpy(),
        x2_grid.cpu().numpy(),
        c.cpu().numpy().T,
        levels=20,
        cmap="viridis",
    )
    plt.colorbar(im, ax=ax, label="Consumption (c)")
    ax.set_xlabel("Factor x1")
    ax.set_ylabel("Factor x2")
    ax.set_title(f"Policy: c(w={w_fixed}, x1, x2)")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved 2D policy slice to {save_path}")
    else:
        plt.show()

    plt.close()


def run_experiment(
    K: int,
    n_iterations: int = 1000,
    batch_size: int = 2048,
    gamma: float = 2.0,
    correlation: float = 0.0,
    hidden_dims: tuple = (128, 64),
    value_lr: float = 1e-3,
    policy_lr: float = 1e-4,
    save_plots: bool = True,
    plot_dir: Optional[pathlib.Path] = None,
    verbose: bool = True,
) -> dict:
    """Run a consumption smoothing experiment.

    Args:
        K: Number of income factors.
        n_iterations: Number of training iterations.
        batch_size: Training batch size.
        gamma: Risk aversion coefficient.
        correlation: Pairwise factor shock correlation.
        hidden_dims: Network hidden layer dimensions.
        value_lr: Value network learning rate.
        policy_lr: Policy network learning rate.
        save_plots: Whether to save diagnostic plots.
        plot_dir: Directory for saving plots.
        verbose: Print progress.

    Returns:
        Dictionary with experiment results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    if verbose:
        print(f"\n{'='*60}")
        print(f"Consumption Smoothing Experiment: K={K}")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Gamma: {gamma}, Correlation: {correlation}")
        print(f"Iterations: {n_iterations}, Batch size: {batch_size}")

    # Create model
    model = create_model(
        K=K,
        gamma=gamma,
        correlation=correlation,
        device=device,
        dtype=dtype,
    )

    # Create networks
    state_dim = model.state_dim
    value_net, policy_net = create_networks(
        state_dim=state_dim,
        bounds_fn=model.consumption_bounds,
        hidden_dims=hidden_dims,
    )
    value_net = value_net.to(device=device, dtype=dtype)
    policy_net = policy_net.to(device=device, dtype=dtype)

    # Configure training
    config = DPIConfig(
        value_lr=value_lr,
        policy_lr=policy_lr,
        batch_size=batch_size,
        eval_method="explicit",
        delta_t=1.0,
        n_eval_steps=5,
        n_improve_steps=1,
        ito_method="finite_diff",
        fd_step_size=1e-2,
        fd_stencil="five",
        lr_decay=0.995,
        lr_decay_steps=500,
    )

    # Train
    trainer = DPITrainer(model, value_net, policy_net, config)

    if verbose:
        print(f"\nTraining...")

    start_time = time.time()
    history = trainer.train(n_iterations)
    elapsed_time = time.time() - start_time

    time_per_iter = elapsed_time / n_iterations

    if verbose:
        print(f"Training complete in {elapsed_time:.2f}s ({time_per_iter*1000:.2f}ms/iter)")
        print(f"Final value loss: {history.value_loss[-1]:.6f}")
        if history.policy_loss[-1] is not None:
            print(f"Final policy loss: {history.policy_loss[-1]:.6f}")

    # Compute HJB residual statistics
    hjb_stats = compute_hjb_stats(
        model, value_net, policy_net,
        n_samples=2000,
        device=device,
        dtype=dtype,
    )

    if verbose:
        print(f"\nHJB Residual Stats:")
        print(f"  Mean: {hjb_stats['mean']:.6f}")
        print(f"  Std:  {hjb_stats['std']:.6f}")
        print(f"  Max:  {hjb_stats['max_abs']:.6f}")

    # Save plots
    if save_plots and HAS_MATPLOTLIB:
        if plot_dir is None:
            plot_dir = ROOT / "plots"

        plot_training_curves(
            history,
            save_path=plot_dir / f"training_K{K}_gamma{gamma}_corr{correlation}.png",
        )

        if K == 1:
            plot_policy_slice_1d(
                model, value_net, policy_net,
                save_path=plot_dir / f"policy_1d_K{K}_gamma{gamma}.png",
                device=device,
                dtype=dtype,
            )

        if K == 2:
            plot_policy_slice_2d(
                model, policy_net,
                w_fixed=5.0,
                save_path=plot_dir / f"policy_2d_K{K}_gamma{gamma}_corr{correlation}.png",
                device=device,
                dtype=dtype,
            )

    return {
        "K": K,
        "gamma": gamma,
        "correlation": correlation,
        "n_iterations": n_iterations,
        "elapsed_time": elapsed_time,
        "time_per_iter": time_per_iter,
        "final_value_loss": history.value_loss[-1],
        "final_policy_loss": history.policy_loss[-1],
        "hjb_stats": hjb_stats,
        "history": history,
    }


def run_scaling_benchmark(
    K_values: list[int] = [1, 2, 5, 10, 20],
    n_iterations: int = 100,
    save_plot: bool = True,
    plot_dir: Optional[pathlib.Path] = None,
) -> list[dict]:
    """Run scaling benchmark across different K values.

    Args:
        K_values: List of K values to test.
        n_iterations: Iterations per experiment.
        save_plot: Whether to save timing plot.
        plot_dir: Directory for saving plots.

    Returns:
        List of experiment results.
    """
    print("\n" + "="*60)
    print("Scaling Benchmark")
    print("="*60)

    results = []
    for K in K_values:
        result = run_experiment(
            K=K,
            n_iterations=n_iterations,
            batch_size=1024,
            save_plots=False,
            verbose=True,
        )
        results.append(result)

    # Plot timing
    if save_plot and HAS_MATPLOTLIB:
        if plot_dir is None:
            plot_dir = ROOT / "plots"

        K_arr = [r["K"] for r in results]
        time_arr = [r["time_per_iter"] * 1000 for r in results]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(K_arr, time_arr, "o-", linewidth=2, markersize=8)
        ax.set_xlabel("Number of Factors (K)")
        ax.set_ylabel("Time per Iteration (ms)")
        ax.set_title("Scaling: Training Time vs Number of Factors")
        ax.grid(True, alpha=0.3)

        plot_path = plot_dir / "scaling_benchmark.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        print(f"\nSaved scaling plot to {plot_path}")
        plt.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Consumption Smoothing Experiment")
    parser.add_argument("--K", type=int, default=1, help="Number of income factors")
    parser.add_argument("--iterations", type=int, default=1000, help="Training iterations")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--gamma", type=float, default=2.0, help="Risk aversion")
    parser.add_argument("--correlation", type=float, default=0.0, help="Factor correlation")
    parser.add_argument("--value-lr", type=float, default=1e-3, help="Value learning rate")
    parser.add_argument("--policy-lr", type=float, default=1e-4, help="Policy learning rate")
    parser.add_argument("--no-plots", action="store_true", help="Skip plotting")
    parser.add_argument("--scaling", action="store_true", help="Run scaling benchmark")

    args = parser.parse_args()

    torch.manual_seed(42)

    if args.scaling:
        run_scaling_benchmark(
            K_values=[1, 2, 5, 10],
            n_iterations=100,
        )
    else:
        run_experiment(
            K=args.K,
            n_iterations=args.iterations,
            batch_size=args.batch_size,
            gamma=args.gamma,
            correlation=args.correlation,
            value_lr=args.value_lr,
            policy_lr=args.policy_lr,
            save_plots=not args.no_plots,
        )


if __name__ == "__main__":
    main()
