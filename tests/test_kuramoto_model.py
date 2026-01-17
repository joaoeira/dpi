"""Tests for Kuramoto model and angle embedding."""

import math
import pathlib
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpi.models.kuramoto import (
    KuramotoModel,
    SwingModel,
    ring_graph,
    watts_strogatz_graph,
)
from dpi.networks.angle_embedding import (
    PeriodicAngleEmbedding,
    PeriodicValueNetwork,
    PeriodicStateBoundedPolicy,
)
from dpi.core.ito import compute_drift_hyperdual
from dpi.core.hjb import compute_hjb_residual


class TestGraphConstruction(unittest.TestCase):
    """Test graph construction utilities."""

    def test_ring_graph_shape(self):
        N, k = 10, 2
        src, dst = ring_graph(N, k)
        expected_edges = 2 * N * k
        self.assertEqual(src.shape[0], expected_edges)
        self.assertEqual(dst.shape[0], expected_edges)

    def test_ring_graph_valid_indices(self):
        N, k = 20, 3
        src, dst = ring_graph(N, k)
        self.assertTrue((src >= 0).all() and (src < N).all())
        self.assertTrue((dst >= 0).all() and (dst < N).all())

    def test_ring_graph_no_self_loops(self):
        N, k = 15, 2
        src, dst = ring_graph(N, k)
        self.assertFalse((src == dst).any())

    def test_watts_strogatz_shape(self):
        N, k = 20, 2
        src, dst = watts_strogatz_graph(N, k, p=0.1, seed=0)
        # Should have approximately same number of edges as ring
        expected = 2 * N * k
        self.assertGreater(src.shape[0], expected * 0.5)
        self.assertLess(src.shape[0], expected * 1.5)


class TestKuramotoModel(unittest.TestCase):
    """Test Kuramoto model implementation."""

    def setUp(self):
        self.N = 10
        self.model = KuramotoModel(
            N=self.N,
            K=1.0,
            sigma=0.1,
            device=None,
            dtype=torch.float64,
        )

    def test_order_parameter_range(self):
        """Order parameter R should be in [0, 1]."""
        theta = self.model.sample_states(100)
        R = self.model.order_parameter(theta)

        self.assertTrue((R >= 0.0).all())
        self.assertTrue((R <= 1.0 + 1e-6).all())

    def test_order_parameter_synchronized(self):
        """R should be 1 when all oscillators have same phase."""
        theta = torch.full((10, self.N), 1.5, dtype=torch.float64)
        R = self.model.order_parameter(theta)

        self.assertTrue(torch.allclose(R, torch.ones_like(R), atol=1e-6))

    def test_order_parameter_uniform(self):
        """R should be near 0 for uniformly spaced phases."""
        theta = torch.linspace(0, 2 * math.pi, self.N + 1)[:-1]
        theta = theta.unsqueeze(0).to(torch.float64)
        R = self.model.order_parameter(theta)

        self.assertTrue(R.item() < 0.15)

    def test_drift_shape(self):
        """Drift should have same shape as state."""
        theta = self.model.sample_states(32)
        u = torch.zeros((32, self.model.M), dtype=torch.float64)
        drift = self.model.drift(theta, u)

        self.assertEqual(drift.shape, theta.shape)

    def test_diffusion_shape(self):
        """Diffusion should have shape (batch, N, N) for N shocks."""
        theta = self.model.sample_states(16)
        diffusion = self.model.diffusion(theta)

        self.assertEqual(diffusion.shape, (16, self.N, self.N))

    def test_utility_sign(self):
        """Utility should be negative or zero (since -(1-R^2) <= 0)."""
        theta = self.model.sample_states(50)
        u = torch.zeros((50, self.model.M), dtype=torch.float64)
        utility = self.model.utility(theta, u)

        self.assertTrue((utility <= 1e-6).all())

    def test_utility_maximized_at_sync(self):
        """Utility should be highest when synchronized."""
        theta_sync = torch.full((10, self.N), 0.0, dtype=torch.float64)
        u_zero = torch.zeros((10, self.model.M), dtype=torch.float64)

        theta_rand = self.model.sample_states(10)

        utility_sync = self.model.utility(theta_sync, u_zero).mean()
        utility_rand = self.model.utility(theta_rand, u_zero).mean()

        self.assertGreater(utility_sync, utility_rand)

    def test_control_bounds(self):
        """Control bounds should be symmetric around 0."""
        theta = self.model.sample_states(20)
        c_min, c_max = self.model.control_bounds(theta)

        self.assertEqual(c_min.shape, (20, self.model.M))
        self.assertEqual(c_max.shape, (20, self.model.M))
        self.assertTrue(torch.allclose(c_min, -c_max))

    def test_sparse_coupling_zero_at_uniform_phase(self):
        """Coupling should be zero when all phases are equal."""
        theta = torch.full((5, self.N), 0.5, dtype=torch.float64)
        coupling = self.model._sparse_coupling(theta)

        self.assertTrue(torch.allclose(coupling, torch.zeros_like(coupling), atol=1e-10))

    def test_partial_actuation(self):
        """Test partial actuation with control mask."""
        control_mask = torch.zeros(self.N, dtype=torch.bool)
        control_mask[:3] = True  # Only first 3 nodes controlled

        model = KuramotoModel(
            N=self.N,
            K=1.0,
            sigma=0.1,
            control_mask=control_mask,
            dtype=torch.float64,
        )

        self.assertEqual(model.M, 3)

        theta = model.sample_states(8)
        u = torch.ones((8, 3), dtype=torch.float64)
        u_full = model._expand_control(u, 8)

        self.assertEqual(u_full.shape, (8, self.N))
        self.assertTrue((u_full[:, :3] == 1.0).all())
        self.assertTrue((u_full[:, 3:] == 0.0).all())


class TestSwingModel(unittest.TestCase):
    """Test second-order swing model."""

    def setUp(self):
        self.N = 8
        self.model = SwingModel(
            N=self.N,
            K=1.0,
            sigma=0.1,
            alpha=0.1,
            device=None,
            dtype=torch.float64,
        )

    def test_state_dim(self):
        """Swing model has 2N state dimensions."""
        self.assertEqual(self.model.state_dim, 2 * self.N)

    def test_sample_states_shape(self):
        """Sample states should have correct shape."""
        states = self.model.sample_states(20)
        self.assertEqual(states.shape, (20, 2 * self.N))

    def test_drift_shape(self):
        """Drift should have shape (batch, 2N)."""
        states = self.model.sample_states(16)
        u = torch.zeros((16, self.model.M), dtype=torch.float64)
        drift = self.model.drift(states, u)

        self.assertEqual(drift.shape, (16, 2 * self.N))

    def test_diffusion_shape(self):
        """Diffusion should have shape (batch, 2N, N)."""
        states = self.model.sample_states(10)
        diffusion = self.model.diffusion(states)

        self.assertEqual(diffusion.shape, (10, 2 * self.N, self.N))

    def test_diffusion_structure(self):
        """Diffusion should be zero in theta components, nonzero in omega."""
        states = self.model.sample_states(5)
        diffusion = self.model.diffusion(states)

        # First N rows (theta) should be zero
        self.assertTrue((diffusion[:, :self.N, :] == 0).all())
        # Last N rows (omega) should have nonzero diagonal
        self.assertTrue((diffusion[:, self.N:, :].abs().sum() > 0))


class TestPeriodicAngleEmbedding(unittest.TestCase):
    """Test periodic angle embedding."""

    def test_embedding_shape(self):
        """Output should double angle dimensions."""
        embed = PeriodicAngleEmbedding(n_angles=5, n_passthrough=0)
        x = torch.randn(10, 5)
        y = embed(x)

        self.assertEqual(y.shape, (10, 10))

    def test_embedding_with_passthrough(self):
        """Passthrough dimensions preserved."""
        embed = PeriodicAngleEmbedding(n_angles=5, n_passthrough=3)
        x = torch.randn(10, 8)
        y = embed(x)

        self.assertEqual(y.shape, (10, 13))  # 2*5 + 3

    def test_periodicity(self):
        """theta and theta + 2*pi should produce same embedding."""
        embed = PeriodicAngleEmbedding(n_angles=4, n_passthrough=0)

        theta = torch.randn(10, 4)
        theta_shifted = theta + 2 * math.pi

        y1 = embed(theta)
        y2 = embed(theta_shifted)

        self.assertTrue(torch.allclose(y1, y2, atol=1e-6))

    def test_embedding_values(self):
        """Check that embedding is [sin, cos]."""
        embed = PeriodicAngleEmbedding(n_angles=3, n_passthrough=0)

        theta = torch.tensor([[0.0, math.pi / 2, math.pi]])
        y = embed(theta)

        expected = torch.tensor([[0.0, 1.0, 0.0, 1.0, 0.0, -1.0]])
        self.assertTrue(torch.allclose(y, expected, atol=1e-6))


class TestVectorizedIto(unittest.TestCase):
    """Test vectorized shock batching in ito.py."""

    def test_vectorized_matches_sequential(self):
        """Vectorized implementation should match sequential."""
        torch.manual_seed(123)

        B, n, m = 8, 5, 10
        s = torch.randn(B, n, dtype=torch.float64)
        drift = torch.randn(B, n, dtype=torch.float64)
        G = torch.randn(B, n, m, dtype=torch.float64)

        def V(x):
            return (x ** 2).sum(dim=1, keepdim=True)

        # Sequential (legacy)
        result_seq = compute_drift_hyperdual(
            V, s, drift, G,
            h=1e-3, stencil="five", method="finite_diff",
            vectorized=False,
        )

        # Vectorized
        result_vec = compute_drift_hyperdual(
            V, s, drift, G,
            h=1e-3, stencil="five", method="finite_diff",
            vectorized=True,
        )

        self.assertTrue(torch.allclose(result_seq, result_vec, rtol=1e-4, atol=1e-5))

    def test_vectorized_with_chunking(self):
        """Chunked vectorized should match non-chunked."""
        torch.manual_seed(456)

        B, n, m = 4, 3, 20
        s = torch.randn(B, n, dtype=torch.float64)
        drift = torch.randn(B, n, dtype=torch.float64)
        G = torch.randn(B, n, m, dtype=torch.float64)

        def V(x):
            return x.sum(dim=1, keepdim=True)

        # No chunking
        result_full = compute_drift_hyperdual(
            V, s, drift, G,
            h=1e-3, stencil="three", method="finite_diff",
            vectorized=True, chunk_size=None,
        )

        # With chunking
        result_chunked = compute_drift_hyperdual(
            V, s, drift, G,
            h=1e-3, stencil="three", method="finite_diff",
            vectorized=True, chunk_size=5,
        )

        self.assertTrue(torch.allclose(result_full, result_chunked, rtol=1e-5, atol=1e-6))

    def test_vectorized_different_stencils(self):
        """Vectorized should work with all stencil types."""
        torch.manual_seed(789)

        B, n, m = 4, 3, 5
        s = torch.randn(B, n, dtype=torch.float64)
        drift = torch.randn(B, n, dtype=torch.float64)
        G = torch.randn(B, n, m, dtype=torch.float64)

        def V(x):
            return (x ** 2).sum(dim=1, keepdim=True)

        for stencil in ("three", "five", "seven", "nine"):
            result = compute_drift_hyperdual(
                V, s, drift, G,
                h=1e-3, stencil=stencil, method="finite_diff",
                vectorized=True,
            )
            self.assertEqual(result.shape, (B, 1))
            self.assertFalse(result.isnan().any())


class TestKuramotoIntegration(unittest.TestCase):
    """Integration tests for Kuramoto with DPI components."""

    def test_hjb_residual_computes(self):
        """HJB residual should compute without error."""
        model = KuramotoModel(N=5, K=1.0, sigma=0.1, dtype=torch.float64)

        def V(x):
            return (x ** 2).sum(dim=1, keepdim=True)

        theta = model.sample_states(10)
        u = torch.zeros((10, model.M), dtype=torch.float64)

        hjb = compute_hjb_residual(
            model, theta, u, V,
            ito_method="finite_diff",
            h=1e-2,
            stencil="three",
            vectorized=True,
        )

        self.assertEqual(hjb.shape, (10, 1))
        self.assertFalse(hjb.isnan().any())

    def test_periodic_value_network_forward(self):
        """PeriodicValueNetwork should compute without error."""
        N = 10
        model = KuramotoModel(N=N, K=1.0, sigma=0.1, dtype=torch.float32)

        value_net = PeriodicValueNetwork(
            n_angles=N,
            n_passthrough=0,
            hidden_dims=(32, 16),
            activation="silu",
        )

        theta = model.sample_states(8).float()
        v = value_net(theta)

        self.assertEqual(v.shape, (8, 1))
        self.assertFalse(v.isnan().any())

    def test_periodic_policy_network_forward(self):
        """PeriodicStateBoundedPolicy should compute without error."""
        N = 10
        M = 3
        control_mask = torch.zeros(N, dtype=torch.bool)
        control_mask[:M] = True

        model = KuramotoModel(
            N=N, K=1.0, sigma=0.1,
            control_mask=control_mask,
            dtype=torch.float32,
        )

        policy_net = PeriodicStateBoundedPolicy(
            n_angles=N,
            output_dim=M,
            bounds_fn=model.control_bounds,
            n_passthrough=0,
            hidden_dims=(32, 16),
            activation="relu",
        )

        theta = model.sample_states(8).float()
        u = policy_net(theta)

        self.assertEqual(u.shape, (8, M))
        self.assertFalse(u.isnan().any())

        # Check bounds are respected
        c_min, c_max = model.control_bounds(theta)
        self.assertTrue((u >= c_min - 1e-6).all())
        self.assertTrue((u <= c_max + 1e-6).all())


if __name__ == "__main__":
    unittest.main()
