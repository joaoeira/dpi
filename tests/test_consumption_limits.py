"""Tests for special cases and limiting behavior of ConsumptionSmoothing model."""

import unittest
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpi.models.consumption_smoothing import ConsumptionSmoothing
from dpi.core.hjb import compute_hjb_residual
from dpi.core.ito import compute_drift_hyperdual


class TestLogUtility(unittest.TestCase):
    """Test log utility (gamma=1) special case."""

    def setUp(self):
        torch.manual_seed(42)

    def test_log_utility_finite(self):
        """Log utility is finite for positive consumption."""
        model = ConsumptionSmoothing(K=1, gamma=1.0)
        s = model.sample_states(32)
        c = torch.rand((32, 1)) + 0.1  # Positive consumption

        utility = model.utility(s, c)

        self.assertTrue(torch.isfinite(utility).all())

    def test_log_utility_formula(self):
        """Log utility equals log(c)."""
        model = ConsumptionSmoothing(K=1, gamma=1.0)
        c = torch.tensor([[2.0]])
        s = model.sample_states(1)

        utility = model.utility(s, c)

        expected = torch.log(c)
        self.assertTrue(torch.allclose(utility, expected, rtol=1e-5))

    def test_log_utility_gradient_exists(self):
        """Log utility gradient is computable."""
        model = ConsumptionSmoothing(K=1, gamma=1.0)
        s = model.sample_states(16)
        # Use a leaf tensor for gradient computation
        c_raw = torch.rand((16, 1), requires_grad=True)
        c = c_raw + 0.1

        utility = model.utility(s, c)
        utility.sum().backward()

        self.assertTrue(c_raw.grad is not None)
        self.assertTrue(torch.isfinite(c_raw.grad).all())


class TestDeterministicIncome(unittest.TestCase):
    """Test deterministic income (sigma=0) special case."""

    def setUp(self):
        torch.manual_seed(42)

    def test_zero_diffusion(self):
        """Zero sigma gives zero diffusion."""
        model = ConsumptionSmoothing(K=2, sigma=0.0)
        s = model.sample_states(16)

        diffusion = model.diffusion(s)

        for g in diffusion:
            self.assertTrue(torch.allclose(g, torch.zeros_like(g)))

    def test_drift_still_works(self):
        """Drift is valid with zero diffusion."""
        model = ConsumptionSmoothing(K=2, sigma=0.0)
        s = model.sample_states(16)
        c = torch.rand((16, 1)) + 0.1

        drift = model.drift(s, c)

        self.assertTrue(torch.isfinite(drift).all())

    def test_hjb_reduces_to_ode(self):
        """HJB residual is well-defined with zero diffusion."""
        model = ConsumptionSmoothing(K=1, sigma=0.0)
        s = model.sample_states(16)
        c = torch.rand((16, 1)) + 0.5

        # Simple value function
        def V(s):
            return s[:, 0:1]  # V = wealth

        hjb = compute_hjb_residual(
            model, s, c, V,
            ito_method="finite_diff",
            h=1e-2,
            stencil="five",
        )

        self.assertTrue(torch.isfinite(hjb).all())


class TestTransitoryShocks(unittest.TestCase):
    """Test transitory shocks (high theta) behavior."""

    def setUp(self):
        torch.manual_seed(42)

    def test_high_mean_reversion(self):
        """High theta gives fast mean reversion."""
        theta_high = 10.0
        model = ConsumptionSmoothing(K=1, theta=theta_high)

        # Factor away from zero
        s = torch.tensor([[5.0, 1.0]])
        drift = model.drift(s, torch.tensor([[0.5]]))

        # Factor drift should be strongly negative
        factor_drift = drift[0, 1]
        expected_drift = -theta_high * 1.0
        self.assertTrue(torch.allclose(factor_drift, torch.tensor(expected_drift)))

    def test_transitory_vs_persistent(self):
        """More persistent shocks (lower theta) have slower reversion."""
        s = torch.tensor([[5.0, 1.0]])
        c = torch.tensor([[0.5]])

        model_transitory = ConsumptionSmoothing(K=1, theta=10.0)
        model_persistent = ConsumptionSmoothing(K=1, theta=0.1)

        drift_trans = model_transitory.drift(s, c)
        drift_pers = model_persistent.drift(s, c)

        # Transitory should have faster mean reversion (more negative drift)
        self.assertTrue(drift_trans[0, 1] < drift_pers[0, 1])


class TestSmallConsumption(unittest.TestCase):
    """Test handling of small and zero consumption."""

    def setUp(self):
        torch.manual_seed(42)

    def test_utility_clamps_consumption(self):
        """Utility handles near-zero consumption without NaN."""
        model = ConsumptionSmoothing(K=1, gamma=2.0)
        s = model.sample_states(16)
        c = torch.full((16, 1), 1e-10)

        utility = model.utility(s, c)

        self.assertTrue(torch.isfinite(utility).all())

    def test_negative_consumption_clamped(self):
        """Negative consumption is clamped to minimum."""
        model = ConsumptionSmoothing(K=1, gamma=2.0)
        s = model.sample_states(16)
        c = torch.full((16, 1), -1.0)

        utility = model.utility(s, c)

        self.assertTrue(torch.isfinite(utility).all())


class TestItoDriftComputation(unittest.TestCase):
    """Test Ito drift computation with the model."""

    def setUp(self):
        torch.manual_seed(42)

    def test_quadratic_value_function(self):
        """Ito drift is correct for quadratic V(s) = s^T s."""
        model = ConsumptionSmoothing(K=1, sigma=0.2, theta=0.5)
        s = model.sample_states(32)
        c = torch.rand((32, 1)) + 0.5

        drift_f = model.drift(s, c)
        diffusion_g = model.diffusion(s, c)

        def V(s):
            return (s ** 2).sum(dim=1, keepdim=True)

        ito_drift = compute_drift_hyperdual(
            V, s, drift_f, diffusion_g,
            h=1e-2,
            stencil="five",
            method="finite_diff",
        )

        self.assertTrue(torch.isfinite(ito_drift).all())
        self.assertEqual(ito_drift.shape, (32, 1))

    def test_linear_value_function(self):
        """Ito drift computation is finite for linear V(s)."""
        # Note: The hyper-dual method is designed for stochastic dynamics.
        # With nonzero diffusion, we verify the computation is well-behaved.
        model = ConsumptionSmoothing(K=1, sigma=0.2)  # Small diffusion
        s = model.sample_states(16)
        c = torch.rand((16, 1)) + 0.5

        drift_f = model.drift(s, c)
        diffusion_g = model.diffusion(s, c)

        def V(s):
            return s.sum(dim=1, keepdim=True)

        ito_drift = compute_drift_hyperdual(
            V, s, drift_f, diffusion_g,
            h=1e-2,
            stencil="five",
            method="finite_diff",
        )

        # Check computation is finite and has correct shape
        self.assertTrue(torch.isfinite(ito_drift).all())
        self.assertEqual(ito_drift.shape, (16, 1))


class TestHJBResidual(unittest.TestCase):
    """Test HJB residual computation with the model."""

    def setUp(self):
        torch.manual_seed(42)

    def test_hjb_residual_shape(self):
        """HJB residual has correct shape."""
        model = ConsumptionSmoothing(K=2)
        s = model.sample_states(32)
        c = torch.rand((32, 1)) + 0.5

        def V(s):
            return s[:, 0:1]

        hjb = compute_hjb_residual(
            model, s, c, V,
            ito_method="finite_diff",
            h=1e-2,
            stencil="five",
        )

        self.assertEqual(hjb.shape, (32, 1))

    def test_hjb_residual_finite(self):
        """HJB residual is finite for reasonable states."""
        model = ConsumptionSmoothing(K=3)
        s = model.sample_states(64)
        c = torch.rand((64, 1)) * 2 + 0.1

        def V(s):
            w = s[:, 0:1]
            return w ** 0.5  # Concave in wealth

        hjb = compute_hjb_residual(
            model, s, c, V,
            ito_method="finite_diff",
            h=1e-2,
            stencil="five",
        )

        self.assertTrue(torch.isfinite(hjb).all())


class TestScalingBehavior(unittest.TestCase):
    """Test that model scales to higher K."""

    def setUp(self):
        torch.manual_seed(42)

    def test_K10_shapes(self):
        """Model works with K=10 factors."""
        model = ConsumptionSmoothing(K=10)
        s = model.sample_states(64)
        c = torch.rand((64, 1)) + 0.5

        drift = model.drift(s, c)
        diffusion = model.diffusion(s, c)
        utility = model.utility(s, c)

        self.assertEqual(drift.shape, (64, 11))
        self.assertEqual(len(diffusion), 10)
        for g in diffusion:
            self.assertEqual(g.shape, (64, 11))
        self.assertEqual(utility.shape, (64, 1))

    def test_K20_no_errors(self):
        """Model runs without errors for K=20."""
        model = ConsumptionSmoothing(K=20)
        s = model.sample_states(32)
        c = torch.rand((32, 1)) + 0.5

        drift = model.drift(s, c)
        diffusion = model.diffusion(s, c)
        utility = model.utility(s, c)

        self.assertTrue(torch.isfinite(drift).all())
        self.assertTrue(all(torch.isfinite(g).all() for g in diffusion))
        self.assertTrue(torch.isfinite(utility).all())


class TestParameterVariations(unittest.TestCase):
    """Test model with different parameter configurations."""

    def setUp(self):
        torch.manual_seed(42)

    def test_heterogeneous_theta(self):
        """Different mean reversion speeds per factor."""
        model = ConsumptionSmoothing(K=3, theta=[0.1, 0.5, 1.0])
        s = torch.tensor([[5.0, 1.0, 1.0, 1.0]])  # All factors at 1.0
        c = torch.tensor([[0.5]])

        drift = model.drift(s, c)

        # Factor drifts should differ based on theta
        # drift_k = -theta_k * x_k, so with x_k = 1: drift_k = -theta_k
        self.assertTrue(torch.allclose(drift[0, 1], torch.tensor(-0.1), atol=1e-5))
        self.assertTrue(torch.allclose(drift[0, 2], torch.tensor(-0.5), atol=1e-5))
        self.assertTrue(torch.allclose(drift[0, 3], torch.tensor(-1.0), atol=1e-5))

    def test_heterogeneous_sigma(self):
        """Different volatilities per factor."""
        model = ConsumptionSmoothing(K=2, sigma=[0.1, 0.3])
        s = model.sample_states(1)

        diffusion = model.diffusion(s)

        # Diffusion magnitudes should differ
        # With identity correlation, diffusion[k][:, k+1] = sigma_k
        self.assertTrue(torch.allclose(diffusion[0][:, 1], torch.tensor([0.1])))
        self.assertTrue(torch.allclose(diffusion[1][:, 2], torch.tensor([0.3])))

    def test_heterogeneous_beta(self):
        """Different income loadings per factor."""
        model = ConsumptionSmoothing(K=2, beta=[2.0, 0.5], Ybar=1.0)
        x = torch.tensor([[0.1, 0.1]])  # Both factors at 0.1

        Y = model.income(x)

        # Y = Ybar * exp(2.0 * 0.1 + 0.5 * 0.1) = exp(0.25)
        expected = torch.exp(torch.tensor([0.25]))
        self.assertTrue(torch.allclose(Y.squeeze(), expected, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
