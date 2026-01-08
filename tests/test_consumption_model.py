"""Unit tests for ConsumptionSmoothing model shapes and behavior."""

import unittest
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dpi.models.consumption_smoothing import ConsumptionSmoothing


class TestConsumptionModelShapes(unittest.TestCase):
    """Test that model outputs have correct shapes."""

    def setUp(self):
        torch.manual_seed(42)

    def test_drift_shape_K1(self):
        """Drift output shape for K=1."""
        model = ConsumptionSmoothing(K=1)
        batch_size = 32
        s = model.sample_states(batch_size)
        c = torch.rand((batch_size, 1)) * 2

        drift = model.drift(s, c)

        self.assertEqual(drift.shape, (batch_size, 2))  # K+1 = 2

    def test_drift_shape_K5(self):
        """Drift output shape for K=5."""
        model = ConsumptionSmoothing(K=5)
        batch_size = 64
        s = model.sample_states(batch_size)
        c = torch.rand((batch_size, 1)) * 2

        drift = model.drift(s, c)

        self.assertEqual(drift.shape, (batch_size, 6))  # K+1 = 6

    def test_diffusion_list_length_K1(self):
        """Diffusion returns K tensors for K=1."""
        model = ConsumptionSmoothing(K=1)
        s = model.sample_states(16)

        diffusion = model.diffusion(s)

        self.assertEqual(len(diffusion), 1)  # K shocks

    def test_diffusion_list_length_K5(self):
        """Diffusion returns K tensors for K=5."""
        model = ConsumptionSmoothing(K=5)
        s = model.sample_states(16)

        diffusion = model.diffusion(s)

        self.assertEqual(len(diffusion), 5)  # K shocks

    def test_diffusion_tensor_shapes(self):
        """Each diffusion tensor has shape (batch, K+1)."""
        model = ConsumptionSmoothing(K=3)
        batch_size = 32
        s = model.sample_states(batch_size)

        diffusion = model.diffusion(s)

        for g in diffusion:
            self.assertEqual(g.shape, (batch_size, 4))  # K+1 = 4

    def test_diffusion_wealth_is_zero(self):
        """Diffusion in wealth (first column) is zero."""
        model = ConsumptionSmoothing(K=3)
        s = model.sample_states(16)

        diffusion = model.diffusion(s)

        for g in diffusion:
            # First column (wealth) should be zero
            self.assertTrue(torch.allclose(g[:, 0], torch.zeros_like(g[:, 0])))

    def test_utility_shape(self):
        """Utility output has shape (batch, 1)."""
        model = ConsumptionSmoothing(K=2)
        batch_size = 32
        s = model.sample_states(batch_size)
        c = torch.rand((batch_size, 1)) + 0.1  # Positive consumption

        utility = model.utility(s, c)

        self.assertEqual(utility.shape, (batch_size, 1))

    def test_sample_states_shape(self):
        """Sample states has correct shape (n, K+1)."""
        model = ConsumptionSmoothing(K=4)
        n = 100

        s = model.sample_states(n)

        self.assertEqual(s.shape, (n, 5))  # K+1 = 5

    def test_income_shape(self):
        """Income function has correct shape."""
        model = ConsumptionSmoothing(K=3)
        batch_size = 32
        x = torch.randn((batch_size, 3))

        Y = model.income(x)

        self.assertEqual(Y.shape, (batch_size, 1))

    def test_income_positive(self):
        """Income is always positive."""
        model = ConsumptionSmoothing(K=3, Ybar=1.0)
        x = torch.randn((100, 3))

        Y = model.income(x)

        self.assertTrue((Y > 0).all())


class TestConsumptionModelBehavior(unittest.TestCase):
    """Test economic behavior of the model."""

    def setUp(self):
        torch.manual_seed(42)

    def test_discount_rate(self):
        """Discount rate property returns correct value."""
        model = ConsumptionSmoothing(K=1, rho=0.05)
        self.assertAlmostEqual(model.discount_rate, 0.05)

    def test_state_dim(self):
        """State dimension is K+1."""
        model = ConsumptionSmoothing(K=5)
        self.assertEqual(model.state_dim, 6)

    def test_wealth_drift_depends_on_consumption(self):
        """Higher consumption reduces wealth drift."""
        model = ConsumptionSmoothing(K=1, r=0.02)
        s = model.sample_states(32)

        c_low = torch.full((32, 1), 0.5)
        c_high = torch.full((32, 1), 2.0)

        drift_low = model.drift(s, c_low)
        drift_high = model.drift(s, c_high)

        # Wealth drift (first column) should be lower with higher consumption
        self.assertTrue((drift_low[:, 0] > drift_high[:, 0]).all())

    def test_factor_drift_mean_reversion(self):
        """Factor drift shows mean reversion toward zero."""
        model = ConsumptionSmoothing(K=2, theta=0.5)

        # Positive factors should have negative drift
        s_pos = torch.tensor([[5.0, 0.5, 0.3]])
        drift_pos = model.drift(s_pos, torch.tensor([[1.0]]))
        self.assertTrue(drift_pos[0, 1] < 0)  # x1 drift negative
        self.assertTrue(drift_pos[0, 2] < 0)  # x2 drift negative

        # Negative factors should have positive drift
        s_neg = torch.tensor([[5.0, -0.5, -0.3]])
        drift_neg = model.drift(s_neg, torch.tensor([[1.0]]))
        self.assertTrue(drift_neg[0, 1] > 0)  # x1 drift positive
        self.assertTrue(drift_neg[0, 2] > 0)  # x2 drift positive

    def test_utility_increases_with_consumption(self):
        """Higher consumption gives higher utility."""
        model = ConsumptionSmoothing(K=1, gamma=2.0)
        s = model.sample_states(1)

        c_low = torch.tensor([[0.5]])
        c_high = torch.tensor([[2.0]])

        u_low = model.utility(s, c_low)
        u_high = model.utility(s, c_high)

        self.assertTrue(u_high > u_low)

    def test_crra_utility_formula(self):
        """CRRA utility formula is correct."""
        gamma = 2.0
        model = ConsumptionSmoothing(K=1, gamma=gamma)
        c = torch.tensor([[2.0]])
        s = model.sample_states(1)

        utility = model.utility(s, c)

        expected = (c ** (1 - gamma)) / (1 - gamma)
        self.assertTrue(torch.allclose(utility, expected))

    def test_income_exponential_in_factors(self):
        """Income Y = Ybar * exp(sum_k beta_k * x_k)."""
        model = ConsumptionSmoothing(K=2, Ybar=1.0, beta=1.0)
        x = torch.tensor([[0.5, 0.3]])

        Y = model.income(x)

        expected = 1.0 * torch.exp(torch.tensor([0.5 + 0.3]))
        self.assertTrue(torch.allclose(Y.squeeze(), expected, rtol=1e-5))


class TestConsumptionModelCorrelation(unittest.TestCase):
    """Test correlation handling in diffusion."""

    def setUp(self):
        torch.manual_seed(42)

    def test_uncorrelated_diffusion_diagonal(self):
        """Uncorrelated factors have diagonal diffusion."""
        model = ConsumptionSmoothing(K=2, sigma=0.2, correlation=None)
        s = model.sample_states(1)

        diffusion = model.diffusion(s)

        # With no correlation, shock 1 only affects factor 1, shock 2 only factor 2
        # diffusion[0] should have non-zero only in position 1 (factor 1)
        # diffusion[1] should have non-zero only in position 2 (factor 2)
        self.assertTrue(torch.allclose(diffusion[0][:, 2], torch.zeros(1)))
        self.assertTrue(torch.allclose(diffusion[1][:, 1], torch.zeros(1)))

    def test_correlated_diffusion_off_diagonal(self):
        """Correlated factors have off-diagonal diffusion terms."""
        corr = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
        model = ConsumptionSmoothing(K=2, sigma=0.2, correlation=corr)
        s = model.sample_states(1)

        diffusion = model.diffusion(s)

        # With correlation, shocks affect both factors
        # The Cholesky factor of [[1, 0.5], [0.5, 1]] is [[1, 0], [0.5, ~0.866]]
        # So shock 1 affects both factors, shock 2 only affects factor 2
        self.assertFalse(torch.allclose(diffusion[0][:, 1], torch.zeros(1)))
        self.assertFalse(torch.allclose(diffusion[0][:, 2], torch.zeros(1)))


class TestConsumptionModelSampling(unittest.TestCase):
    """Test state sampling."""

    def setUp(self):
        torch.manual_seed(42)

    def test_wealth_in_bounds(self):
        """Sampled wealth is within specified bounds."""
        model = ConsumptionSmoothing(K=1, w_min=0.5, w_max=5.0)
        s = model.sample_states(1000)

        wealth = s[:, 0]
        self.assertTrue((wealth >= 0.5).all())
        self.assertTrue((wealth <= 5.0).all())

    def test_factors_reasonable_range(self):
        """Sampled factors are within reasonable range."""
        model = ConsumptionSmoothing(K=2, sigma=0.2, theta=0.5, x_std_range=3.0)
        s = model.sample_states(1000)

        factors = s[:, 1:]
        # Stationary std = sigma / sqrt(2*theta) = 0.2 / sqrt(1) = 0.2
        # Range should be +/- 3 * 0.2 = 0.6
        self.assertTrue((factors.abs() <= 0.6 + 1e-6).all())


class TestConsumptionModelDeviceAndDtype(unittest.TestCase):
    """Test device and dtype handling."""

    def setUp(self):
        torch.manual_seed(42)

    def test_float64_dtype(self):
        """Model works with float64."""
        model = ConsumptionSmoothing(K=1, dtype=torch.float64)
        s = model.sample_states(16)

        self.assertEqual(s.dtype, torch.float64)

        drift = model.drift(s, torch.rand((16, 1), dtype=torch.float64))
        self.assertEqual(drift.dtype, torch.float64)

    def test_device_propagation(self):
        """Model uses specified device."""
        device = torch.device("cpu")
        model = ConsumptionSmoothing(K=1, device=device)
        s = model.sample_states(16)

        self.assertEqual(s.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
