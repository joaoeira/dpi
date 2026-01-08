# Deep Policy Iteration (DPI) - Python Implementation Plan

## Overview

This document outlines the plan for implementing the Deep Policy Iteration (DPI) method in Python, based on the paper "Machine Learning for Continuous-Time Finance" by Duarte, Duarte, and Silva (2024). The DPI method solves high-dimensional continuous-time stochastic control problems by combining deep neural networks with automatic differentiation.

## Core Innovation

The key insight is that computing the expected drift of a value function V(s) in continuous time can be done via Ito's lemma, which transforms high-dimensional integration into differentiation. The **hyper-dual approach** further reduces this to computing the second derivative of a univariate auxiliary function:

$$F(\epsilon) = \sum_{i=1}^{m} V\left(\mathbf{s} + \frac{\epsilon}{\sqrt{2}}\mathbf{g}_i(\mathbf{s}) + \frac{\epsilon^2}{2m}\mathbf{f}(\mathbf{s})\right)$$

where $F''(0) = \frac{\mathbb{E}[dV]}{dt}$.

This makes the computational cost **independent of the number of state variables** and scales **linearly with the number of shocks**.

---

## Implementation Architecture

### Module Structure

```
dpi/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── ito.py              # Hyper-dual Ito's lemma implementation
│   ├── hjb.py              # HJB residual computation
│   └── sampler.py          # State space sampling utilities
├── networks/
│   ├── __init__.py
│   ├── value_network.py    # Value function neural network
│   └── policy_network.py   # Policy function neural network
├── algorithms/
│   ├── __init__.py
│   ├── dpi.py              # Main DPI algorithm
│   ├── policy_eval.py      # Policy evaluation (explicit/implicit)
│   └── policy_improve.py   # Policy improvement via gradients
├── models/
│   ├── __init__.py
│   ├── base.py             # Base model class
│   ├── two_trees.py        # Two-trees asset pricing model
│   ├── lucas_orchard.py    # N-tree Lucas orchard model
│   └── hennessy_whited.py  # Corporate finance model
└── utils/
    ├── __init__.py
    ├── training.py         # Training loop utilities
    └── diagnostics.py      # Accuracy metrics, plotting
```

---

## Component Specifications

### 1. Hyper-Dual Ito's Lemma (`core/ito.py`)

**Purpose:** Efficiently compute the drift of a value function using the auxiliary function approach.

**Key Functions:**

```python
def compute_drift_hyperdual(
    V: Callable,           # Value function V(s) -> scalar
    s: Tensor,             # State batch (n_states, n_dims)
    drift: Tensor,         # f(s) drift vector (n_states, n_dims)
    diffusion: List[Tensor],  # [g_i(s)] diffusion columns, each (n_states, n_dims)
    method: str = "autodiff"  # "autodiff" or "finite_diff"
) -> Tensor:
    """Compute E[dV]/dt using hyper-dual approach."""
```

**Implementation Options:**

1. **Pure autodiff (PyTorch):**
   - Use `torch.autograd.functional.hessian` or nested `grad` calls
   - Cleaner but may have issues with mixed-mode AD

2. **Finite-difference for second derivative:**
   - Compute $F''(0) \approx \frac{F(h) - 2F(0) + F(-h)}{h^2}$
   - More robust when V itself uses autodiff
   - Higher-order stencils (5-point, 7-point, 9-point) for better accuracy

**Recommended:** Use finite-differences for the outer second derivative to avoid mixed-mode AD issues (as suggested in the paper for the implicit policy evaluation).

---

### 2. Neural Networks (`networks/`)

**Value Network (`value_network.py`):**

```python
class ValueNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = "silu",  # SiLU is C2, needed for Ito's lemma
        output_transform: Optional[Callable] = None,  # e.g., for boundary conditions
    ):
        ...
```

**Policy Network (`policy_network.py`):**

```python
class PolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = "relu",
        output_bounds: Optional[Tuple[float, float]] = None,  # Apply sigmoid scaling
    ):
        ...
```

**Activation Functions:**
- Value networks: SiLU (Swish), GELU, or Softplus (need C2 for Hessian)
- Policy networks: ReLU, LeakyReLU, or ELU

---

### 3. HJB Residual Computation (`core/hjb.py`)

**Purpose:** Compute HJB(s, c, V) for given states, controls, and value function.

```python
def compute_hjb_residual(
    model: BaseModel,      # Economic model with dynamics
    s: Tensor,             # States (batch_size, n_states)
    c: Tensor,             # Controls (batch_size, n_controls)
    V: Callable,           # Value function
    method: str = "hyperdual"
) -> Tensor:
    """
    HJB(s, c, V) = u(c) - ρV + E[dV]/dt

    where E[dV]/dt = ∇V·f(s,c) + (1/2)Tr[g'·H_V·g]
    """
```

---

### 4. DPI Algorithm (`algorithms/dpi.py`)

**Main Training Loop:**

```python
class DPITrainer:
    def __init__(
        self,
        model: BaseModel,
        value_network: ValueNetwork,
        policy_network: Optional[PolicyNetwork],
        config: DPIConfig,
    ):
        ...

    def train(self, n_iterations: int) -> TrainingHistory:
        """
        Main DPI loop:
        1. Sample mini-batch of states
        2. Policy improvement (gradient ascent on HJB w.r.t. policy params)
        3. Policy evaluation (minimize HJB residual w.r.t. value params)
        """
```

**Policy Evaluation Options:**

1. **Explicit (Eq. 17 in paper):**
   - Target: $V^{target} = V + HJB \cdot \Delta t$
   - MSE loss between V and target
   - Faster but can be unstable

2. **Implicit (Eq. 18 in paper):**
   - Directly minimize HJB residual squared
   - More stable but requires 3rd order derivatives
   - Use finite-diff for outer derivative to avoid mixed-mode AD

**Policy Improvement (Eq. 14):**
- Gradient ascent on HJB w.r.t. policy parameters
- Update: $\theta_C = \theta_C + \eta_C \nabla_{\theta_C} HJB$

---

### 5. Economic Models (`models/`)

**Base Model Interface:**

```python
class BaseModel(ABC):
    """Abstract base class for economic models."""

    @abstractmethod
    def drift(self, s: Tensor, c: Optional[Tensor] = None) -> Tensor:
        """Compute drift f(s, c)."""

    @abstractmethod
    def diffusion(self, s: Tensor, c: Optional[Tensor] = None) -> List[Tensor]:
        """Compute diffusion columns [g_1(s), ..., g_m(s)]."""

    @abstractmethod
    def utility(self, c: Tensor) -> Tensor:
        """Compute instantaneous utility u(c)."""

    @property
    @abstractmethod
    def discount_rate(self) -> float:
        """Return discount rate ρ."""

    @abstractmethod
    def sample_states(self, n: int) -> Tensor:
        """Sample states from state space."""
```

**Implemented Models:**

1. **TwoTrees** - Two-tree asset pricing (analytical solution available)
2. **LucasOrchard** - N-tree generalization (for high-dimensional testing)
3. **HennessyWhited** - Corporate finance with kinks (policy + value)

---

### 6. Sampling (`core/sampler.py`)

**Sampling Strategies:**

```python
class UniformSampler:
    """Uniform sampling within bounds."""

class DirichletSampler:
    """Sample from simplex (for dividend shares)."""

class MixtureSampler:
    """Mix interior and boundary sampling."""

class ErgodicSampler:
    """Sample from estimated ergodic distribution."""
```

---

### 7. Training Utilities (`utils/training.py`)

```python
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
    delta_t: float = 1.0  # Time step for explicit method
    n_eval_steps: int = 10
    n_improve_steps: int = 1

    # Ito computation
    ito_method: str = "finite_diff"
    fd_step_size: float = 5e-2
    fd_stencil: str = "nine"  # "three", "five", "seven", "nine"
```

---

## Implementation Order

### Phase 1: Core Infrastructure
1. `core/ito.py` - Hyper-dual Ito's lemma
2. `networks/value_network.py` - Basic value network
3. `networks/policy_network.py` - Basic policy network
4. `models/base.py` - Abstract model interface

### Phase 2: Simple Model (Two Trees)
5. `models/two_trees.py` - Two-trees model
6. `core/hjb.py` - HJB residual computation
7. `algorithms/policy_eval.py` - Policy evaluation
8. `algorithms/dpi.py` - Basic DPI trainer (asset pricing only)

### Phase 3: Full DPI with Controls
9. `algorithms/policy_improve.py` - Policy improvement
10. `models/hennessy_whited.py` - Corporate finance model
11. Update `algorithms/dpi.py` for problems with controls

### Phase 4: High-Dimensional
12. `models/lucas_orchard.py` - Lucas orchard
13. `core/sampler.py` - Advanced sampling strategies
14. `utils/diagnostics.py` - Accuracy metrics

### Phase 5: Polish & Extend
15. `utils/training.py` - Configuration, checkpointing
16. Comprehensive tests
17. Example notebooks

---

## Key Implementation Details

### Autodiff Considerations

**Challenge:** Computing $\nabla_{\theta_V} HJB$ requires differentiating through the Hessian of V (needed for drift), which is a 3rd-order derivative.

**Solution:** Use finite-differences for the 2nd derivative in the auxiliary function F, then use autodiff for the gradient w.r.t. network parameters:

```python
def hjb_with_fd_ito(V, s, model, h=0.05):
    """HJB using finite-diff for Ito drift."""
    def F(eps):
        s_perturbed = s + model.diffusion(s)[0] * eps / sqrt(2) + model.drift(s) * eps**2 / 2
        return V(s_perturbed)

    # Finite-diff 2nd derivative
    drift_term = (F(h) - 2*F(0) + F(-h)) / h**2
    return model.utility(c) - model.rho * V(s) + drift_term
```

This allows Zygote/PyTorch to differentiate `hjb_with_fd_ito` w.r.t. network parameters without mixed-mode issues.

### Boundary Conditions

For some models, the value function must satisfy boundary conditions (e.g., V(0) = 0). Use output transformations:

```python
# For V(k=0) = 0, multiply by k^α
v_raw = network(s)
v = s[:, 0:1] ** alpha * v_raw
```

### Multiple Brownian Shocks

When there are m shocks, sum over all columns of the diffusion matrix:

```python
def compute_drift(V, s, drift, diffusions, h=0.05):
    """diffusions is a list of m column vectors."""
    m = len(diffusions)

    def F(eps):
        total = 0.0
        for g_i in diffusions:
            s_pert = s + g_i * eps / sqrt(2) + drift * eps**2 / (2*m)
            total = total + V(s_pert)
        return total

    # Second derivative at 0
    return (F(h) - 2*F(0) + F(-h)) / h**2
```

---

## Testing Strategy

1. **Unit tests for Ito lemma:**
   - Test with known analytical functions (e.g., $V(s) = \sum s_i^2$)
   - Compare against explicit gradient/Hessian computation

2. **Two-trees model:**
   - Compare against analytical solution from Cochrane et al. (2008)
   - Track both absolute error and HJB residuals

3. **Corporate finance model:**
   - Compare against finite-difference benchmark
   - Verify kinks are captured correctly

4. **Scaling tests:**
   - Verify computational cost is independent of n_states
   - Verify linear scaling with n_shocks

---

## Dependencies

```
torch>=2.0
numpy
scipy (for Dirichlet sampling)
matplotlib (for diagnostics)
tqdm (for progress bars)
```

Optional:
```
jax (alternative autodiff backend)
```

---

## References

- Duarte, V., Duarte, D., & Silva, D. H. (2024). Machine Learning for Continuous-Time Finance. *The Review of Financial Studies*, 37(11), 3217-3271.
- Cochrane, J. H., Longstaff, F. A., & Santa-Clara, P. (2008). Two trees. *The Review of Financial Studies*, 21(1), 347-385.
- Martin, I. (2013). The Lucas orchard. *Econometrica*, 81(1), 55-111.
- Hennessy, C. A., & Whited, T. M. (2007). How costly is external financing? Evidence from a structural estimation. *Journal of Finance*, 62(4), 1705-1745.
