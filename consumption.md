# Consumption Smoothing with Multi-Factor Income Risk (Plan)

This document lays out an extensive, concrete plan for implementing the
consumption smoothing project using the existing DPI codebase and the
Python libraries already installed (torch, numpy, scipy, matplotlib, tqdm).
It does not implement the model; it describes what to build, where to
place it, and how to validate it.

## 0) Goals and success criteria

- Implement the multi-factor consumption smoothing model with K factors
  using DPI (value and policy networks).
- Validate against low-dimensional benchmarks (K=1 and K=2) and analytic
  limiting cases (log utility, deterministic income, transitory shocks).
- Scale to K=5, 10, 20 and demonstrate stable training, sensible
  economic behavior, and near-linear scaling in K.
- Produce diagnostics and plots that confirm economic intuition:
  precautionary savings, factor persistence effects, and correlation
  effects.

## 1) Environment and dependency setup

Libraries:
- torch: neural nets, autograd
- numpy: parameter handling, basic array ops
- scipy: sampling, correlated shocks (if needed for offline validation)
- matplotlib: plots
- tqdm: progress bars

Steps:
1. Create and activate a virtual environment.
2. Install the dependencies listed in `pyproject.toml` (or the minimal
   set above).
3. Confirm that the existing DPI tests pass before adding new files.

## 2) Project structure (new files to add)

Add a new model file and a small scripts folder for experiments:

- `dpi/models/consumption_smoothing.py`
  - Holds the model class for the consumption smoothing problem.
- `scripts/consumption_experiment.py`
  - Runs an end-to-end training experiment with configurable K.
- `tests/test_consumption_model.py`
  - Unit tests for drift, diffusion, and utility shapes.
- `tests/test_consumption_limits.py`
  - Check special cases (log utility, deterministic income).
- `notebooks/` (optional)
  - For exploratory plots and debugging (kept out of core code).

No changes are required to core DPI code unless new utility features
are identified during development.

## 3) Model specification in code

### 3.1 State, control, and dynamics

State reveals a vector with wealth and K factors:
- s = [w, x_1, ..., x_K]
- Control: c (consumption), scalar.

Wealth dynamics:
- dw = (r * w + Y(x) - c) dt
- no diffusion in w (only in x), unless you extend to risky assets.

Factor dynamics (OU):
- dx_k = -theta_k * x_k dt + sigma_k dZ_k

Correlation:
- allow correlated Brownian shocks using a Cholesky factor of
  the correlation matrix Sigma.
- diffusion for x is (diag(sigma) @ L), where L is Cholesky of
  Sigma, so the diffusion matrix has K columns (shocks).

### 3.2 Income function

- Y(x) = Ybar * exp(sum_k beta_k * x_k)
- implement stable handling of exponentials by clipping x or
  using torch.exp with safeguards if needed.

### 3.3 Utility

- CRRA: u(c) = c^(1-gamma) / (1-gamma)
- log utility is the gamma -> 1 case; treat explicitly for
  numerical stability.
- enforce c > 0 using policy network output transform or
  softplus parameterization.

### 3.4 Model class API

Implement a `ConsumptionSmoothing` model that matches `BaseModel`:

- `drift(s, c)` returns a Tensor (batch, K+1):
  - drift for w plus drifts for all x_k.
- `diffusion(s, c)` returns list of tensors, one per shock column:
  - each tensor shape (batch, K+1)
  - drift in w has zero diffusion column entries.
- `utility(s, c)` returns (batch, 1) Tensor.
- `discount_rate` property.
- `sample_states(n)` returns (n, K+1) Tensor.

Sampling:
- Use a mixture: uniform for w in [w_min, w_max] and normal
  for x_k in a reasonable band (e.g. +/- 3 std of stationary
  distribution).
- Consider an ergodic sampler once a policy exists.

## 4) Hyper-dual Ito evaluation

Use existing `compute_drift_hyperdual` in `dpi/core/ito.py`.

Key choices:
- start with finite-difference stencil "five" or "seven" with
  h around 1e-3 to 1e-2 for stable gradients.
- verify that changes in h do not change results materially.
- default to "finite_diff" to avoid mixed-mode AD.

## 5) Networks and parameterization

### 5.1 Value network

- Use `ValueNetwork` with smooth activation (SiLU or GELU).
- Add an output transform if needed to enforce boundary conditions
  (e.g., V(0, x) = 0). A simple transform is V = w^a * v_raw.

### 5.2 Policy network

- Use `PolicyNetwork` with output bounds.
- Ensure c >= c_min > 0 with either sigmoid bounds or softplus.
- Define bounds based on wealth and income scale (e.g.,
  c in [c_min, c_max], where c_max might be a multiple of Ybar).

### 5.3 Scaling and normalization

- Normalize inputs: scale wealth and factors to roughly unit range.
- Keep a simple linear rescaling in the model or in a wrapper
  to avoid distorting the economics.

## 6) DPI training loop

Use `DPITrainer` from `dpi/algorithms/dpi.py`:

- Policy improvement (actor): update c network by maximizing HJB.
- Policy evaluation (critic): use explicit or implicit update.

Recommended schedule:
1. Start with explicit evaluation and small delta_t.
2. Move to implicit once stable to reduce bias.
3. Use a small number of improvement steps (1) per iteration.

Hyperparameters to tune:
- batch size: 512 to 4096
- value_lr and policy_lr
- fd_step_size and stencil
- n_eval_steps and n_improve_steps

## 7) Diagnostics and validation

### 7.1 Internal checks

- HJB residual statistics on a held-out state sample:
  mean, std, max abs.
- Value loss and policy loss curves over training iterations.

### 7.2 Economic sanity checks

For small K, verify:
- consumption increases in wealth (monotone).
- for persistent shocks, c reacts more strongly than for
  transitory shocks.
- correlation increases precautionary savings (lower c).

### 7.3 Benchmarking

- K=1: compare against finite-difference solution in a grid
  (simple PDE solver in scipy or a pre-computed benchmark).
- K=2: compare slices of policy and value functions to
  low-dimensional numerical methods.
- scaling: report training time per iteration vs K.

## 8) Tests to add

### 8.1 Model shape tests

- `drift` output shape (batch, K+1).
- `diffusion` length equals K; each tensor shape (batch, K+1).
- `utility` shape (batch, 1) and correct handling of c.

### 8.2 Special-case tests

- Deterministic income (sigma_k=0): check that diffusion is zero
  and HJB reduces to ODE in w.
- Log utility (gamma=1): verify utility uses log and is finite
  for positive c.

### 8.3 Ito drift test

- Use a quadratic V(s) and compare hyper-dual drift to analytic
  drift for the chosen drift/diffusion.

## 9) Experiment plan (phases)

### Phase 1: Baseline replication

- Run existing two-trees example to confirm DPI environment works.
- Confirm tests and smoke training run end-to-end.

### Phase 2: K=1 factor

- Implement model with a single factor.
- Run DPI and compare to finite-difference benchmark.
- Validate shape and monotonicity.

### Phase 3: K=2 factors

- Uncorrelated factors first.
- Visualize policy slices: c(w, x1, x2) for fixed w.

### Phase 4: K=2 correlated

- Use positive correlation (e.g., 0.5).
- Verify stronger precautionary savings.

### Phase 5: Scaling

- Run K=5, 10, 20.
- Report timing and HJB residual stats.

### Phase 6: Economic analysis

- Compute consumption volatility and savings rates by
  factor persistence and correlation structure.
- Summarize in plots and short tables.

## 10) Plotting and reporting

Use matplotlib to generate:
- Loss curves
- HJB residual histograms
- Policy slices for K=1 and K=2
- Timing vs K

Keep a simple script that saves plots to `plots/` with
predictable filenames, to avoid clutter.

## 11) Risks and mitigations

- Numerical instability with large K: normalize inputs and
  reduce fd_step_size if HJB residuals explode.
- Slow training: reduce network size and batch size for
  initial debugging.
- Output constraints: use bounded policy outputs to keep
  c positive and avoid utility explosions.

## 12) Deliverables checklist

- Model class: `dpi/models/consumption_smoothing.py`
- Experiment script: `scripts/consumption_experiment.py`
- Tests: `tests/test_consumption_model.py`, `tests/test_consumption_limits.py`
- Plots and metrics in `plots/`
- Short summary report (optional) with calibration and key findings

This plan keeps the implementation aligned with the existing DPI
infrastructure, while providing enough structure to validate
the method in low dimensions and scale up to high-dimensional
factor risk.
