# Economic Analysis: Health Management with Interior Exercise and Diet (Convex Diet Cost)

This report summarizes the DPI outcomes after switching to a convex diet cost
and reducing its intensity. The goal was to obtain *interior* solutions for both
exercise and diet while preserving the state‑dependent, high‑dimensional policy
structure.

## Experiments run

- K=1, gamma=2.0, 800 iterations
- K=2, corr=0.3, gamma=2.0, 800 iterations
- K=8, corr=0.2, gamma=2.0, 300 iterations

Key outputs referenced:
- `plots/health_policy_1d_K1_gamma2.0.png`
- `plots/health_policy_age_K1_gamma2.0.png`
- `plots/health_policy_2d_K2_gamma2.0.png`
- `plots/health_policy_age_K8_gamma2.0.png`
- `plots/health_policy_2d_K8_gamma2.0.png`

## K=1: Single health dimension

Observed:
- Exercise is interior: ~0.11–0.15 across the health range.
- Diet is also interior: ~0.38–0.51, increasing with health.
- Leisure is ~0.30 and flat.
- Value is increasing in health.
- HJB residuals: mean -0.00025, std 0.153, max 0.794.

Economic interpretation:
- The convex diet cost creates a clear interior tradeoff. Diet is no longer
  pinned to 0 or 1 and responds to health state, with better health supporting
  slightly higher diet quality (a complementarity pattern).
- Exercise remains interior due to the direct utility benefit and strengthened
  health returns. The policy’s health sensitivity is modest but economically
  meaningful.

## K=2: Two health dimensions with correlation

Observed:
- Exercise varies smoothly with (h1, h2), roughly 0.096 to 0.18.
- The policy surface slopes upward in both dimensions.
- HJB residuals: mean 0.0014, std 0.267, max 1.151.

Economic interpretation:
- Exercise responds to both dimensions, indicating that cross‑effects in health
  are active. Higher health in either dimension raises the marginal return to
  exercise, consistent with complementarity between current health stock and
  exercise productivity.

## K=8: Full health system

Observed:
- Exercise is interior across the lifecycle (~0.16 rising to ~0.20 by age 80).
- Diet is interior and declines with age (~0.48 at 20 to ~0.37 by 80).
- Leisure remains stable just under 0.30.
- HJB residuals: mean -0.027, std 1.562, max 7.501.

Economic interpretation:
- The lifecycle profiles show a realistic composition shift: diet intensity
  declines with age while exercise rises to offset accelerating depreciation.
  This implies that dietary investment is more effective earlier in life, while
  exercise becomes relatively more valuable later.
- Exercise is state‑dependent even in K=8, confirming that the high‑dimensional
  policy is meaningful rather than collapsing to a constant rule.
- Residual dispersion is higher at K=8, so these results are best interpreted
  qualitatively. The directional patterns are economically coherent.

## Cross‑cutting economic takeaways

1) **Interior solutions on both margins are achieved.** Exercise and diet are
   neither at zero nor at their maximum, and both vary with health and age.

2) **Health investment composition shifts over the lifecycle.** Diet declines
   with age, exercise rises, and leisure remains stable—an economically plausible
   response to faster depreciation later in life.

3) **Complementarity is visible.** Better health states are associated with
   higher exercise and diet, suggesting that investments reinforce one another
   rather than substitute strongly.

4) **High‑dimensional structure matters.** Even at K=8, the policy remains
   sensitive to health states, justifying the need for DPI rather than low‑dim
   approximations.

## Summary

Switching to a convex diet cost and lowering its magnitude delivered the desired
outcome: both diet and exercise are interior and state‑dependent. The resulting
policies exhibit meaningful lifecycle patterns and cross‑health gradients,
providing a strong foundation for the next stage of analysis (welfare losses
from simple rules, shock responses, and health inequality experiments).

## Cost of Simple Rules (K=8)

We compared the optimal policy to two heuristics using forward simulation:

- **Constant rule:** e = 0.12, d = 0.35, l = 0.25 for all states and ages.
- **Age‑only rule:** use the learned policy at average health (h = 0) and the
  current age, applied to everyone.

Simulation setup:
- 256 paths, monthly time step (dt = 1/12), 40‑year horizon (age 30–70).
- Optimal policy trained for 200 iterations (implicit evaluation).

Results (consumption‑equivalent loss relative to optimal):
- **Constant rule:** V = -19.966 vs V_opt = -19.400 → **2.84% loss**
- **Age‑only rule:** V = -20.066 vs V_opt = -19.400 → **3.32% loss**

Economic interpretation:
- Simple heuristics capture much of the value, but ignoring health state still
  leaves measurable welfare on the table. The extra 3% consumption‑equivalent
  loss quantifies the value of full state‑contingent health management.
- The age‑only rule performs worse than the constant rule here, indicating that
  health‑state heterogeneity matters at least as much as lifecycle variation
  for optimal investment decisions under this calibration.
