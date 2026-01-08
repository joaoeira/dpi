

# Machine Learning for Computational Economics

## Module 05: The Deep Policy Iteration Method

EDHEC Business School

Dejanir Silva

Purdue University

January 2026

![Small house icon, likely a logo or decorative element.](0538daaa5583c23e17db3a12f2281a55_img.jpg)

Small house icon, likely a logo or decorative element.

### Introduction

In this module, we use machine-learning tools to solve high-dimensional problems in continuous time.

- We introduce the **Deep Policy Iteration (DPI)** method, proposed by Duarte, Duarte, and Silva (2024).
- It combines stochastic optimization, automatic differentiation, and neural-network function approximation.

We proceed in two steps.

1. We show how to overcome the three curses of dimensionality.
2. We describe a range of applications of the DPI method in *asset pricing*, *corporate finance*, and *portfolio choice*.

The module is organized as follows:

1. The Hyper-Dual Approach to Itô's lemma
2. The DPI Method
3. Applications

### I. Dealing with the Three Curses

![Small house icon](7ee2d12e8cbaacaf65b0c332d1c76daf_img.jpg)

Small house icon

#### The Dynamic Programming Problem

Consider a continuous-time optimal control problem in which an infinitely lived agent faces a *Markovian decision process (MDP)*.

- The system's state is represented by a vector  $\mathbf{s}_t \in \mathbb{R}^n$
- The control is a vector  $\mathbf{c}_t \in \Gamma(\mathbf{s}_t) \subset \mathbb{R}^p$ .

The agent's objective is to maximize the expected discounted value of the reward function  $u(\mathbf{c}_t)$ :

$$V(\mathbf{s}) = \max_{\{\mathbf{c}_t\}_{t=0}^{\infty}} \mathbb{E} \left[ \int_0^{\infty} e^{-\rho t} u(\mathbf{c}_t) dt \mid \mathbf{s}_0 = \mathbf{s} \right],$$

subject to the stochastic law of motion

$$d\mathbf{s}_t = \mathbf{f}(\mathbf{s}_t, \mathbf{c}_t) dt + \mathbf{g}(\mathbf{s}_t, \mathbf{c}_t) d\mathbf{B}_t.$$

where  $\mathbf{B}_t$  is an  $m \times 1$  vector of Brownian motions,  $\mathbf{f}(\mathbf{s}_t, \mathbf{c}_t) \in \mathbb{R}^n$  is the drift, and  $\mathbf{g}(\mathbf{s}_t, \mathbf{c}_t) \in \mathbb{R}^{n \times m}$  is the diffusion matrix.

! The HJB Equation.

![Small house icon.](4b9457ad2400572dbf0c0c9f7c825643_img.jpg)

Small house icon.

#### Dealing with the Three Curses

As discussed in Module 1, dynamic programming methods face three intertwined computational obstacles:

1<sup>st</sup> curse

Representing  $V(s)$

2<sup>nd</sup> curse

Solving for  $c$  given  $V(s)$

3<sup>rd</sup> curse

Computing  $E[V(s')]$

In this module, we present an algorithm that addresses each of these curses using modern machine-learning tools:

1. **Neural networks** provide compact representations of value and policy functions, circumventing the *curse of representation*;
2. **Stochastic optimization** replace costly root-finding procedures, alleviating the *curse of optimization*;
3. **Automatic differentiation** enables efficient computation of drift terms needed for the HJB, mitigating the *curse of expectation*;

### II. Overcoming the Curse of Expectation

![Small house icon, likely a page marker or logo.](afe5eb459b7c9cfe880b067777d876d8_img.jpg)

Small house icon, likely a page marker or logo.

#### From integration to differentiation

In discrete time, the Bellman equation involves the expectation of the continuation value:

$$\mathbb{E}[V(s')] = \int \cdots \int V(s + f(s, c)\Delta t + g(s, c)\sqrt{\Delta t}Z)\phi(Z)dZ_1 \cdots dZ_m,$$

where  $\phi(Z)$  is the joint density of the shocks.

In continuous time, the expected change in the value function can be written as

$$\mathbb{E}[dV(s)] = \nabla_s V(s) \cdot f(s, c) + \frac{1}{2} \text{Tr}(g(s, c)H_s V(s)g(s, c)),$$

Hence, instead of computing high-dimensional integrals, we only need to compute derivatives of the value function.

#### Computational challenge.

One could use finite differences to compute these derivatives

- But storing the solution on a grid becomes infeasible in higher dimensions.

Suppose  $n = 10$  and we use a grid of 100 points for each state variable.

![Small house icon.](b241be04490fd12c763d098c5213e7c2_img.jpg)

Small house icon.

#### Comparing the Computational Costs

To illustrate the computational challenge, we study a simple example.

Consider the following function:

$$V(\mathbf{s}) = \sum_{i=1}^{n} s_i^2,$$

where  $\mathbf{s} = (s_1, \dots, s_n)$  is a vector of state variables.

A high-dimensional problem with  $n = 100$  state variables

- We evaluate the drift at the point  $\mathbf{s} = \mathbf{1}_{n \times 1}$ .
- We focus on the case of a single shock ( $m = 1$ ).

We consider initially the following methods to compute the drift  $\mathbb{E}[dV(\mathbf{s})]$ :

1. Finite differences
2. Naive autodiff

![Lightbulb icon indicating a tip or note.](89986656b45c3b6896256f1a22f7c186_img.jpg)

Lightbulb icon indicating a tip or note.

Comparing the Computational Costs.

| Method                | FLOPs     | Memory      | Error |
|-----------------------|-----------|-------------|-------|
| 1. Finite differences | 9,190,800 | 112,442,048 | 1.58% |

![Small house icon.](a69d6cb43ee0ba9cdf803f6454db2f8e_img.jpg)

Small house icon.

#### Overcoming the Curse of Expectation

One of the key insights from Module 4 is that the **Jacobian-vector product** (JVP) can be efficiently computed.

- Much less expensive than forming the full Jacobian.

In the absence of shocks, computing the drift of  $V(\mathbf{s})$  amounts to evaluating a JVP:

$$\mathbb{E}[dV(\mathbf{s})] = \nabla_{\mathbf{s}} V(\mathbf{s})^{\top} \mathbf{f}(\mathbf{s}, \mathbf{c}) dt,$$

which can be efficiently computed using forward-mode AD.

In the presence of shocks, the drift also depends on quadratic forms involving the Hessian of  $V(\mathbf{s})$ .

- In this case, a JVP is no longer sufficient.

The **Hyper-dual approach** to Itô's lemma extends the idea of dual numbers to compute the drift of  $V(\mathbf{s})$ .

- Regular dual numbers only store the function value and its derivative.
- Hyper-dual numbers store the function value, its drift, and its diffusion matrix.

#### The Hyper-Dual Approach to Itô's lemma

The next result formalizes the hyper-dual approach to Itô's lemma.

- It reduces the problem of computing the drift of  $V(\mathbf{s})$  to evaluating the second derivative of a *univariate auxiliary function*.

##### Hyper-dual Itô's lemma

For a given  $\mathbf{s}$ , define the auxiliary functions  $F_i: \mathbb{R} \to \mathbb{R}$  as

$$F_i(\epsilon; \mathbf{s}) = V\left(\mathbf{s} + \frac{\mathbf{g}_i(\mathbf{s})}{\sqrt{2}} \epsilon + \frac{\mathbf{f}(\mathbf{s})}{2m} \epsilon^2\right),$$

where  $\mathbf{f}(\mathbf{s})$  is the drift of  $\mathbf{s}$  and  $\mathbf{g}_i(\mathbf{s})$  is the  $i$ -th column of the diffusion matrix  $\mathbf{g}(\mathbf{s})$ .

Then:

**Diffusion:** The  $1 \times m$  diffusion matrix of  $V(\mathbf{s})$  is

**Drift:** The drift of  $V(\mathbf{s})$  is

$$\nabla V(\mathbf{s})^\top \mathbf{g}(\mathbf{s}) = \sqrt{2} [F'_1(0; \mathbf{s}), F'_2(0; \mathbf{s}), \dots, F'_m(0; \mathbf{s})].$$

$$\square V(\mathbf{s}) = F''(0; \mathbf{s}),$$

where  $F(\epsilon; \mathbf{s}) = \sum_{i=1}^{m} F_i(\epsilon; \mathbf{s})$

##### Julia Implementation

It is straightforward to implement the hyper-dual approach to Itô's lemma in Julia.

```
1 using ForwardDiff, LinearAlgebra
2 V(s) = sum(s.^2) # example function
3 n, m = 100, 1 # number of state variables and shocks
4 s0, f, g = ones(n), ones(n), ones(n,m) # example values
5
6 # Analytical drift
7 ∇f, H = 2*s0, Matrix(2.0*I, n,n) # gradient and Hessian
8 drift_analytical = ∇f'*f + 0.5*tr(g'*H*g) # analytical drift
9
10 # Hyper-dual approach
11 F(ϵ) = sum([V(s0 + g[:,i]*ϵ/sqrt(2) + f/(2m)*ϵ^2) for i = 1:m])
12 drift_hyperdual = ForwardDiff.derivative(ϵ -> ForwardDiff.derivative(F, ϵ), 0.0) # scalar 2nd derivative
```

To verify correctness, we can compare the analytical and hyper-dual drifts:

```
1 # Compare
2 drift_analytical, drift_hyperdual
(300.0, 300.0)
```

#### Computational Cost

This figure shows how the cost of computing the drift of a function  $V$  scales with the number of state variables and Brownian shocks.

- The cost is independent of the number of state variables.
- It increases *linearly* – instead of exponentially – with the number of Brownian shocks.

![](42ff8b598a0818ca8b6ef30850ad5f4e_img.jpg)

One Brownian Shock

Computational Cost of  $\frac{E dV}{dt}$  (Y-axis) vs. Number of State Variables (X-axis).

Legend: Actual Cost (solid line), Theoretical Lower Bound (dashed line).

The plot shows that the computational cost is relatively stable and close to the theoretical lower bound (around 1.0) as the number of state variables increases from 1 to 100.

![](252ea48d02dce93965b91746fb376f35_img.jpg)

100 State Variables

Computational Cost of  $\frac{E dV}{dt}$  (Y-axis) vs. Number of Shocks (X-axis).

The plot shows that the computational cost increases linearly with the number of shocks, ranging from 1 to 100 shocks.

Notes: Cost is measured as the execution time of  $\frac{E dV}{dt}(s)$  relative to that of  $V(s)$ . The left panel fixes  $m = 1$  and varies  $n$  from 1 to 100; the right panel fixes  $n = 100$  and varies  $m$  from 1 to 100. In this example,  $V$  is a two-layer neural network, and execution times are averaged over 10,000 runs on a mini-batch of 512 states.

#### Computational Cost (continued)

We benchmark the performance of the hyper-dual approach to Itô's lemma by comparing the execution time of alternative methods.

- In addition to finite differences and naive autodiff, we also consider the analytical expression for the derivatives

The **hyper-dual Itô's lemma** method is faster and less memory-intensive than using the analytical expression for the derivatives.

| Method                | FLOPs     | Memory      | Error |
|-----------------------|-----------|-------------|-------|
| 1. Finite differences | 9,190,800 | 112,442,048 | 1.58% |
| 2. Naive autodiff     | 2,100,501 | 25,673,640  | 0.00% |
| 3. Analytical         | 20,501    | 44,428      | 0.00% |
| 4. Hyper-dual Itô     | 599       | 6,044       | 0.00% |

### III. The Deep Policy Iteration Algorithm

#### Overcoming the Curse of Representation

Our objective is to compute the value function  $V(s)$  and policy function  $c(s)$  satisfying the coupled functional equations:

$$0 = \text{HJB}(s, c(s), V(s)), \quad c(s) = \arg \max_{c \in \Gamma(s)} \text{HJB}(s, c, V(s)),$$

where

$$\text{HJB}(s, c, V) = u(c) - \rho V + \nabla_s V(s) \cdot f(s, c) + \frac{1}{2} \text{Tr}(g(s, c) H_s V(s) g(s, c)),$$

and  $f(s, c)$  is the drift of  $s$  and  $g(s, c)$  is the diffusion matrix.

To solve for  $V(s)$  and  $c(s)$  numerically, we must represent them on a computer.

- A traditional approach is to discretize the state space and interpolate between grid points.
- In Module 4, we showed that such grid-based approximations can be viewed as shallow neural networks with fixed breakpoints.

**Neural networks** generalize this idea

#### Overcoming the Curse of Optimization

We now turn to the challenge of training the DNNs to satisfy the functional equations above.

- A key difficulty lies in performing the maximization step efficiently, without resorting to costly root-finding procedures.

Our approach builds on **generalized policy iteration** (see, e.g., Sutton and Barto (2018))

- Combining with deep function approximation, alternating between policy evaluation and policy improvement.
- This leads to the **Deep Policy Iteration (DPI)** algorithm.

The algorithm proceeds in three stages:

1. Sampling
2. Policy improvement
3. Policy evaluation

![Lightbulb icon representing a tip or simplifying assumption.](0656422bf374a8a7bcc6fe99adc48599_img.jpg)

Lightbulb icon representing a tip or simplifying assumption.

Simplifying assumptions.

For clarity, we make several simplifying assumptions that can be relaxed in practice.

![Small house icon.](4a104be5f84f688417d8c222ec4ce4fa_img.jpg)

Small house icon.

##### Step 1: Sampling

We begin by sampling a mini-batch of states  $\{s_i\}_{i=1}^I$  from the state space.

- This can be done using a uniform distribution within plausible state-space bounds.
- Alternatively, we can use an estimated ergodic distribution based on previous iterations.

##### Step 2: Policy Improvement

The *policy improvement* step involves solving the maximization step for each state in the mini-batch.

- This step can be computationally demanding and lies at the heart of the *curse of optimization*.

##### ⓘ Generalized policy iteration.

In Module 3, we introduced the policy function iteration, which alternates between *policy evaluation* and *policy improvement* steps.

- In the policy evaluation step, we solve for the new value function  $V_{n+1}(s)$  given the policy  $c_n(s)$ .
- In the policy improvement step, we solve for the new policy  $c_{n+1}(s)$  given the current value function  $V_{n+1}(s)$ .

However, when the initial guess for  $V$  is far from optimal, fully solving the maximization problem at each iteration is inefficient.

- This motivates an *approximate policy improvement* step that performs only a single gradient-based update in the direction of improvement.

Fix the current parameter vectors for the value and policy functions,  $\theta_V^{j-1}$  and  $\theta_C^{j-1}$ .

##### Step 2: Policy Improvement (continued)

Our goal is to choose the policy function to maximize (or minimize minus) the value of the HJB operator

- We can then define the loss function as follows:

$$\square_p(\theta_C) = -\frac{1}{2I} \sum_{i=1}^{I} \text{HJB}(s_i, \theta_C^{j-1}, \theta_V^{j-1}),$$

$$\text{where } \text{HJB}(s_i, \theta_C^{j-1}, \theta_V^{j-1}) \equiv \text{HJB}(s_i, c_{j-1,i}, V_{j-1,i}) .$$

We perform *one step* of gradient descent on the loss function to update the policy function parameters:

! Policy improvement step.

$$\theta_C^j = \theta_C^{j-1} + \eta_C \frac{1}{I} \sum_{i=1}^{I} \nabla_{\theta_C} \text{HJB}(s_i, \theta_C^{j-1}, \theta_V^{j-1}),$$

where  $\eta_C$  is the learning rate controlling the step size in parameter space.

##### Step 3: Policy Evaluation I

We now update the value function given the new policy parameters,  $\theta_C^j$ .

- We present two alternative update rules, each with distinct trade-offs.

The first rule mirrors the **explicit method** for finite-differences in Module 3.

- Consider a *false-transient* formulation that iterates the value function backward in (pseudo-)time:

$$\frac{V(\mathbf{s}; \theta_V^j) - V(\mathbf{s}_i; \theta_V^{j-1})}{\Delta t} = \text{HJB}(\mathbf{s}_i, \theta_C^j, \theta_V^{j-1}) \Rightarrow V(\mathbf{s}; \theta_V^j) = V(\mathbf{s}_i; \theta_V^{j-1}) + \text{HJB}(\mathbf{s}_i, \theta_C^j, \theta_V^{j-1})\Delta t,$$

target value  $V_i^{j-1}$

where the HJB is evaluated at the new policy parameters,  $\theta_C^j$ , but *old* value function parameters,  $\theta_V^{j-1}$ .

This turns the problem into a supervised learning task given the training data  $\{\mathbf{s}_i, V_i^{j-1}\}_{i=1}^I$ .

##### Step 3: Policy Evaluation I (continued)

Define the loss function as the mean-squared error between the target value and the value function:

$$\square(\theta_V) = \frac{1}{2I} \sum_{i=1}^{I} (V(\mathbf{s}_i; \theta_V) - V_i^{j-1})^2,$$

Evaluating the gradient at  $\theta_V^{j-1}$  yields

$$\nabla_{\theta_V} \square(\theta_V^{j-1}) = -\frac{\Delta t}{I} \sum_{i=1}^{I} \text{HJB}(\mathbf{s}_i, \theta_C^j, \theta_V^{j-1}) \nabla_{\theta_V} V(\mathbf{s}_i; \theta_V),$$

Taking one step of gradient descent, the corresponding update rule is:

! Policy evaluation step.

$$\theta_V^j = \theta_V^{j-1} + \eta_V \frac{\Delta t}{I} \sum_{i=1}^{I} \text{HJB}(\mathbf{s}_i, \theta_C^j, \theta_V^{j-1}) \nabla_{\theta_V} V(\mathbf{s}_i; \theta_V^{j-1}),$$

where  $\eta_V$  is the learning rate.

##### Step 3: Policy Evaluation II

The second rule mirrors the **implicit method** for finite-differences in Module 3:

$$\frac{V(s; \theta_V^j) - V(s_i; \theta_V^{j-1})}{\Delta t} = \text{HJB}(s_i, \theta_C^j, \theta_V^j),$$

As in the implicit finite-difference method, we can take the limit as  $\Delta t \to \infty$ .

In this case, the loss function becomes MSE of HJB residuals:

$$\square(\theta_V) = \frac{1}{2I} \sum_{i=1}^{I} \text{HJB}(s_i, \theta_C^j, \theta_V)^2,$$

The gradient is given by

$$\nabla_{\theta_V} \square(\theta_V) = \frac{1}{I} \sum_{i=1}^{I} \text{HJB}(s_i, \theta_C^j, \theta_V) \nabla_{\theta_V} \text{HJB}(s_i, \theta_C^j, \theta_V),$$

Evaluating at  $\theta_V^{j-1}$  yields

! Policy evaluation step 2.

#### The Deep Policy Iteration Algorithm

We can now summarize the complete algorithm:

**Algorithm:** Deep Policy Iteration (DPI)

**Input:** Initial parameters  $\theta_V^0$  and  $\theta_C^0$

**Output:** Value function  $V(s; \theta_V)$ , policy function  $c(s; \theta_C)$

**Initialize:**  $j \leftarrow 0$

**Repeat for**  $j = 1, 2, \dots$  :

1. **Sampling:**

Sample a mini-batch of states  $\{s_i\}_{i=1}^I$ .

2. **Policy improvement (actor update):**

Update  $\theta_C$  with one step of gradient descent on the loss function.

3. **Policy evaluation (critic update):**

Update  $\theta_V$  using the explicit or implicit policy evaluation step.

![Lightbulb icon indicating a tip or important note.](99f676c3b1fcc3176dec6a5cdccbb8ec_img.jpg)

Lightbulb icon indicating a tip or important note.

The trade-off between the two update rules.

![Small house icon.](890eb6a150d26785299fabeb41f9d34b_img.jpg)

Small house icon.

### IV. Applications: Asset Pricing

#### Asset Pricing: The Two-Trees Model

Next, we apply the DPI algorithm to solve a variety of economic and financial problems.

- We focus on three canonical domains of finance: asset pricing, corporate finance, and portfolio choice.
- We illustrate how the DPI algorithm can be applied to solve a variety of economic and financial problems.

To start, we consider the familiar **two-trees model** from Module 3.

The pricing condition for a log investor implies

$$v_t = \mathbb{E}_t \left[ \int_0^\infty e^{-\rho s} s_{t+s} ds \right],$$

where the relative share process  $s_t$  evolves as

$$ds_t = -2\sigma^2 s_t (1 - s_t) \left(s_t - \frac{1}{2}\right) dt + \sigma s_t (1 - s_t) (dB_{1,t} - dB_{2,t}) 1/\rho.$$

The price-consumption ratio  $v_t$  satisfies the HJB equation:

$$\rho v = s - v_s 2\sigma^2 s(1-s)\left(s - \frac{1}{2}\right) + \frac{1}{2} v_{ss} \left(2\sigma^2 s^2(1-s)^2\right)$$

with boundary conditions  $v(0) = 0$  and

It is straightforward to solve this one-dimensional problem using finite differences or collocation methods

![Small house icon indicating a footnote or reference.](5fffa62bedbc76c885147fa8a31b9c1e_img.jpg)

Small house icon indicating a footnote or reference.

##### Julia Implementation

We now implement the two-trees model in Julia, starting with the *model struct*.

```
1 @kwdef struct TwoTrees
2     p::Float64 = 0.04
3     σ::Float64 = sqrt(0.04)
4     μ::Float64 = 0.02
5     μs::Function = s -> @. -2 * σ^2 * s * (1-s) * (s-0.5) # drift of s
6     σs::Function = s -> @. sqrt(2) * σ * s * (1-s) # diffusion of s
7 end;
8
9 m = TwoTrees()
```

TwoTrees(0.04, 0.2, 0.02, var"#6#10"{Float64}(0.2), var"#7#11"{Float64}(0.2))

We next implement the **hyper-dual** approach to Ito's lemma to compute the drift and diffusion of the state variable  $s$ .

```
1 using ForwardDiff
2 function drift_hyper(V::Function, s::AbstractMatrix, m::TwoTrees)
3     F(ε) = V(s + m.σs(s)/sqrt(2)*ε + m.μs(s)/2*ε^2)
4     return ForwardDiff.derivative(ε -> ForwardDiff.derivative(F, ε), 0.0)
5 end;
```

To validate the implementation, we can compare the analytical and hyper-dual drifts:

```
1 using Random
2 rng = Xoshiro(0)
3 s = rand(rng, 1, 1000)
4 # Exact drift for test function
```

![Small house icon](eaa5fbc353eb95b90302cfbe7c299576_img.jpg)

Small house icon

##### Neural Network Implementation

We now implement the neural network representation of the value function.

```
1 using Lux
2 model = Chain(
3     Dense(1 => 25, Lux.gelu),
4     Dense(25 => 25, Lux.gelu),
5     Dense(20 => 1)
6 )
```

```
Chain(
    layer_1 = Dense(1 => 25, gelu_tanh),      # 50 parameters
    layer_2 = Dense(25 => 25, gelu_tanh),      # 650 parameters
    layer_3 = Dense(20 => 1),                  # 21 parameters
)
# Total: 721 parameters,
# plus 0 states.
```

We initialize the parameters and optimizer state using the Adam optimizer with a learning rate of  $10^{-3}$ .

```
1 using Optimisers
2 rng = Xoshiro(0)
3 ps, ls = Lux.setup(rng, model) |> f64
4 opt = Adam(1e-3)
5 os = Optimisers.setup(opt, ps)
```

```
(layer_1 = (weight = Leaf(Adam(eta=0.001, beta=(0.9, 0.999), epsilon=1.0e-8), ([0.0; 0.0; ... ; 0.0; 0.0;],
```

We define the loss function as the mean-squared error between the value function and the target value.

![Small house icon](1695df64fe320e3f81049cfe402c8155_img.jpg)

Small house icon

##### Training the Neural Network

We now train the neural network using the Adam optimizer.

```
1 # Training loop
2 loss_history = Float64[]
3 for i = 1:40_000
4     s_batch = rand(rng, 1, 128)
5     tgt = target(s-> model(s, ps, ls)[1], s_batch, m, Δt = 1.0)
6     loss = loss_fn(ps, ls, s_batch, tgt)
7     grad = gradient(p -> loss_fn(p, ls, s_batch, tgt), ps)[1]
8     os, ps = Optimisers.update(os,ps, grad)
9     push!(loss_history, loss)
10 end
```

#### The Lucas Orchard Model

We now extend the two-trees model to a multi-tree economy, known as the **Lucas orchard model** (see, e.g., Martin (2013)).

- By varying the number of trees, we can examine how the DPI algorithm scales with the dimensionality of the state space.

Consider a representative investor with log utility who can invest in a riskless asset and  $N$  risky assets.

- Each risky asset  $i$  pays a continuous dividend stream  $D_{i,t}$  that follows a geometric Brownian motion:

$$\frac{dD_{i,t}}{D_{i,t}} = \mu_i dt + \sigma_i dB_{i,t},$$

where each  $B_{i,t}$  is a Brownian motion satisfying  $dB_{i,t} dB_{j,t} = 0$  for  $i \neq j$ .

! The HJB equation.

Define the vector of state variables  $\mathbf{s}_t = (s_{1,t}, \dots, s_{N,t})^\top$ , where  $s_{i,t} \equiv D_{i,t}/C_t$  and  $C_t = \sum_{i=1}^N D_{i,t}$ .

![Small house icon.](b800561cd10527de6f3f41b23b562990_img.jpg)

Small house icon.

##### Julia Implementation

We now implement the Lucas orchard model in Julia.

- The workflow of the DPI algorithm for the Lucas orchard model is virtually identical to that for the simple two-trees model.

As usual, we start by defining the *model struct*.

```
1 @kwdef struct LucasOrchard
2     p::Float64 = 0.04
3     N::Int = 10
4     σ::Vector{Float64} = sqrt(0.04) * ones(N)
5     μ::Vector{Float64} = 0.02 * ones(N)
6     μc::Function = s -> μ' * s
7     oc::Function = s -> [s[i,:]'] * σ[i] for i in 1:N
8     μs::Function = s -> s .* (μ .- μc(s) .- s.*σ.^2 .+ sum(oc(s)[i].^2 for i in 1:N))
9     σs::Function = s -> [s .* ([j == i ? σ[i] : 0 for j in 1:N] .- oc(s)[i]) for i in 1:N]
10 end;
```

```
1 # Instantiate the model
2 using Distributions, Random
3 m = LucasOrchard(N = 10) # number of assets
4 rng, d = MersenneTwister(0), Dirichlet(ones(m.N)) # Dirichlet distribution
5 s_samples = rand(rng, d, 1_000) # N x 1_000 matrix
6 m.μs(s_samples) # N x 1_000 matrix
```

```
10×1000 Matrix{Float64}:
 4.24309e-5   0.000237275   0.000815818   ...   8.57681e-5   -0.00169363
-0.000259516  0.000134325   0.000132953   ...   0.000550971  0.000463926
```

![Small house icon](c80dd550f724de455f5efebaed25198d_img.jpg)

Small house icon

#### The Hyper-dual Approach to Ito's Lemma

We next implement the hyper-dual approach to Ito's lemma to compute the drift and diffusion of the state variable  $s$ .

```
1 using ForwardDiff
2 function drift_hyper(V::Function, s::AbstractMatrix, m::LucasOrchard)
3     N, σs, μs = m.N, m.σs(s), m.μs(s) # Preallocations
4     F(ε) = sum(V(s .+ σs[i] .* (ε / sqrt(2)) .+ μs .* (ε^2 / (2 * N))) for i in 1:N)
5     return ForwardDiff.derivative(ε -> ForwardDiff.derivative(F, ε), 0.0)
6 end;
```

The implementation is virtually identical to that for the two-trees model.

- But now we are dealing with the case of multiple state variables and Brownian motions.

![Lightbulb icon indicating a tip or important note.](11728b408ca6402f502858c9bc161c4a_img.jpg)

Lightbulb icon indicating a tip or important note.

##### The importance of preallocations.

Relative to the two-trees model, we now preallocate arrays for the drift and diffusion of the state variable  $s$ ,

- Instead of constructing them inside loops over  $i = 1, \dots, N$ .
- In higher-dimensional problems, preallocations avoid repeated memory allocation and garbage collection, improving performance.

##### Neural Network Implementation

We next implement the neural network representation of the value function.

- We use essentially the same architecture as in the two-trees model.
- But now the input is the  $N$ -dimensional vector of state variables  $s_t$ .

```
1 model = Chain(  
2     Dense(m.N => 25, Lux.gelu),  
3     Dense(25 => 25, Lux.gelu),  
4     Dense(25 => 1)  
5 )
```

```
Chain(  
    layer_1 = Dense(10 => 25, gelu_tanh),      # 275 parameters  
    layer_2 = Dense(25 => 25, gelu_tanh),      # 650 parameters  
    layer_3 = Dense(25 => 1),                  # 26 parameters  
)  
# Total: 951 parameters,  
# plus 0 states.
```

We initialize the parameters and optimizer state using the Adam optimizer with a learning rate of  $10^{-3}$ .

```
1 ps, ls = Lux.setup(rng, model) |> f64  
2 opt = Adam(1e-3)  
3 os = Optimisers.setup(opt, ps)
```

```
(layer_1 = (weight = Leaf(Adam(eta=0.001, beta=(0.9, 0.999), epsilon=1.0e-8), ([0.0 0.0 ... 0.0 0.0; 0.0 0.0
```

We define the loss function and the target function as before.

![Small house icon](e6fbffa8f0a33d829216b3e99c9e1103_img.jpg)

Small house icon

##### Training the Neural Network

```
1 # Training parameters
2 max_iter, Δt = 40_000, 1.0
3 # Sampling interior and boundary states
4 d_int = Dirichlet(ones(m.N)) # Interior region
5 d_edge = Dirichlet(0.05 .* ones(m.N)) # Boundary region
6 # Loss history and exponential moving average loss
7 loss_history, loss_ema_history, α_ema = Float64[], Float64[]
8 # Training loop
9 p = Progress(max_iter; desc="Training...", dt=1.0) #progress bar
10 for i = 1:max_iter
11     if rand(rng) < 0.50
12         s_batch = rand(rng, d_int, 128)
13     else
14         s_batch = rand(rng, d_edge, 128)
15     end
16     v(s) = model(s, ps, ls)[1] # define value function
17     tgt, hjb_res = target(v, s_batch, m, Δt = Δt) #target
18     loss, back = Zygote.pullback(p -> loss_fn(p, ls, s_batch))
19     grad = first(back(1.0)) # gradient
20     os, ps = Optimisers.update(os, ps, grad) # update parameters
21     loss_ema = i==1 ? loss : α_ema*loss_ema + (1.0-α_ema)*loss
22     push!(loss_history, loss)
23     push!(loss_ema_history, loss_ema)
24     next!(p, showvalues = [(i, i) ("Loss", loss)])
```

We sample from two Dirichlet distributions:

- Interior region: uniform distribution over the simplex.
- Boundary region: highly concentrated near the edges.

![A line graph showing training loss over iterations. The x-axis is 'Iteration (in thousands)' ranging from 0 to 40. The y-axis is 'Loss' on a logarithmic scale, ranging from 10^-4 to 10^-2. Two lines are plotted: 'Loss' (blue, fluctuating) and 'Loss EMA' (red, smoother). Both lines show a rapid decrease in loss initially, stabilizing around 10^-4 after approximately 20 thousand iterations.](5dc5581cd2aad0e683c73b959f637b31_img.jpg)

A line graph showing training loss over iterations. The x-axis is 'Iteration (in thousands)' ranging from 0 to 40. The y-axis is 'Loss' on a logarithmic scale, ranging from 10^-4 to 10^-2. Two lines are plotted: 'Loss' (blue, fluctuating) and 'Loss EMA' (red, smoother). Both lines show a rapid decrease in loss initially, stabilizing around 10^-4 after approximately 20 thousand iterations.

Training loss.

##### Test set evaluation I

We next evaluate the model's performance on *out-of-sample* test sets.

![A plot showing the function v(s) versus s_1. The x-axis is s_1 (0.00 to 1.00) and the y-axis is v(s) (0 to 25). Two lines, 'Prediction' (blue solid line) and 'Exact' (orange dashed line), are plotted and overlap perfectly, indicating the model's prediction matches the analytical solution.](fae82236e4211f753df5789eb276d3a4_img.jpg)

A plot showing the function v(s) versus s\_1. The x-axis is s\_1 (0.00 to 1.00) and the y-axis is v(s) (0 to 25). Two lines, 'Prediction' (blue solid line) and 'Exact' (orange dashed line), are plotted and overlap perfectly, indicating the model's prediction matches the analytical solution.

Two-trees prediction.

We consider an extremely asymmetric configuration:

- $\mathbf{s} = (s_1, 1 - s_1, 0, \dots, 0)$ , the two-trees special case.
- This configuration lies outside the region used for training.

The network's prediction replicates the analytical solution.

- Even though was not trained on this configuration,

##### Test set evaluation II

Our second test set draws states from a symmetric Dirichlet distribution with parameters

$$\alpha = \alpha_{\text{scale}}(1, 1, \dots, 1)$$

- $\alpha_{\text{scale}}$  controls the concentration of points within the simplex.
- Samples are concentrated near the center (or edges) of the simplex when  $\alpha_{\text{scale}} > 1$  (or  $< 1$ ).

![Small house icon/logo](fd0bd97f68ab12fb22f937db56c801a5_img.jpg)

Small house icon/logo

![](152efae989544ee653283e8de26cc9b1_img.jpg)

Four plots showing Dirichlet densities (pdf) in the simplex space defined by  $x_1, x_2, x_3$  (where  $x_1 + x_2 + x_3 = 1$ ).

- Top Left:  $\alpha = [1.2, 1.2, 1.2]$ . The density is highest near the center of the simplex.
- Top Right:  $\alpha = [0.8, 0.8, 0.8]$ . The density is highest near the vertices of the simplex.
- Bottom Left:  $\alpha = [4.0, 1.0, 1.0]$ . The density is highest near the vertex corresponding to  $x_1$ .
- Bottom Right:  $\alpha = [1.0, 4.0, 1.0]$ . The density is highest near the vertex corresponding to  $x_2$ .

Dirichlet densities.

![](fe655d77258397f7242c2df72b965b56_img.jpg)

Two bar charts showing HJB residual (MSE) on a logarithmic scale.

- Top Chart: Residuals versus  $\alpha_{scale}$  (ranging from 0.1 to 1.5). The residuals are relatively constant, around  $10^{-5}$ .
- Bottom Chart: Residuals versus tree number (1 to 10). The residual for tree 1 is significantly higher (around  $10^{-4}$ ), while residuals for trees 2 through 10 are lower (around  $10^{-5}$ ).

Orchard residuals.

#### Comparison with other methods

We next compare the DPI algorithm with other methods for solving high-dimensional dynamic models.

- We use the Lucas orchard model as a benchmark.

Finite-difference schemes become computationally infeasible beyond a few dimensions

- Chebyshev collocation on full tensor-product grids also suffers from exponential growth in cost.

We then compare the time to solution of the DPI algorithm with the **Smolyak method** (Smolyak (1963)).

- The Smolyak method is a sparse-grid technique for approximating multivariate functions
- It is commonly used for solving high-dimensional dynamic models.

##### Time to solution

![](6ee57fd30c7e609827c2a11d0983eeba_img.jpg)

Minutes of Training for accuracy  $< 10^{-8}$

Minutes

Number of Trees

Mean squared errors

90th percentile (squared errors)

Notes: Figure shows the time-to-solution of the DPI algorithm, measured by the number of minutes required for the MSE or 90th-percentile squared error to fall below  $10^{-8}$ . The parameter values are  $\rho = 0.04$ ,  $\gamma = 1$ ,  $\varrho = 0.0$ ,  $\mu = 0.015$ , and  $\sigma = 0.1$ . The HJB residuals are computed on a random sample of  $2^{13}$  points from the state space.

![Small house icon.](0baabed74e5fce9eaf1cac18837415d8_img.jpg)

Small house icon.

#### Smolyak vs. DPI algorithm

![](27b09ea51378a0f896d21b3ebad0b22f_img.jpg)

Minutes of Training for Accuracy  $< 10^{-3}$

Minutes

Number of Trees

DPI

Smolyak<sub>2</sub>

Smolyak<sub>3</sub>

Smolyak<sub>4</sub>

Notes: Figure compares the time-to-solution of the DPI method and the Smolyak methods of orders 2, 3, and 4. The tolerance is set to  $10^{-3}$ , the highest accuracy threshold reached by all Smolyak variants. The parameter values are  $\rho = 0.04$ ,  $\gamma = 1$ ,  $\xi = 0.0$ ,  $\mu = 0.015$ , and  $\sigma = 0.1$ . The HJB residuals are computed on a random sample of  $2^{13}$  points from the state space.

![Small house icon.](4bc95c6b4eac6323f8a6498457333a75_img.jpg)

Small house icon.

### V. Applications: Corporate Finance

#### The Hennessy and Whited (2007) Model

We now apply the DPI algorithm to a corporate finance problem, a simplified version of the model in Hennessy and Whited (2007).

- This problem illustrates how the DPI algorithm can handle problems with **kinks** and **inaction regions**.

Consider a firm with operating profits

$$\pi(k_t, z_t) = e^{z_t} k_t^\alpha.$$

- Log productivity follows an Ornstein-Uhlenbeck process:

$$dz_t = -\theta(z_t - \bar{z}) dt + \sigma dB_t,$$

where  $\theta, \sigma > 0$ .

- Given investment rate  $i_t$  and depreciation rate  $\delta$ , capital evolves as

$$dk_t = (i_t - \delta) k_t dt.$$

The firm faces linear equity issuance costs  $\lambda > 0$ .

- Operating profits net of adjustment costs are

$$D^*(k_t, z_t, i_t) = e^{z_t} k_t^\alpha - (i_t + 0.5\chi i_t^2) k_t,$$

where  $\chi > 0$  is the adjustment cost parameter.

- The firm's dividend policy is given by

$$D_t = D_t^*(1 + \lambda 1_{D_t^* < 0}).$$

#### The HJB Equation

The HJB equation is given by

$$0 = \max_{i} \text{HJB}(s, i, v(s)),$$

where

$$\text{HJB}(s, i, v) = D(k, z, i) + \nabla v^{\top} \mu_s(s, i) + \frac{1}{2} \sigma_s(s, i)^{\top} \mathbf{H}_s v \sigma_s(s, i) - \rho v.$$

The first-order condition for the optimal investment rate is

$$\frac{\partial \text{HJB}}{\partial i} = -\left(1 + \lambda \mathbf{1}_{D^*(k,z,i)<0}\right) \left[1 + \chi i\right] k + v_k(s)k = 0.$$

Consider a *special case* where investment is fixed at  $i = \delta$  and productivity is constant  $\theta = \sigma = 0$ .

- In this case, capital is constant
- From the HJB equation, we obtain the value function

$$v(k, z) = \frac{D(k, z, \delta)}{\rho} = \begin{cases} \frac{e^{z k^{\alpha} - \delta k}}{\rho}, & \text{if } k \le k_{\max}(z), \\ \frac{e^{z k^{\alpha} - \delta k}}{\rho} (1 + \lambda), & \text{if } k > k_{\max}(z), \end{cases}$$

![Small house icon, likely a page marker or logo.](ab488a6d7d5801f36752e3906ad1b3b5_img.jpg)

Small house icon, likely a page marker or logo.

where  $k_{\max}(Z) = \left(\frac{e^z}{\delta}\right)^{\frac{1}{1-\alpha}}$ .

##### Model struct and loss function

We start by defining the model struct.

```
1 @kwdef struct HennessyWhited
2     α::Float64 = 0.55
3     θ::Float64 = 0.26
4     z̄::Float64 = 0.0
5     σz::Float64 = 0.123
6     δ::Float64 = 0.1
7     χ::Float64 = 0.1
8     λ::Float64 = 0.059
9     ρ::Float64 = -log(0.96)
10    μs::Function = (s,i) -> vcat((i .- δ) .* s[1,:]', -θ .* (s[2,:] .- z̄)')
11    σs::Function = (s,i) -> vcat(zeros(1,size(s,2)), σz*ones(1,size(s,2)))
12 end;
```

The HJB is particularly simple in this case.

```
1 function hjb_special_case(m, s, θv)
2     (; α, λ, ρ, δ) = m
3     k, z = s[1,:]', s[2,:]'
4     D_star = exp.(z) .* k.^α - δ * k
5     D = D_star .* (1 .+ λ * (D_star .< 0))
6     hjb = D - ρ * v_net(s, θv)
7     return hjb
8 end
```

##### Network architecture and training

We use a simple feedforward network with two hidden layers.

```
1 v_core = Chain(  
2     Dense(2, 32, Lux.swish),  
3     Dense(32, 24, Lux.swish),  
4     Dense(24, 1)  
5 )
```

To enforce the boundary condition, we multiply the network by a function of  $k$  that vanishes at zero.

- A convenient choice is  $g(k) = k^\alpha$ .

```
1 v_net(s,  $\theta_v$ ) = (s[1, :].^m. $\alpha$ )' .* v_core(s,  $\theta_v$ , stv)[1]
```

We initialize the parameters, optimizer, and training parameters.

```
1 # Initialize the parameters and optimizer  
2 rng = Xoshiro(1234)  
3  $\theta_v$ , stv = Lux.setup(rng, v_core) |> Lux.f64  
4 optv = Optimisers.Adam(1e-3)  
5 osv = Optimisers.setup(optv,  $\theta_v$ )  
6  
7 # Training parameters  
8 max_iter = 300_000
```

![Small house icon](f254a67565344d514e13763a4e556a70_img.jpg)

Small house icon

##### Training history and value function for special case

```
1 loss_history_special_case = Float64[];
2 for it in 1:max_iter
3     s_batch = vcat(rand(rng, dk, 150)', rand(rng, dz, 150)')
4     loss_v, back_v = Zygote.pullback(p -> loss_v_special_case(m, s_batch, p), θ_v)
5     grad_v = first(back_v(1.0))
6     os_v, θ_v = Optimisers.update(os_v, θ_v, grad_v)
7     push!(loss_history_special_case, loss_v)
8     if loss_v < 1e-6
9         println("Iteration ", it, "| Loss_v = ", loss_v)
10         break
11     end
12 end
```

![](7fef73f27d4372a53355cc9bf8ac2703_img.jpg)

Plot showing the training history (Loss vs. Iteration). The loss decreases rapidly initially and then stabilizes at a low level (around  $10^{-5}$  to  $10^{-6}$ ) after approximately 200 thousand iterations.

Y-axis: Loss (log scale,  $10^{-6}$  to  $10^{-2}$ ). X-axis: Iteration (in thousands, 0 to 250).

![](068b3a3247570c4b78342a943f15de9e_img.jpg)

Plot showing the value function  $V(k, z)$  versus  $k$  for different values of  $z$ . The curves show that the value function is maximized when  $z = \bar{z} + 0.10$  and minimized when  $z = \bar{z} - 0.10$ .

Y-axis:  $V(k, z)$  (ranging from -20 to 5). X-axis:  $k$  (ranging from 0.0 to 10.0).

Legend:

- DNN:  $z = \bar{z}$
- Exact:  $z = \bar{z}$
- DNN:  $z = \bar{z} - 0.10$
- Exact:  $z = \bar{z} - 0.10$
- DNN:  $z = \bar{z} + 0.10$
- Exact:  $z = \bar{z} + 0.10$

#### A Q-theory of investment

Consider the case where there are no equity issuance costs, i.e.,  $\lambda = 0$ .

- In this case, the investment rate is given by the q-theory of investment:  $i(k, z) = \frac{v_k(k, z) - 1}{\chi}$ .
- We will solve for the optimal investment without using the first-order condition explicitly.

In addition to a neural network for the value function, we will also use a neural network for the optimal investment policy.

```
1 i_core = Chain(  
2     Dense(2, 32, Lux.swish),  
3     Dense(32, 24, Lux.swish),  
4     Dense(24, 1)  
5 )  
6 i_net(s,  $\theta_i$ ) = i_core(s,  $\theta_i$ , st_i)[1]  
7  
8 # Initialize the parameters and optimizer  
9 rng = Xoshiro(1234)  
10  $\theta_i$ , st_i = Lux.setup(rng, i_core) |> Lux.f64  
11 opt_i = Optimisers.Adam(1e-3)  
12 os_i = Optimisers.setup(opt_i,  $\theta_i$ )
```

##### Implementing the implicit version of policy evaluation

We will use the implicit version of policy evaluation:

$$\theta_V^j = \theta_V^{j-1} - \eta_V \frac{1}{I} \sum_{i=1}^{I} HJB(s_i, \theta_C^j, \theta_V^{j-1}) \nabla_{\theta_V} HJB(s_i, \theta_C^j, \theta_V^{j-1}),$$

The term  $\nabla_{\theta_V} HJB(s_i, \theta_C^j, \theta_V^{j-1})$  depends on the gradient and Hessian of the value function.

- Hence, this requires computing a **third-order derivative** of the value function.
- This type of **mixed-mode** automatic differentiation is not supported by many AD systems.

To overcome this challenge, we will use the hyper-dual approach to Itô's lemma.

- This requires computing a second derivative of a univariate function.
  - With automatic differentiation, we would face the mixed-mode limitation.
- We will compute this second derivative using **finite-differences**.
  - This way, Zygote never interacts with dual numbers, and can differentiate the HJB residuals without difficulty.

##### The HJB residuals

First, we define a function that computes the second derivative of the value function using finite-differences.

- We consider the usual three-point stencil, but also include higher-order stencils as options.

```
1 function second_derivative_FD(F::Function, h::Float64; stencil::Symbol = :three)
2     if stencil == :nine
3         return (-9.0 .* F(-4h) .+ 128.0 .* F(-3h) .- 1008.0 .* F(-2h) .+ 8064.0 .* F(-h) .- 14350.0 .
4     elseif stencil == :seven
5         return (2.0 .* F(-3h) .- 27.0 .* F(-2h) .+ 270.0 .* F(-h) .- 490.0 .* F(0.0) .+ 270.0 .* F(h)
6     elseif stencil == :five
7         return (-F(2h) .+ 16.0 .* F(h) .- 30.0 .* F(0.0) .+ 16.0 .* F(-h) .- F(-2h)) ./ (12.0 * h^2)
8     else
9         return (F(h) - 2.0 * F(0.0) + F(-h)) / (h*h) # Three point stencil
10 end
11 end
```

Given this function, we can define the HJB residual.

```
1 function hjb_residual(m, s,  $\theta_v$ ,  $\theta_i$ ; h = 5e-2, stencil::Symbol = :three)
2     (;  $\alpha$ ,  $\lambda$ ,  $\rho$ ,  $\delta$ ,  $\chi$ ) = m
3     k, z = s[1,:]', s[2,:]'
4     i = i_net(s,  $\theta_i$ )
5     D_star = exp.(z) .* k.^ $\alpha$  - (i + 0.5* $\chi$ *i.^2).*k
6     D = D_star .* (1 .+  $\lambda$  * (D_star .< 0))
7      $\mu_k$  = (i .-  $\delta$ ) .* k
8      $\mu_z$  = - $\theta$  .* (z .-  $\bar{z}$ )
9      $\mu_s$  = vcat( $\mu_k$ ,  $\mu_z$ )
```

![Small house icon](1675c9ac5116bd7a283fe5bdbf53f969_img.jpg)

Small house icon

##### Loss functions and training parameters

First, we define the loss functions for the value function

- In this formulation, we minimize directly the mean squared HJB residuals.

```
1 function loss_v(m, s,  $\theta_v$ ,  $\theta_i$ ; h = 5e-2, stencil::Symbol = :nine)2     hjb = hjb_residual(m, s,  $\theta_v$ ,  $\theta_i$ ; h = h, stencil = stencil)3     return mean(abs2, hjb)4 end
```

Next, we define the loss function for the optimal investment policy.

- The loss corresponds to minus the mean HJB residuals, as we want to maximize the right-hand side of the HJB equation.

```
1 function loss_i(m, s,  $\theta_v$ ,  $\theta_i$ ; h = 5e-2, stencil::Symbol = :nine)2     hjb = hjb_residual(m, s,  $\theta_v$ ,  $\theta_i$ ; h = h, stencil = stencil)3     return -mean(hjb)4 end
```

Finally, we define the training parameters.

```
1 max_iter = 300_0002 kmin, kmax = 1.0, 8.0
```

![Small house icon](ff487673f3a22a7dac3ae2dc59d2fc51_img.jpg)

Small house icon

##### Training loop

We can now define the training loop.

```
1 loss_history_v = Float64[];
2 loss_history_i = Float64[];
3 nsteps_v, nsteps_i = 10, 1
4 for it in 1:max_iter
5     s_batch = vcat(rand(rng, dk, 150)', rand(rng, dz, 150)')
6     loss_v, loss_i = zero(Float64), zero(Float64)
7     # Policy evaluation step
8     for _ = 1:nsteps_v
9         loss_v, back_v = Zygote.pullback(p -> loss_v(m, s_batch, p, θ_i), θ_v)
10        grad_v = first(back_v(1.0))
11        os_v, θ_v = Optimisers.update(os_v, θ_v, grad_v)
12    end
13    # Policy improvement step
14    for _ = 1:nsteps_i
15        loss_i, back_i = Zygote.pullback(p -> loss_i(m, s_batch, θ_v, p), θ_i)
16        grad_i = first(back_i(1.0))
17        os_i, θ_i = Optimisers.update(os_i, θ_i, grad_i)
18    end
19    # Compute loss
20    loss_v = loss_v(m, s_batch, θ_v, θ_i)
21    loss_i = loss_i(m, s_batch, θ_v, θ_i)
22    push!(loss_history_v, loss_v)
23    push!(loss_history_i, loss_i)
24    if mod(it, 100) == 0
25        println("Iteration $it: loss_v = $loss_v, loss_i = $loss_i")
26    end
27 end
```

##### Training history and value function

![](a387e0c81bfc615ececcd1b55dbf5de4_img.jpg)

Loss history plot showing Loss (log scale) versus Iteration (in thousands). The plot displays two loss components: Loss :  $v$  (blue line) and Loss :  $i$  (orange line). Both losses start high (around  $10^5$ ) and decrease rapidly, stabilizing around  $10^0$  after approximately 1 thousand iterations. The losses continue to fluctuate and gradually decrease further, reaching values around  $10^{-5}$  after 3 thousand iterations.

Loss history

![](fd8369b549b3d1a5c848cbd83659cae9_img.jpg)

Value function plot showing  $i(k, z)$  versus  $k$ . The plot compares the DNN approximation ( $i(k, z)$ , blue solid line) and the Exact solution ( $(v_k(k, z) - 1)/x$ , orange dashed line). Both functions start around 0.3 at  $k=0$  and decrease monotonically as  $k$  increases, approaching zero as  $k$  approaches 8. The DNN approximation closely matches the exact solution.

Value function

#### Optimal Dividend Policy and Investment Rate

We consider next the optimal dividend policy and investment rate for the full model with  $\lambda > 0$ .

- We compare the results with the finite-difference solution.

![](6752cee124f693bc4cebc66180f4f91f_img.jpg)

Graph showing Dividends (Y-axis, ranging from -1.25 to 0.75) versus Capital (X-axis, ranging from 5 to 40). Three curves are plotted for different values of  $z$ :  $z=0.87$  (dotted blue line),  $z=1.00$  (solid orange line), and  $z=1.15$  (dashed green line). The curves show that dividends are negative for low capital levels (below approximately 10) and increase towards positive values as capital increases, with the dividend level increasing as  $z$  increases.

Dividends

![](8e8ee6d2f8a17f2da03923bc97e226d7_img.jpg)

Graph showing Investment Rate (Y-axis, ranging from -0.75 to 1.50) versus Capital (X-axis, ranging from 5 to 40). Three curves are plotted for different values of  $z$ :  $z=0.87$  (dotted blue line),  $z=1.00$  (solid orange line), and  $z=1.15$  (dashed green line). The curves show that the investment rate is high for low capital levels (above 1.00) and decreases as capital increases, with the investment rate decreasing as  $z$  increases.

Investment Rate

##### Global sensitivity analysis

This problem has only two state variables, so it can be easily solved using finite differences

- But we are often interested in the solution for a large number of parameter values
- For instance, how does the solution vary with investment or equity issuance costs?

In particular, we are interested in knowing which equilibrium moments are more sensitive to parameters

- This way we know which aspects of the data are more informative about the parameters
- To answer these questions, we need to perform a **global sensitivity analysis**

Performing such global sensitivity analysis can be computationally very costly

- Using the DPI method, we can overcome this challenge
- **Solution:** add the vector of parameters to the state space
- In deep-reinforcement learning, this is known as **universal value functions**

##### Global sensitivity analysis results

![](42827b610e5711ab5fedfa3262c5cc37_img.jpg)

This figure displays the results of a global sensitivity analysis, showing 20 plots arranged in a 5x4 grid. The plots illustrate the sensitivity of five key metrics (Equity Issuance, Investment Rate, Profitability, Prof. Autocorr., and STD(Residual)) to four input parameters ( $\lambda$ ,  $\delta$ ,  $\alpha$ , and  $\sigma$ ).

The metrics analyzed are:

- Equity Issuance
- Investment Rate
- Profitability
- Prof. Autocorr.
- STD(Residual)

The input parameters analyzed are:

- $\lambda$
- $\delta$
- $\alpha$
- $\sigma$

The plots show the relationship between the input parameter (x-axis) and the corresponding metric (y-axis). The x-axis ranges vary depending on the parameter, while the y-axis ranges are generally consistent across the metrics.

Summary of observed trends:

- **Equity Issuance:** Shows a strong positive relationship with  $\delta$  and  $\sigma$ , and a slight positive relationship with  $\lambda$ . It is relatively insensitive to  $\alpha$ .
- **Investment Rate:** Shows a strong positive relationship with  $\delta$  and  $\sigma$ . It is relatively insensitive to  $\lambda$  and  $\alpha$ .
- **Profitability:** Shows a strong negative relationship with  $\alpha$ . It is relatively insensitive to  $\lambda$ ,  $\delta$ , and  $\sigma$ .
- **Prof. Autocorr.:** Shows a strong negative relationship with  $\delta$  and  $\sigma$ . It is relatively insensitive to  $\lambda$  and  $\alpha$ .
- **STD(Residual):** Shows a strong positive relationship with  $\delta$  and  $\sigma$ . It is relatively insensitive to  $\lambda$  and  $\alpha$ .

### VI. Applications: Portfolio Choice

#### Portfolio Choice with Realistic Dynamics

As our third application, we consider a **portfolio choice** problem with realistic dynamics

- The problem will feature a large number of state variables, shocks, and controls

We consider the problem of an investor with Epstein-Zin preferences

- Investor must choose consumption and portfolio
- Investor has access to 5 risky assets and a risk-free asset

Volatility is constant and expected returns are affine functions of the state  $\mathbf{x}_t \in \mathbb{R}^n$

- The state variable follows a multivariate Ornstein-Uhlenbeck (O-U) process:

$$d\mathbf{x}_t = -\Phi\mathbf{x}_t + \sigma_{\mathbf{x}}d\mathbf{B}_t,$$

To ensure absence of arbitrage, expected returns are derived from a **state-price density** (SPD)

#### State variables

##### List of State Variables Driving the Expected Returns of Assets

| Variable        | Description                                  | Mean   | S.D. (%) |
|-----------------|----------------------------------------------|--------|----------|
| $\pi_t$         | Log Inflation                                | 0.032  | 2.3      |
| $y_t^{\$(1)}$   | Log 1-Year Nominal Yield                     | 0.043  | 3.1      |
| $yspr_t^{\$}$   | Log 5-Year Minus 1-Year Nominal Yield Spread | 0.006  | 0.7      |
| $\Delta z_t$    | Log Real GDP Growth                          | 0.030  | 2.4      |
| $\Delta d_t$    | Log Stock Dividend-to-GDP Growth             | -0.002 | 6.3      |
| $d_t$           | Log Stock Dividend-to-GDP Level              | -0.270 | 30.5     |
| $pd_t$          | Log Stock Price-to-Dividend Ratio            | 3.537  | 42.6     |
| $\Delta \tau_t$ | Log Tax Revenue-to-GDP Growth                | 0.000  | 5        |

![Small house icon](337807384666e63ebce2c59d5b60978a_img.jpg)

Small house icon

| Variable     | Description                  | Mean   | S.D. (%) |
|--------------|------------------------------|--------|----------|
| $\tau_t$     | Log Tax Revenue-to-GDP Level | -1.739 | 6.5      |
| $\Delta g_t$ | Log Spending-to-GDP Growth   | 0.006  | 7.6      |
| $g_t$        | Log Spending-to-GDP Level    | -1.749 | 12.9     |

Notes: The table shows the list of 11 state variables driving expected returns in our economy, along with their mean and standard deviation. The data are collected from <https://www.publicdebtvaluation.com/data>. See Jiang et al. (2024) for more details.

##### Estimation of continuous-time SPD

##### State dynamics estimation

- We run a discrete time VAR on the  $n = 11$  state variables:

$$\mathbf{x}_t = \Psi \mathbf{x}_{t-1} + \mathbf{u}_t.$$

- Find  $\Phi$  and  $\sigma_x$  such that the time-integrated continuous-time process coincides with VAR

##### State-price density estimation

- Find  $(r_0, \mathbf{r}_1, \eta_0, \eta_1)$  to minimize squared deviations of model-implied and data moments
- Moments: time-integrated time series for nominal yields, real yields, and expected stock returns

#### Bond Yields and Equity Expected Returns

![](9996a51651209af4c8adad41ffe45393_img.jpg)

This figure displays six time series plots comparing historical data (dashed line) and a model fit (solid line) for various bond yields and equity returns from 1947-01-01 to 2019-01-01.

The plots are arranged in two rows of three:

- **Top Row (Nominal Yields):** 1-yr Nominal Yield, 5-yr Nominal Yield, and 10-yr Nominal Yield. The Y-axis is Rate (%).
- **Bottom Row (Real Yields and Equity Returns):** Equity Return, 5-yr Real Yield, and 10-yr Real Yield. The Y-axis is Rate (%).

All plots show significant volatility, particularly during the 1980s and 2008-2009 period.

##### Time Series of Expected Returns and Optimal Allocations

![](78fcfedad60b6dd255b4cc3c69d02c62_img.jpg)

Line chart showing the Time Series of Expected Returns (%) from 1949 to 2019. The Y-axis ranges from -10% to 40%. The X-axis shows the Year. The chart displays the expected returns for six assets: Risk-free, Stock, Medium Real Bond, Medium Nominal Bond, Long Real Bond, and Long Nominal Bond. The Stock returns (dashed orange line) show high volatility, peaking around 40% in the late 1940s and again around 2009. The Risk-free rate (solid blue line) generally stays below 5%. The bond returns (dotted lines) are generally lower and less volatile than stocks.

Expected Returns

![](e0113695dbf148bf5ec34354e544414b_img.jpg)

Stacked area chart showing the Time Series of Optimal Asset Allocation Weights (%) from 1950 to 2020. The Y-axis represents Portfolio Weights (%), ranging from 0 to 100. The X-axis shows the Year. The chart displays the optimal weights for six assets: Risk-free rate, Stock, Medium Real Bond, Medium Nominal Bond, Long Real Bond, and Long Nominal Bond. The allocation is highly dynamic, showing significant shifts in weight across assets over time. Stocks (orange) and Long Nominal Bonds (brown) are prominent in many periods, while Medium Real Bonds (green) and Long Real Bonds (purple) show periods of high allocation, particularly around 1970 and 2010.

Asset Allocation

#### Optimal allocation

![](55086c71bc8ee4c0abc7160d26a8092c_img.jpg)

Graph showing the Consumption-Wealth Ratio (%) versus State (ranging from -1.00 to 1.00). The Y-axis ranges from 5.00 to 6.75. The legend identifies variables:  $\pi$ ,  $y_t^s(1)$ ,  $y_{sprt}^s$ ,  $\Delta z$ ,  $\Delta d$ ,  $d$ ,  $p_d$ ,  $\Delta \tau$ ,  $\tau$ ,  $\Delta g$ , and  $g$ . The curves generally intersect near the State = 0.00 line.

Consumption-Wealth Ratio

![](7b96fce298a23fd76a01ff6c176c1059_img.jpg)

Graph showing Stock Allocation (%) versus State (ranging from -1.00 to 1.00). The Y-axis ranges from 20 to 100. The curves show a general trend of decreasing allocation for negative states and increasing allocation for positive states, with most curves intersecting near the State = 0.00 line.

Stock

#### Optimal portfolio (continued)

![](230c84ff14c9f8fa832d2e29c63b7aff_img.jpg)

Graph showing Nominal Long-Term Bond Allocation (%) versus State. The allocation is zero for negative states and increases sharply for positive states, with different curves representing different optimal portfolios.

Nominal Long-Term Bond

![](c742b64169a38a7f3f990172019878c8_img.jpg)

Graph showing Real Long-Term Bond Allocation (%) versus State. Multiple curves show complex, non-linear relationships, generally peaking around State 0.50 and decreasing towards State 1.00.

Real Long-Term Bond

### Conclusion

In this lecture, we learned methods that alleviate the *three curses of dimensionality*

1. Use deep neural networks to represent  $V(s)$
2. Compute expectations using Itô's lemma and automatic differentiation
3. Gradient-based update rule that does not rely on root-finding procedures

The **DPI algorithm** allows us to solve high-dimensional problems

- Method is effective in situations where leading numerical methods fail

Ability to solve rich problems can be an invaluable tool in economic analysis

- We oftentimes make assumptions that have no clear economic interest
  - Assumptions are made only for *tractability* reasons
- Our method enables researchers to focus on economically interesting models
  - Instead of focusing on models that we could easily solve

### References

Duarte, Victor, Diogo Duarte, and Dejanir H. Silva. 2024. "Machine Learning for Continuous-Time Finance." *The Review of Financial Studies* 37 (11): 3217–71. <https://doi.org/10.1093/rfs/hhae043>.

Hennessy, Christopher A, and Toni M Whited. 2007. "How Costly Is External Financing? Evidence from a Structural Estimation." *Journal of Finance* 62 (4): 1705–45.

Jiang, Zhengyang, Hanno Lustig, Stijn Van Nieuwerburgh, and Mindy Z. Xiaolan. 2024. "The u.s. Public Debt Valuation Puzzle." *Econometrica* 92 (4): 1309–47. <https://doi.org/10.3982/ECTA17674>.

Martin, Ian. 2013. "The Lucas Orchard." *Econometrica* 81 (1): 55–111. <https://doi.org/10.3982/ECTA8446>.

Smolyak, Sergei Abramovich. 1963. "Quadrature and Interpolation Formulas for Tensor Products of Certain Classes of Functions." In *Doklady Akademii Nauk*, 148:1042–45. 5. Russian Academy of Sciences.

Sutton, Richard S., and Andrew G. Barto. 2018. *Reinforcement Learning: An Introduction*. 2nd ed. Cambridge, MA: MIT Press.