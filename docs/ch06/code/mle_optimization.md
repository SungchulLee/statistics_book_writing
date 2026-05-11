# Maximum Likelihood Optimization Examples

## Overview

Maximum Likelihood Estimation often requires numerical optimization when closed-form solutions are unavailable. This page demonstrates the key optimization techniques used in practice -- grid search, gradient-based methods, and the Expectation-Maximization algorithm -- along with diagnostics for verifying convergence and the effect of starting values.

## The Optimization Problem

Given observed data $x_1, \ldots, x_n$ from a parametric family $f(x; \theta)$, the MLE solves:

$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta) = \arg\max_\theta \sum_{i=1}^n \log f(x_i; \theta)
$$

Equivalently, we minimize the negative log-likelihood:

$$
\hat{\theta}_{\text{MLE}} = \arg\min_\theta \left[-\ell(\theta)\right]
$$

!!! warning "Practical Considerations"

    - The log-likelihood may have multiple local maxima (e.g., mixture models).
    - Constraints on the parameter space (e.g., $\sigma^2 > 0$) require reparameterization or constrained optimization.
    - Poor starting values can cause convergence to a local optimum or numerical failure.

## Grid Search

The simplest optimization strategy evaluates $\ell(\theta)$ over a grid of candidate values. This is feasible for one or two parameters and provides a useful visualization of the likelihood surface.

```python
import numpy as np
from scipy import stats

def grid_search_normal_mean(data, mu_grid):
    """Evaluate the log-likelihood of Normal(mu, sigma^2) over a grid of mu values."""
    sigma_hat = data.std(ddof=0)
    log_liks = np.array([
        np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma_hat))
        for mu in mu_grid
    ])
    best_idx = np.argmax(log_liks)
    return mu_grid[best_idx], log_liks

# Example
rng = np.random.default_rng(42)
data = rng.normal(loc=5.0, scale=2.0, size=100)
mu_grid = np.linspace(3.0, 7.0, 500)
mu_hat, log_liks = grid_search_normal_mean(data, mu_grid)
print(f"Grid search MLE: mu_hat = {mu_hat:.4f}")
print(f"Closed-form MLE: mu_hat = {data.mean():.4f}")
```

## Gradient-Based Optimization

For multi-parameter problems, gradient-based methods are essential. The **score function** (gradient of the log-likelihood) is:

$$
S(\theta) = \frac{\partial}{\partial\theta}\ell(\theta)
$$

At the MLE, the score equals zero: $S(\hat{\theta}) = 0$.

### Reparameterization for Unconstrained Optimization

When parameters are constrained (e.g., $\sigma^2 > 0$), a common trick is to optimize over a transformed parameter:

$$
\phi = \log(\sigma^2) \quad \Rightarrow \quad \sigma^2 = e^\phi
$$

This transforms the constrained problem into an unconstrained one.

```python
import numpy as np
from scipy import optimize

def mle_normal_numerical(data):
    """Find Normal MLE via numerical optimization with reparameterization."""
    def neg_log_lik(params):
        mu, log_sigma2 = params
        sigma2 = np.exp(log_sigma2)
        n = len(data)
        return 0.5 * n * np.log(2 * np.pi * sigma2) + np.sum((data - mu) ** 2) / (2 * sigma2)

    # Try multiple starting points
    best_result = None
    for mu0 in [0, data.mean(), data.median()]:
        for ls0 in [0, np.log(data.var())]:
            result = optimize.minimize(neg_log_lik, x0=[mu0, ls0], method="Nelder-Mead")
            if best_result is None or result.fun < best_result.fun:
                best_result = result

    mu_hat = best_result.x[0]
    sigma2_hat = np.exp(best_result.x[1])
    return mu_hat, sigma2_hat

rng = np.random.default_rng(42)
data = rng.normal(5.0, 2.0, 100)
mu_hat, sigma2_hat = mle_normal_numerical(data)
print(f"Numerical MLE: mu = {mu_hat:.4f}, sigma^2 = {sigma2_hat:.4f}")
print(f"Closed-form:   mu = {data.mean():.4f}, sigma^2 = {np.mean((data - data.mean())**2):.4f}")
```

## Newton-Raphson Method

The Newton-Raphson method uses second-order information (the Hessian) for faster convergence:

$$
\theta^{(t+1)} = \theta^{(t)} - \left[\ell''(\theta^{(t)})\right]^{-1} \ell'(\theta^{(t)})
$$

In the multivariate case, this becomes:

$$
\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \mathbf{H}^{-1}(\boldsymbol{\theta}^{(t)})\, \nabla\ell(\boldsymbol{\theta}^{(t)})
$$

where $\mathbf{H}$ is the Hessian matrix of the log-likelihood.

!!! info "Fisher Scoring"
    Replacing the observed Hessian with its expectation $-I(\theta)$ (the negative Fisher information matrix) gives the **Fisher scoring** algorithm. Near the MLE, Fisher scoring and Newton-Raphson behave similarly.

## Sensitivity to Starting Values

For non-convex likelihoods (e.g., mixture models), the optimization result can depend on the starting point.

```python
import numpy as np
from scipy import optimize, stats

def mixture_log_likelihood(params, data):
    """Negative log-likelihood for a two-component Gaussian mixture."""
    pi, mu1, mu2, sigma = params[0], params[1], params[2], np.exp(params[3])
    pi = 1 / (1 + np.exp(-pi))  # sigmoid transform for mixing weight
    ll = np.sum(np.log(
        pi * stats.norm.pdf(data, mu1, sigma) +
        (1 - pi) * stats.norm.pdf(data, mu2, sigma)
    ))
    return -ll

# Generate mixture data
rng = np.random.default_rng(42)
n = 200
z = rng.binomial(1, 0.4, n)
data = np.where(z, rng.normal(0, 1, n), rng.normal(4, 1, n))

# Try different starting values
starts = [[0, -1, 5, 0], [0, 2, 2, 0], [0, 0, 3, 0.5]]
for i, x0 in enumerate(starts):
    result = optimize.minimize(mixture_log_likelihood, x0, args=(data,), method="Nelder-Mead")
    pi_hat = 1 / (1 + np.exp(-result.x[0]))
    print(f"Start {i+1}: pi={pi_hat:.3f}, mu1={result.x[1]:.3f}, "
          f"mu2={result.x[2]:.3f}, nll={result.fun:.2f}")
```

## Interpretation

- **Grid search** is reliable for low-dimensional problems and provides direct visualization of the likelihood surface.
- **Gradient-based methods** (Nelder-Mead, BFGS, Newton-Raphson) scale to high dimensions but may converge to local optima.
- **Reparameterization** converts constrained optimization into unconstrained optimization, improving numerical stability.
- **Multiple restarts** with different starting values help diagnose multimodality.

## Exercises

**Exercise 1.** For $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Exp}(\lambda)$, write the negative log-likelihood and find the MLE analytically. Verify your answer by implementing a grid search over $\lambda \in [0.1, 5]$.

??? success "Solution to Exercise 1"
    The log-likelihood is:

    $$
    \ell(\lambda) = n\log\lambda - \lambda\sum_{i=1}^n x_i
    $$

    Setting $\ell'(\lambda) = n/\lambda - \sum x_i = 0$ gives $\hat{\lambda}_{\text{MLE}} = n/\sum x_i = 1/\bar{X}$.

    ```python
    import numpy as np

    rng = np.random.default_rng(42)
    data = rng.exponential(scale=2.0, size=50)  # true lambda = 0.5
    lam_grid = np.linspace(0.1, 5, 1000)
    ll = len(data) * np.log(lam_grid) - lam_grid * data.sum()
    lam_hat_grid = lam_grid[np.argmax(ll)]
    lam_hat_exact = 1 / data.mean()
    print(f"Grid MLE:    {lam_hat_grid:.4f}")
    print(f"Analytic MLE: {lam_hat_exact:.4f}")
    ```

    Both values should agree closely. $\square$

---

**Exercise 2.** Explain why optimizing $\phi = \log(\sigma^2)$ instead of $\sigma^2$ directly is preferable in numerical MLE. What property of the transformation ensures the optimizer never evaluates at $\sigma^2 \leq 0$?

??? success "Solution to Exercise 2"
    The exponential function $\sigma^2 = e^\phi$ maps $\phi \in \mathbb{R}$ to $\sigma^2 \in (0, \infty)$. Since $e^\phi > 0$ for all real $\phi$, the optimizer is free to search over all of $\mathbb{R}$ without ever producing an invalid (non-positive) variance. Without this reparameterization, gradient steps could push $\sigma^2$ below zero, causing the log-likelihood to be undefined (since $\log(\sigma^2)$ appears in the normal log-likelihood). The transformation also improves the optimization landscape by making the curvature more uniform. $\square$

---

**Exercise 3.** Derive the Newton-Raphson update for estimating $p$ in $\text{Binomial}(n, p)$ given a single observation $x$. Start from $p^{(0)} = 0.5$ and compute the first two iterates for $n = 20, x = 14$.

??? success "Solution to Exercise 3"
    The log-likelihood (ignoring the constant) is:

    $$
    \ell(p) = x\log p + (n - x)\log(1 - p)
    $$

    Score: $\ell'(p) = x/p - (n-x)/(1-p)$.

    Hessian: $\ell''(p) = -x/p^2 - (n-x)/(1-p)^2$.

    Newton-Raphson update: $p^{(t+1)} = p^{(t)} - \ell'(p^{(t)})/\ell''(p^{(t)})$.

    With $n = 20, x = 14, p^{(0)} = 0.5$:

    - $\ell'(0.5) = 14/0.5 - 6/0.5 = 28 - 12 = 16$
    - $\ell''(0.5) = -14/0.25 - 6/0.25 = -56 - 24 = -80$
    - $p^{(1)} = 0.5 - 16/(-80) = 0.5 + 0.2 = 0.7$

    At $p^{(1)} = 0.7$:

    - $\ell'(0.7) = 14/0.7 - 6/0.3 = 20 - 20 = 0$

    So $p^{(2)} = 0.7$, which is already the MLE $\hat{p} = x/n = 14/20 = 0.7$. Newton-Raphson converged in one step. $\square$

---

**Exercise 4.** For a Gaussian mixture with two components, show that the log-likelihood is unbounded above (hint: let one component's variance shrink to zero around a data point). Why does this not invalidate MLE in practice?

??? success "Solution to Exercise 4"
    Consider the mixture density $\pi \cdot N(x_1, \sigma_1^2) + (1-\pi) \cdot N(\mu_2, \sigma_2^2)$. If we set $\mu_1 = x_1$ (a data point) and let $\sigma_1 \to 0$, the first component's density at $x_1$ diverges as $1/\sigma_1 \to \infty$, making the log-likelihood unbounded.

    In practice, this is not a problem because:

    1. These degenerate solutions correspond to overfitting a single point and are statistically meaningless.
    2. The EM algorithm, the standard method for mixture models, cannot reach such degenerate solutions from reasonable starting values.
    3. Practitioners impose minimum variance constraints or use penalized likelihood.
    4. The *useful* MLE is a local maximum of the likelihood, not the global supremum. $\square$

---

**Exercise 5.** Implement the Fisher scoring algorithm to estimate the parameter $p$ of a Bernoulli distribution. The Fisher information is $I(p) = 1/[p(1-p)]$. Compare the convergence speed to Newton-Raphson for $n = 50$ observations with true $p = 0.3$.

??? success "Solution to Exercise 5"
    ```python
    import numpy as np

    rng = np.random.default_rng(42)
    n = 50
    data = rng.binomial(1, 0.3, n)
    x_sum = data.sum()

    # Newton-Raphson
    p_nr = 0.5
    for i in range(10):
        score = x_sum / p_nr - (n - x_sum) / (1 - p_nr)
        hessian = -x_sum / p_nr**2 - (n - x_sum) / (1 - p_nr)**2
        p_nr = p_nr - score / hessian
        print(f"NR  iter {i+1}: p = {p_nr:.8f}")

    # Fisher scoring (use expected information)
    p_fs = 0.5
    for i in range(10):
        score = x_sum / p_fs - (n - x_sum) / (1 - p_fs)
        fisher_info = n / (p_fs * (1 - p_fs))
        p_fs = p_fs + score / fisher_info
        print(f"FS  iter {i+1}: p = {p_fs:.8f}")
    ```

    Both converge to $\hat{p} = x_{\text{sum}}/n$. For the Bernoulli, the observed and expected information are closely related, so convergence speeds are nearly identical. In general, Fisher scoring can be more stable when the observed Hessian is poorly conditioned. $\square$
