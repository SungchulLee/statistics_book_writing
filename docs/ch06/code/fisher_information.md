# Fisher Information Computation

## Overview

The **Fisher information** quantifies how much information a random sample carries about an unknown parameter. It plays a central role in estimation theory: it determines the best achievable precision (the Cramer-Rao lower bound), governs the asymptotic variance of the MLE, and guides experimental design. This page demonstrates both analytical and numerical computation of Fisher information.

## Definition

Let $X$ have density (or PMF) $f(x; \theta)$. The **score function** is:

$$
S(\theta) = \frac{\partial}{\partial\theta}\log f(X; \theta)
$$

The **Fisher information** for a single observation is:

$$
I(\theta) = E\left[S(\theta)^2\right] = \text{Var}\left[S(\theta)\right]
$$

Under regularity conditions, this equals the negative expected second derivative:

$$
I(\theta) = -E\left[\frac{\partial^2}{\partial\theta^2}\log f(X; \theta)\right]
$$

For $n$ iid observations, the total Fisher information is $I_n(\theta) = nI(\theta)$.

## Key Theoretical Results

!!! info "Cramer-Rao Lower Bound"
    For any unbiased estimator $\hat{\theta}$:

    $$
    \text{Var}(\hat{\theta}) \geq \frac{1}{nI(\theta)}
    $$

    An estimator achieving this bound is called **efficient**.

!!! info "Asymptotic Normality of the MLE"
    Under regularity conditions:

    $$
    \hat{\theta}_{\text{MLE}} \overset{d}{\to} N\!\left(\theta,\, \frac{1}{nI(\theta)}\right)
    $$

    The MLE is asymptotically efficient -- its variance achieves the CRLB.

## Analytical Examples

### Normal Mean

For $X \sim N(\mu, \sigma^2)$ with $\sigma^2$ known:

$$
\log f(x; \mu) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}
$$

$$
S(\mu) = \frac{x - \mu}{\sigma^2}, \qquad -\frac{\partial^2}{\partial\mu^2}\log f = \frac{1}{\sigma^2}
$$

$$
I(\mu) = \frac{1}{\sigma^2}
$$

### Bernoulli

For $X \sim \text{Bernoulli}(p)$:

$$
\log f(x; p) = x\log p + (1-x)\log(1-p)
$$

$$
I(p) = \frac{1}{p(1-p)}
$$

### Poisson

For $X \sim \text{Poisson}(\lambda)$:

$$
\log f(x; \lambda) = x\log\lambda - \lambda - \log(x!)
$$

$$
I(\lambda) = \frac{1}{\lambda}
$$

### Exponential

For $X \sim \text{Exp}(\lambda)$ (rate parameterization):

$$
I(\lambda) = \frac{1}{\lambda^2}
$$

## Numerical Computation of Fisher Information

When the Fisher information cannot be computed in closed form, we can estimate it numerically by:

1. **Score variance method**: Sample $X_1, \ldots, X_N$ from $f(x; \theta)$, compute the score at each point, and estimate $I(\theta) \approx \text{Var}(\{S_i\})$.
2. **Finite difference method**: Approximate the score by $S(\theta) \approx [\log f(X; \theta + \delta) - \log f(X; \theta - \delta)]/(2\delta)$.

```python
import numpy as np
from scipy import stats

def fisher_information_numerical(dist_name="norm", true_params=None,
                                  param_name="loc", n_samples=100_000, delta=1e-5):
    """Compute Fisher information numerically via the score variance."""
    if true_params is None:
        true_params = {"loc": 5, "scale": 2}

    dist = getattr(stats, dist_name)
    rng = np.random.default_rng(42)
    data = dist.rvs(size=n_samples, random_state=rng, **true_params)

    theta = true_params[param_name]

    # Finite-difference approximation to the score
    params_plus = {**true_params, param_name: theta + delta}
    params_minus = {**true_params, param_name: theta - delta}

    logf_plus = dist.logpdf(data, **params_plus)
    logf_minus = dist.logpdf(data, **params_minus)
    score = (logf_plus - logf_minus) / (2 * delta)

    I_numerical = np.var(score)
    return I_numerical

# Normal mean: theoretical I(mu) = 1/sigma^2
I_num = fisher_information_numerical("norm", {"loc": 5, "scale": 2}, "loc")
I_theory = 1 / 2**2
print(f"Normal mean Fisher information:")
print(f"  Numerical:   I(mu) = {I_num:.6f}")
print(f"  Theoretical: I(mu) = {I_theory:.6f}")
```

## Verifying the Cramer-Rao Bound

We can check that the sample mean achieves the CRLB for the normal mean.

```python
import numpy as np

def verify_crlb(mu_true=5.0, sigma=2.0, n=50, n_sim=20_000):
    """Verify that the sample mean achieves the CRLB."""
    rng = np.random.default_rng(42)
    crlb = sigma**2 / n

    means = np.array([rng.normal(mu_true, sigma, n).mean() for _ in range(n_sim)])
    empirical_var = means.var()

    print(f"CRLB = sigma^2/n = {crlb:.6f}")
    print(f"Var(X_bar)       = {empirical_var:.6f}")
    print(f"Ratio            = {empirical_var / crlb:.4f}")

    # Compare with median (does not achieve CRLB)
    medians = np.array([np.median(rng.normal(mu_true, sigma, n)) for _ in range(n_sim)])
    print(f"\nVar(median) = {medians.var():.6f}")
    print(f"Efficiency of median = {crlb / medians.var():.4f}")

verify_crlb()
```

!!! note "Efficiency of the Median"
    For the normal distribution, the asymptotic relative efficiency of the median versus the mean is $2/\pi \approx 0.637$. The median uses only about 64% of the information in the data.

## Multiparameter Fisher Information Matrix

For a parameter vector $\boldsymbol{\theta} = (\theta_1, \ldots, \theta_k)$, the Fisher information is a $k \times k$ matrix:

$$
[I(\boldsymbol{\theta})]_{ij} = -E\left[\frac{\partial^2}{\partial\theta_i\,\partial\theta_j}\log f(X; \boldsymbol{\theta})\right]
$$

For the normal distribution with both $\mu$ and $\sigma^2$ unknown:

$$
I(\mu, \sigma^2) = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 1/(2\sigma^4) \end{pmatrix}
$$

The off-diagonal zeros show that $\mu$ and $\sigma^2$ are informationally orthogonal.

## Interpretation

- Fisher information measures the **curvature** of the log-likelihood at the true parameter: high curvature means the data are informative and the MLE is precise.
- A larger $I(\theta)$ implies a tighter Cramer-Rao bound, so estimators can be more precise.
- Fisher information depends on the true parameter value. For example, $I(p) = 1/[p(1-p)]$ for Bernoulli is largest when $p$ is near 0 or 1 (each observation is highly informative) and smallest at $p = 0.5$ (maximum uncertainty).

## Exercises

**Exercise 1.** Derive the Fisher information for the Poisson distribution $P(\lambda)$ using both the score variance definition and the negative expected second derivative. Verify they agree.

??? success "Solution to Exercise 1"
    The log-PMF is $\log f(x; \lambda) = x\log\lambda - \lambda - \log(x!)$.

    **Score:** $S(\lambda) = X/\lambda - 1$.

    **Score variance:** $\text{Var}(S) = \text{Var}(X/\lambda) = \text{Var}(X)/\lambda^2 = \lambda/\lambda^2 = 1/\lambda$.

    **Negative second derivative:** $-\partial^2\log f/\partial\lambda^2 = X/\lambda^2$, so $E[-\partial^2\log f/\partial\lambda^2] = E[X]/\lambda^2 = \lambda/\lambda^2 = 1/\lambda$.

    Both give $I(\lambda) = 1/\lambda$. $\square$

---

**Exercise 2.** For the exponential distribution with rate $\lambda$ (density $f(x; \lambda) = \lambda e^{-\lambda x}$ for $x > 0$), compute the Fisher information and the Cramer-Rao lower bound for estimating $\lambda$ from $n$ observations.

??? success "Solution to Exercise 2"
    The log-density is $\log f(x; \lambda) = \log\lambda - \lambda x$.

    Score: $S(\lambda) = 1/\lambda - X$. Second derivative: $\partial^2\log f/\partial\lambda^2 = -1/\lambda^2$.

    Fisher information: $I(\lambda) = 1/\lambda^2$.

    The CRLB for $n$ observations is:

    $$
    \text{Var}(\hat{\lambda}) \geq \frac{1}{nI(\lambda)} = \frac{\lambda^2}{n}
    $$

    The MLE is $\hat{\lambda} = 1/\bar{X}$. By the delta method, $\text{Var}(\hat{\lambda}) \approx \lambda^2/n$ for large $n$, so the MLE achieves the bound asymptotically. $\square$

---

**Exercise 3.** Show that for the Bernoulli distribution, the Fisher information $I(p) = 1/[p(1-p)]$ is minimized at $p = 1/2$. Interpret this result in terms of coin flipping.

??? success "Solution to Exercise 3"
    Taking the derivative: $\frac{d}{dp}I(p) = \frac{d}{dp}[p(1-p)]^{-1} = -\frac{1-2p}{[p(1-p)]^2}$.

    Setting this to zero gives $p = 1/2$. Since $I(p) \to \infty$ as $p \to 0$ or $p \to 1$, and $I(1/2) = 4$ is a finite minimum, $p = 1/2$ minimizes the Fisher information.

    **Interpretation:** A fair coin ($p = 1/2$) is the hardest to distinguish from nearby values. Each flip provides the least information about $p$ when outcomes are most uncertain. Conversely, when $p$ is near 0 or 1, each flip is very informative because outcomes are highly predictable, and any deviation from that pattern is strongly diagnostic. $\square$

---

**Exercise 4.** The Fisher information for a Gamma distribution $\text{Gamma}(\alpha, \beta)$ with respect to $\alpha$ involves the trigamma function $\psi_1(\alpha)$: $I_{\alpha\alpha} = \psi_1(\alpha)$. Use `scipy.special.polygamma(1, alpha)` to numerically compute the CRLB for estimating $\alpha$ from $n = 100$ observations at $\alpha = 3$.

??? success "Solution to Exercise 4"
    ```python
    from scipy.special import polygamma

    alpha = 3.0
    n = 100
    I_alpha = polygamma(1, alpha)  # trigamma function
    crlb = 1 / (n * I_alpha)
    print(f"Trigamma(alpha=3) = {I_alpha:.6f}")
    print(f"CRLB for alpha:     {crlb:.6f}")
    ```

    The trigamma function $\psi_1(3) \approx 0.3949$, giving a CRLB of approximately $1/(100 \times 0.3949) \approx 0.0253$. This means no unbiased estimator of $\alpha$ can have variance smaller than about 0.025 with 100 observations. $\square$

---

**Exercise 5.** Prove that the Fisher information satisfies the additive property: for $n$ iid observations, $I_n(\theta) = nI_1(\theta)$. Why does this make intuitive sense?

??? success "Solution to Exercise 5"
    Let $X_1, \ldots, X_n$ be iid with density $f(x; \theta)$. The joint log-likelihood is:

    $$
    \ell_n(\theta) = \sum_{i=1}^n \log f(X_i; \theta)
    $$

    The total score is $S_n(\theta) = \sum_{i=1}^n S_i(\theta)$ where $S_i(\theta) = \partial\log f(X_i; \theta)/\partial\theta$.

    Since the $X_i$ are independent, the $S_i$ are independent with $E[S_i] = 0$ and $\text{Var}(S_i) = I_1(\theta)$. Therefore:

    $$
    I_n(\theta) = \text{Var}(S_n) = \sum_{i=1}^n \text{Var}(S_i) = nI_1(\theta)
    $$

    **Intuition:** Each independent observation contributes the same amount of information about $\theta$. Doubling the sample size doubles the total information, which halves the CRLB and thus halves the minimum achievable variance. $\square$
