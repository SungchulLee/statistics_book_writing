# Berry–Esseen Theorem

## Overview

The Central Limit Theorem guarantees that the standardized sample mean converges to a normal distribution, but it says nothing about **how fast** this convergence occurs. The **Berry–Esseen theorem** fills this gap by providing an explicit upper bound on the approximation error for any finite sample size $n$.

---

## Statement

Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with:

- Mean $\mu = E[X_i]$
- Variance $\sigma^2 = \text{Var}(X_i) > 0$
- Finite third absolute moment $\rho = E\left[|X_i - \mu|^3\right] < \infty$

Let $F_n(x) = P\left(\frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \leq x\right)$ be the CDF of the standardized sample mean, and let $\Phi(x)$ be the CDF of the standard normal. Then:

$$
\sup_{x \in \mathbb{R}} \left|F_n(x) - \Phi(x)\right| \leq \frac{C \cdot \rho}{\sigma^3 \sqrt{n}}
$$

where $C$ is a universal constant. The best known value is $C \leq 0.4748$ (Shevtsova, 2011).

---

## Interpretation

The theorem provides a **non-asymptotic** guarantee: for any finite $n$, the maximum error of the normal approximation is bounded by $O(1/\sqrt{n})$. Key implications:

- The approximation error decreases at rate $1/\sqrt{n}$.
- Distributions with larger third moments (more skewness or heavy tails) converge more slowly.
- The ratio $\rho / \sigma^3$ captures the "non-normality" of the original distribution.

---

## Connection to the CLT

| Aspect | CLT | Berry–Esseen |
|:---|:---|:---|
| **Statement** | $F_n(x) \to \Phi(x)$ as $n \to \infty$ | $\|F_n - \Phi\|_\infty \leq C\rho / (\sigma^3\sqrt{n})$ |
| **Type of result** | Asymptotic | Non-asymptotic (finite $n$) |
| **Rate of convergence** | Not specified | $O(1/\sqrt{n})$ |
| **Assumptions** | Finite $\mu, \sigma^2$ | Finite $\mu, \sigma^2, \rho$ |

The Berry–Esseen theorem **quantifies** what the CLT merely asserts qualitatively.

---

## Examples

### Example: Fair Coin Flips

For $X_i \sim \text{Bernoulli}(0.5)$: $\mu = 0.5$, $\sigma^2 = 0.25$, $\rho = E[|X_i - 0.5|^3] = 0.125$.

$$
\text{Bound} = \frac{0.4748 \times 0.125}{0.25^{3/2} \sqrt{n}} = \frac{0.4748}{n^{1/2}}
$$

For $n = 100$: bound $\approx 0.0475$, meaning the CDF is within 4.75% of the normal CDF at every point.

### Example: Exponential Distribution

For $X_i \sim \text{Exponential}(1)$: $\mu = 1$, $\sigma^2 = 1$, $\rho = E[|X_i - 1|^3] = 2 + e^{-1} \approx 2.368$.

$$
\text{Bound} = \frac{0.4748 \times 2.368}{\sqrt{n}} \approx \frac{1.124}{\sqrt{n}}
$$

For $n = 100$: bound $\approx 0.112$. The larger bound reflects the exponential distribution's skewness—it converges to normality more slowly than the symmetric Bernoulli.

---

## Python Exploration

```python
import numpy as np
from scipy import stats

def berry_esseen_bound(sigma, rho, n, C=0.4748):
    """Compute the Berry-Esseen upper bound."""
    return C * rho / (sigma**3 * np.sqrt(n))

# Bernoulli(0.5)
sigma_b = np.sqrt(0.25)
rho_b = 0.125
print("=== Bernoulli(0.5) ===")
for n in [10, 30, 100, 1000]:
    bound = berry_esseen_bound(sigma_b, rho_b, n)
    print(f"n = {n:5d}: Berry–Esseen bound = {bound:.4f}")

print()

# Exponential(1)
sigma_e = 1.0
rho_e = 2.368
print("=== Exponential(1) ===")
for n in [10, 30, 100, 1000]:
    bound = berry_esseen_bound(sigma_e, rho_e, n)
    print(f"n = {n:5d}: Berry–Esseen bound = {bound:.4f}")
```

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def berry_esseen_visualization(dist_name, rvs_fn, mu, sigma, rho, sample_sizes):
    """Compare the actual CDF error with the Berry-Esseen bound."""
    C = 0.4748
    x_grid = np.linspace(-4, 4, 1000)
    n_sim = 50_000

    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(12, 3),
                             sharey=True)
    fig.suptitle(f'Berry–Esseen: {dist_name}', fontsize=14)

    for ax, n in zip(axes, sample_sizes):
        np.random.seed(42)
        # Simulate standardized sample means
        samples = rvs_fn(size=(n_sim, n))
        x_bar = samples.mean(axis=1)
        z = (x_bar - mu) / (sigma / np.sqrt(n))

        # Empirical CDF vs normal CDF
        ecdf = np.array([np.mean(z <= x) for x in x_grid])
        ncdf = stats.norm.cdf(x_grid)
        actual_error = np.abs(ecdf - ncdf)
        max_error = actual_error.max()

        bound = C * rho / (sigma**3 * np.sqrt(n))

        ax.plot(x_grid, actual_error, lw=1.5, label=f'Actual max: {max_error:.4f}')
        ax.axhline(bound, color='r', linestyle='--', lw=1.5,
                   label=f'BE bound: {bound:.4f}')
        ax.set_title(f'n = {n}')
        ax.set_xlabel('x')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel('|Fₙ(x) − Φ(x)|')
    plt.tight_layout()
    plt.show()

# Exponential(1): skewed distribution
berry_esseen_visualization(
    'Exponential(1)',
    lambda size: np.random.exponential(1, size),
    mu=1.0, sigma=1.0, rho=2.368,
    sample_sizes=[5, 30, 100]
)
```

```python
import numpy as np
import matplotlib.pyplot as plt

def convergence_rate_comparison():
    """Compare convergence rates for different distributions."""
    C = 0.4748
    ns = np.arange(5, 501)

    distributions = {
        'Bernoulli(0.5)': {'sigma': np.sqrt(0.25), 'rho': 0.125},
        'Uniform(0,1)':   {'sigma': 1/np.sqrt(12), 'rho': 1/32},
        'Exponential(1)': {'sigma': 1.0,            'rho': 2.368},
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    for name, params in distributions.items():
        bounds = C * params['rho'] / (params['sigma']**3 * np.sqrt(ns))
        ax.plot(ns, bounds, label=name, lw=2)

    ax.set_xlabel('Sample Size n')
    ax.set_ylabel('Berry–Esseen Bound')
    ax.set_title('Convergence Rate to Normal: Berry–Esseen Bounds')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

convergence_rate_comparison()
```

---

## Key Takeaways

- The Berry–Esseen theorem gives a **finite-sample** bound on the normal approximation error: $O(1/\sqrt{n})$.
- The bound depends on $\rho / \sigma^3$—distributions with more skewness or heavier tails converge more slowly.
- It complements the CLT by answering "how large must $n$ be?" for a desired approximation accuracy.
- Symmetric distributions (e.g., Bernoulli(0.5)) converge faster than skewed ones (e.g., Exponential).
