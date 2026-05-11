# Berry–Esseen Theorem

## Overview

The Central Limit Theorem guarantees that the standardized sample mean converges to a normal distribution, but it says nothing about **how fast** this convergence occurs. The **Berry–Esseen theorem** fills this gap by providing an explicit upper bound on the approximation error for any finite sample size $n$.

---

## Statement

Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables with:

- Mean $\mu = E[X_i]$
- Variance $\sigma^2 = \text{Var}(X_i) > 0$
- Finite third absolute moment $\rho = E\left[|X_i - \mu|^3\right] < \infty$

Let $F_n(x) = P\left(\frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \leq x\right)$ be the CDF of the standardized sample mean, and let $\mathcal{N}(x)$ be the CDF of the standard normal. Then:

$$
\sup_{x \in \mathbb{R}} \left|F_n(x) - \mathcal{N}(x)\right| \leq \frac{C \cdot \rho}{\sigma^3 \sqrt{n}}
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
| **Statement** | $F_n(x) \to \mathcal{N}(x)$ as $n \to \infty$ | $\|F_n - \mathcal{N}\|_\infty \leq C\rho / (\sigma^3\sqrt{n})$ |
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

## Exercises

**Exercise 1.**
The Berry-Esseen theorem states that $\sup_x |F_n(x) - \mathcal{N}(x)| \leq \frac{C \rho}{\sigma^3 \sqrt{n}}$ where $C \leq 0.4748$. For the Bernoulli(0.5) distribution, $\sigma^2 = 0.25$ and $\rho = E[|X - \mu|^3] = 0.125$. How large must $n$ be for the Berry-Esseen bound to guarantee an approximation error of at most 0.01?

??? success "Solution to Exercise 1"
    We need:

    $$
    \frac{C \rho}{\sigma^3 \sqrt{n}} \leq 0.01
    $$

    With $C = 0.4748$, $\rho = 0.125$, and $\sigma = 0.5$:

    $$
    \frac{0.4748 \times 0.125}{0.5^3 \sqrt{n}} \leq 0.01
    $$

    $$
    \frac{0.05935}{0.125 \sqrt{n}} \leq 0.01
    $$

    $$
    \frac{0.4748}{\sqrt{n}} \leq 0.01
    $$

    $$
    \sqrt{n} \geq 47.48 \implies n \geq 2254.3
    $$

    Therefore $n \geq 2255$ is sufficient to guarantee the normal approximation error is at most 0.01 for the Bernoulli(0.5) case.

---

**Exercise 2.**
Consider the Exponential(1) distribution with $\mu = 1$, $\sigma = 1$, and $\rho = E[|X-1|^3] \approx 2.368$. Compare the Berry-Esseen bound for $n = 30$ to the bound for Bernoulli(0.5) at the same sample size. Which distribution converges faster to normal and why?

??? success "Solution to Exercise 2"
    **Bernoulli(0.5):** With $\rho = 0.125$ and $\sigma = 0.5$:

    $$
    \text{Bound} = \frac{0.4748 \times 0.125}{0.5^3 \sqrt{30}} = \frac{0.05935}{0.125 \times 5.477} = \frac{0.05935}{0.6847} \approx 0.0867
    $$

    **Exponential(1):** With $\rho = 2.368$ and $\sigma = 1$:

    $$
    \text{Bound} = \frac{0.4748 \times 2.368}{1^3 \sqrt{30}} = \frac{1.1243}{5.477} \approx 0.2053
    $$

    The Bernoulli(0.5) bound (0.087) is much smaller than the Exponential(1) bound (0.205). The Bernoulli(0.5) distribution converges faster because it is symmetric ($\rho/\sigma^3$ is small), while the Exponential(1) is right-skewed with a large third absolute moment relative to $\sigma^3$.

---

**Exercise 3.**
Explain why the Berry-Esseen theorem is necessary given that the Central Limit Theorem already guarantees convergence to the normal distribution.

??? success "Solution to Exercise 3"
    The CLT is an **asymptotic** result: it states that $\bar{X}_n$ converges in distribution to a normal as $n \to \infty$, but it says nothing about how good the approximation is for any finite sample size $n$. In practice, we always work with finite samples, so we need to know whether $n = 30$ or $n = 100$ or $n = 10{,}000$ is "large enough."

    The Berry-Esseen theorem fills this gap by providing an explicit, finite-sample **bound** on the maximum error of the normal approximation. It answers the practical question: "For my specific distribution and sample size, how accurate is the CLT approximation?" This is particularly important for skewed or heavy-tailed distributions, where the normal approximation may require much larger sample sizes than one might naively assume.

---

**Exercise 4.**
The Berry-Esseen bound decreases as $O(1/\sqrt{n})$. If you need to reduce the approximation error by a factor of 10 (e.g., from 0.1 to 0.01), by what factor must you increase the sample size?

??? success "Solution to Exercise 4"
    Since the bound is proportional to $1/\sqrt{n}$, reducing the bound by a factor of 10 requires:

    $$
    \frac{1}{\sqrt{n_{\text{new}}}} = \frac{1}{10} \cdot \frac{1}{\sqrt{n_{\text{old}}}}
    $$

    $$
    \sqrt{n_{\text{new}}} = 10 \sqrt{n_{\text{old}}}
    $$

    $$
    n_{\text{new}} = 100 \cdot n_{\text{old}}
    $$

    You must increase the sample size by a factor of **100** to reduce the approximation error by a factor of 10. This $O(1/\sqrt{n})$ rate of convergence is relatively slow, which explains why very large samples are sometimes needed for accurate normal approximations, especially for skewed distributions.

---

**Exercise 5.**
Show the **Edgeworth expansion** refines the CLT to second order: the CDF of $\sqrt n (\bar X_n - \mu)/\sigma$ can be approximated by $\Phi(x) + (\gamma_1/(6\sqrt n)) \phi(x) (1 - x^2) + O(1/n)$, where $\gamma_1$ is the skewness. Explain how this provides a more accurate approximation than the plain CLT.

??? success "Solution to Exercise 5"
    The plain CLT approximates the CDF by $\Phi(x)$, with error $O(1/\sqrt n)$ — the Berry-Esseen rate.

    The **Edgeworth expansion** adds a correction term proportional to the skewness $\gamma_1$ of the parent distribution:

    $$
    P\!\left(\frac{\sqrt n (\bar X_n - \mu)}{\sigma} \le x\right) = \Phi(x) - \frac{\gamma_1}{6\sqrt n}(x^2 - 1)\phi(x) + O(1/n)
    $$

    This reduces the residual error from $O(1/\sqrt n)$ to $O(1/n)$ — an order of magnitude tighter. Further terms involve kurtosis ($\gamma_2$) and higher cumulants.

    **Practical use:** for moderate $n$ with heavily-skewed data, the Edgeworth correction can be substantially more accurate than plain Gaussian. Many high-precision approximations in statistical software use Edgeworth or related saddlepoint corrections.

    **Caveat:** Edgeworth expansions can produce negative "probability density" in the tails, so they should not be used outside their region of validity (typically $|x| \le 2$ or so).

---

**Exercise 6.**
**Berry-Esseen with non-i.i.d.** sums. State a generalization for independent (but not identically distributed) random variables. Why is this important for regression and time-series analysis?

??? success "Solution to Exercise 6"
    **Generalized Berry-Esseen:** for independent (not necessarily identical) $X_i$ with $\mathbb{E}[X_i] = 0$, $\mathrm{Var}(X_i) = \sigma_i^2$, $\mathbb{E}[|X_i|^3] = \rho_i$, $B_n^2 = \sum_i \sigma_i^2$:

    $$
    \sup_x \left| P\!\left(\frac{S_n}{B_n} \le x\right) - \Phi(x) \right| \le \frac{C \sum_i \rho_i}{B_n^3}
    $$

    The bound depends on the *sum* of third moments relative to the cube of the sum's standard deviation. When one $X_i$ has disproportionately large $\rho_i$, the bound is loose — reflecting that one large term can prevent Gaussian convergence.

    **Importance for applied statistics:**

    - **Regression**: errors $\varepsilon_i$ may have different variances (heteroscedasticity), so a non-i.i.d. CLT is needed to justify normal-based confidence intervals for $\hat\beta$.
    - **Time series**: weakly dependent (mixing) sequences satisfy CLTs with appropriate corrections.
    - **Survey sampling**: stratified samples mix independent contributions from different strata with different variances.

    Modern asymptotic statistics relies on these generalizations to legitimize the Gaussian CLT in real-data settings where the i.i.d. assumption rarely holds exactly.
