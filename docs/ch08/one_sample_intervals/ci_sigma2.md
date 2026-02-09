# CI for σ²

## Confidence Interval for the Population Variance

When the goal is to estimate the variability in a population, we construct a confidence interval for the population variance $\sigma^2$ (or equivalently, the population standard deviation $\sigma$).

### Formula

$$
\left[\frac{(n-1)s^2}{\chi^2_{\alpha/2,\,n-1}},\;\frac{(n-1)s^2}{\chi^2_{1-\alpha/2,\,n-1}}\right]
$$

where

- $s^2$ is the sample variance (with Bessel's correction, $\text{ddof}=1$),
- $n - 1$ is the degrees of freedom,
- $\chi^2_{\alpha/2, n-1}$ and $\chi^2_{1-\alpha/2, n-1}$ are the lower and upper critical values from the chi-square distribution.

### Sampling Distribution

The pivotal quantity is

$$
\frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}
$$

This result holds **exactly** when the population is normally distributed.

### Conditions for Validity

$$
\text{Chi-square CI for } \sigma^2
\quad\text{if}\quad
\begin{cases}
\text{population distribution is normal, so that sampling distribution is known exactly} \\
n \le 0.1N \text{ (IID approximation)}
\end{cases}
$$

!!! warning "Critical Normality Assumption"
    This CI is **exact only for Normal data**. The pivotal result $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$ holds **if and only if** the population is Normal. If the population is skewed or heavy-tailed, this relationship breaks and the chi-square CI can under- or over-cover, even for large $n$. The CLT that helps means does **not** rescue this variance CI.

### When to Use

- Data plausibly come from a **Normal population** (check with histogram or Q-Q plot; look for symmetry and light tails).
- Measurement-error or process data that are well-modeled by Normal noise.
- Teaching and demonstration of exact small-sample inference under Normality.

### When to Be Cautious

- **Skewed or heavy-tailed** data or notable outliers → chi-square CI can miscover. Consider a bootstrap CI for $\sigma$ or $\sigma^2$ (percentile or BCa), or use a robust scale estimator (e.g., MAD) with bootstrap.
- **Transformations** (e.g., log) may normalize, but then the CI is for the variance on the transformed scale.
- For comparing two variances, the F-interval has the same Normality requirement.

### Python Code

```python
import numpy as np
from scipy.stats import chi2

# Given data
n = 12
sigma = 2.0       # true population std dev (for simulation)
alpha = 0.05       # significance level

# Simulate a sample
rng = np.random.default_rng(42)
x = rng.normal(loc=0, scale=sigma, size=n)

# Sample variance
s2 = x.var(ddof=1)
df = n - 1

# Chi-square critical values
chi2_lo = chi2(df=df).ppf(alpha / 2.0)
chi2_hi = chi2(df=df).ppf(1 - alpha / 2.0)

# Confidence interval for σ²
ci_lower = df * s2 / chi2_hi
ci_upper = df * s2 / chi2_lo

print(f"95% CI for σ²: ({ci_lower:.4f}, {ci_upper:.4f})")
print(f"95% CI for σ:  ({np.sqrt(ci_lower):.4f}, {np.sqrt(ci_upper):.4f})")
```

---

## Simulation: Variance CI Coverage

The following script simulates many samples from a Normal population, constructs chi-square CIs for $\sigma^2$, and tracks how many intervals capture the true variance.

```python
#!/usr/bin/env python3
"""
Variance CI via Chi-square simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

rng_seed = None
n_simulations = 100
n_samples = 12
mu = 0.0
sigma = 2.0
alpha = 0.05
report_sigma_not_sigma2 = False  # if True, show CI for σ instead of σ²


def main():
    if rng_seed is not None:
        np.random.seed(rng_seed)

    true_var = sigma**2
    lowers = np.empty(n_simulations)
    uppers = np.empty(n_simulations)
    centers = np.empty(n_simulations)

    df = n_samples - 1
    chi2_lo = chi2(df=df).ppf(alpha / 2.0)
    chi2_hi = chi2(df=df).ppf(1 - alpha / 2.0)

    for i in range(n_simulations):
        x = np.random.normal(loc=mu, scale=sigma, size=n_samples)
        s2 = x.var(ddof=1)
        lowers[i] = df * s2 / chi2_hi
        uppers[i] = df * s2 / chi2_lo
        centers[i] = s2

    covered = (lowers <= true_var) & (true_var <= uppers)
    n_fail = int((~covered).sum())
    coverage_pct = 100.0 * covered.mean()

    if report_sigma_not_sigma2:
        lowers, uppers, centers = np.sqrt(lowers), np.sqrt(uppers), np.sqrt(centers)
        true_ref = np.sqrt(true_var)
        x_label = "Standard Deviation (σ)"
    else:
        true_ref = true_var
        x_label = "Variance (σ²)"

    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(n_simulations):
        color = "k" if covered[i] else "r"
        ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
        ax.plot(centers[i], i, marker="o", ms=3, color=color)

    ax.axvline(true_ref, linestyle="--", linewidth=1.5, color="r")
    ax.set_title(
        f"{n_simulations} Chi-square CIs | n={n_samples}, df={df}, "
        f"CL={int((1 - alpha) * 100)}% | Fail={n_fail} (Coverage ≈ {coverage_pct:.1f}%)")
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel(x_label)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```
