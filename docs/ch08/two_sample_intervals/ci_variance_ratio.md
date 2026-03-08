# CI for σ₁² / σ₂²

## Confidence Interval for the Ratio of Two Variances

When comparing the variability of two independent populations, we construct a confidence interval for the ratio $\theta = \sigma_1^2 / \sigma_2^2$. This is based on the F-distribution.

### Formula

$$
\left[\frac{s_1^2 / s_2^2}{F_{\alpha/2,\, n_1-1,\, n_2-1}},\;\; \frac{s_1^2 / s_2^2}{F_{1-\alpha/2,\, n_1-1,\, n_2-1}}\right]
$$

where

- $s_1^2$ and $s_2^2$ are the sample variances (with Bessel's correction),
- $F_{\alpha/2, n_1-1, n_2-1}$ and $F_{1-\alpha/2, n_1-1, n_2-1}$ are the critical values from the F-distribution with degrees of freedom $\text{df}_1 = n_1 - 1$ and $\text{df}_2 = n_2 - 1$.

### Sampling Distribution

The pivotal quantity is

$$
\frac{s_1^2 / \sigma_1^2}{s_2^2 / \sigma_2^2} \sim F_{n_1-1, \, n_2-1}
$$

This result holds exactly when both populations are normally distributed.

### Conditions for Validity

$$
\text{F-interval for } \sigma_1^2/\sigma_2^2
\quad\text{if}\quad
\begin{cases}
\text{both population distributions are normal} \\
n_i \le 0.1 N_i \text{ for each group (IID approximation)}
\end{cases}
$$

!!! warning "Normality Requirement"
    Like the chi-square variance CI, the F-interval is **exact only under Normality**. Non-Normal data (skewed, heavy-tailed, or with outliers) can cause serious miscoverage. Consider bootstrap or robust alternatives for non-Normal data.

### Python Code

```python
import numpy as np
from scipy.stats import f

n1, n2 = 15, 12
alpha = 0.05

# Simulate samples
rng = np.random.default_rng(42)
x = rng.normal(loc=0, scale=1.0, size=n1)
y = rng.normal(loc=0, scale=1.5, size=n2)

s1_sq = x.var(ddof=1)
s2_sq = y.var(ddof=1)
rhat = s1_sq / s2_sq

df1, df2 = n1 - 1, n2 - 1
F_lo = f(dfn=df1, dfd=df2).ppf(alpha / 2.0)
F_hi = f(dfn=df1, dfd=df2).ppf(1 - alpha / 2.0)

ci_lower = rhat / F_hi
ci_upper = rhat / F_lo

print(f"95% CI for σ₁²/σ₂²: ({ci_lower:.4f}, {ci_upper:.4f})")
```

---

## Simulation: Variance Ratio CI Coverage

```python
#!/usr/bin/env python3
"""
F-interval simulation for θ = σ₁²/σ₂².
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

rng_seed = None
n_simulations = 100
n1, n2 = 15, 12
mu1, mu2 = 0.0, 0.0
sigma1, sigma2 = 1.0, 1.5
alpha = 0.05


def main():
    if rng_seed is not None:
        np.random.seed(rng_seed)

    theta_true = (sigma1**2) / (sigma2**2)
    df1, df2 = n1 - 1, n2 - 1

    lowers = np.empty(n_simulations)
    uppers = np.empty(n_simulations)
    centers = np.empty(n_simulations)

    F_lo = f(dfn=df1, dfd=df2).ppf(alpha / 2.0)
    F_hi = f(dfn=df1, dfd=df2).ppf(1 - alpha / 2.0)

    for i in range(n_simulations):
        x = np.random.normal(loc=mu1, scale=sigma1, size=n1)
        y = np.random.normal(loc=mu2, scale=sigma2, size=n2)
        s1_sq = x.var(ddof=1)
        s2_sq = y.var(ddof=1)
        rhat = s1_sq / s2_sq
        lowers[i] = rhat / F_hi
        uppers[i] = rhat / F_lo
        centers[i] = rhat

    covered = (lowers <= theta_true) & (theta_true <= uppers)
    n_fail = (~covered).sum()
    coverage_pct = 100.0 * covered.mean()

    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(n_simulations):
        color = "k" if covered[i] else "r"
        ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
        ax.plot(centers[i], i, marker="o", ms=3, color=color)

    ax.axvline(theta_true, linestyle="--", linewidth=1.5, color="r")
    ax.set_title(
        f"{n_simulations} F-intervals for σ₁²/σ₂² | n1={n1}, n2={n2}, "
        f"df=({df1},{df2}), CL={int((1 - alpha) * 100)}% | "
        f"Fail={n_fail} (Coverage ≈ {coverage_pct:.1f}%)")
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("θ = σ₁² / σ₂²")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

---

## Key Points

- The F-interval for $\sigma_1^2 / \sigma_2^2$ requires **both populations to be Normal**.
- If the CI includes 1, there is no evidence that the two population variances differ.
- The F-distribution is asymmetric, so the CI is not centered symmetrically around the point estimate.
- For non-Normal data, consider bootstrap-based alternatives.
