# CI for p₁ − p₂

## Two-Sample Proportion Confidence Interval

In many practical situations, we compare the proportions of two populations — for instance, the proportion of people who support two different policies or the defect rates from two production lines.

### Formula (Wald)

Let $p_1$ and $p_2$ be the population proportions for two independent groups. The confidence interval for $p_1 - p_2$ is

$$
(\hat{p}_1 - \hat{p}_2) \pm z_{\alpha/2} \times \sqrt{\frac{\hat{p}_1(1 - \hat{p}_1)}{n_1} + \frac{\hat{p}_2(1 - \hat{p}_2)}{n_2}}
$$

where $\hat{p}_1 = x_1/n_1$ and $\hat{p}_2 = x_2/n_2$ are the sample proportions.

### Conditions for Validity

For the normal approximation to hold:

- $n_1\hat{p}_1 \ge 5$ and $n_1(1 - \hat{p}_1) \ge 5$,
- $n_2\hat{p}_2 \ge 5$ and $n_2(1 - \hat{p}_2) \ge 5$.

### Alternative Methods

| Method | Description | When to Use |
|---|---|---|
| **Wald** | $\Delta \pm z \cdot \text{SE}$ | Large $n$, not near 0 or 1 |
| **Newcombe (Wilson-based)** | Wilson CI per group, then combine: $[L_1 - U_2,\; U_1 - L_2]$ | **Recommended default** |
| **Clopper–Pearson combined** | Exact CI per group, then combine | Small $n$, regulatory settings |

### Python Code

```python
import numpy as np
import scipy.stats as stats

n1, n2 = 200, 250
x1, x2 = 120, 130
confidence_level = 0.95

p1 = x1 / n1
p2 = x2 / n2

standard_error = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
margin_of_error = z_critical * standard_error

confidence_interval = ((p1 - p2) - margin_of_error, (p1 - p2) + margin_of_error)
print(f"{confidence_interval = }")
```

---

## Examples

### Example 1: 95% CI for Difference in Proportions

Sample 1: $n_1 = 200$, $x_1 = 120$ successes. Sample 2: $n_2 = 250$, $x_2 = 130$ successes.

**Solution.**

$$
\hat{p}_1 = 0.60, \qquad \hat{p}_2 = 0.52
$$

$$
\text{SE} = \sqrt{\frac{0.60 \times 0.40}{200} + \frac{0.52 \times 0.48}{250}} = \sqrt{0.0012 + 0.001} = \sqrt{0.0022} \approx 0.0469
$$

$$
\text{ME} = 1.96 \times 0.0469 \approx 0.0919
$$

$$
\boxed{(-0.0119,\ 0.1719)}
$$

We are 95% confident that the true difference lies between $-0.0119$ and $0.1719$. Since the interval includes zero, there is no statistically significant difference at the 95% level.

### Example 2: New High School Construction

Duncan compares support for a new high school in north and south parts of the city.

| Support? | North | South |
|---|---|---|
| Yes | 54 | 77 |
| No | 66 | 63 |
| Total | 120 | 140 |

Construct a 90% CI for $p_N - p_S$.

**Solution.**

```python
import numpy as np
from scipy import stats

n_1 = 120  # north
n_2 = 140  # south
p_1_hat = 54 / n_1
p_2_hat = 77 / n_2

confidence_level = 0.90
alpha = 1 - confidence_level
z_star = -stats.norm().ppf(alpha / 2)
margin_of_error = z_star * np.sqrt(
    p_1_hat * (1 - p_1_hat) / n_1 + p_2_hat * (1 - p_2_hat) / n_2
)
print(f"90% CI: {p_1_hat - p_2_hat:.4f} ± {margin_of_error:.4f}")
```

---

## Simulation: Difference of Two Proportions CI Coverage

```python
#!/usr/bin/env python3
"""
Difference of two proportions CI simulation: Newcombe, Wald, Clopper-Pearson.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta

rng_seed = None
n_simulations = 100
n1, n2 = 50, 40
p1_true, p2_true = 0.60, 0.50
alpha = 0.05
method = "newcombe"  # 'newcombe' | 'wald' | 'cp'


def main():
    if rng_seed is not None:
        np.random.seed(rng_seed)

    delta_true = p1_true - p2_true
    z = norm.ppf(1 - alpha / 2.0)
    lowers = np.empty(n_simulations)
    uppers = np.empty(n_simulations)
    centers = np.empty(n_simulations)

    for i in range(n_simulations):
        k1 = np.random.binomial(n1, p1_true)
        k2 = np.random.binomial(n2, p2_true)
        p1hat, p2hat = k1 / n1, k2 / n2
        centers[i] = p1hat - p2hat

        if method == "wald":
            se = np.sqrt(p1hat * (1 - p1hat) / n1 + p2hat * (1 - p2hat) / n2)
            lo, hi = centers[i] - z * se, centers[i] + z * se
        elif method == "newcombe":
            denom1 = 1 + z**2 / n1
            center1 = (p1hat + z**2 / (2 * n1)) / denom1
            half1 = z * np.sqrt(p1hat * (1 - p1hat) / n1 + z**2 / (4 * n1**2)) / denom1
            L1, U1 = center1 - half1, center1 + half1
            denom2 = 1 + z**2 / n2
            center2 = (p2hat + z**2 / (2 * n2)) / denom2
            half2 = z * np.sqrt(p2hat * (1 - p2hat) / n2 + z**2 / (4 * n2**2)) / denom2
            L2, U2 = center2 - half2, center2 + half2
            lo, hi = L1 - U2, U1 - L2
        elif method == "cp":
            L1 = 0.0 if k1 == 0 else beta.ppf(alpha / 2.0, k1, n1 - k1 + 1)
            U1 = 1.0 if k1 == n1 else beta.ppf(1 - alpha / 2.0, k1 + 1, n1 - k1)
            L2 = 0.0 if k2 == 0 else beta.ppf(alpha / 2.0, k2, n2 - k2 + 1)
            U2 = 1.0 if k2 == n2 else beta.ppf(1 - alpha / 2.0, k2 + 1, n2 - k2)
            lo, hi = L1 - U2, U1 - L2

        lowers[i] = max(-1.0, lo)
        uppers[i] = min(1.0, hi)

    covered = (lowers <= delta_true) & (delta_true <= uppers)
    n_fail = int((~covered).sum())
    coverage_pct = 100.0 * covered.mean()

    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(n_simulations):
        color = "k" if covered[i] else "r"
        ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
        ax.plot(centers[i], i, marker="o", ms=3, color=color)
    ax.axvline(delta_true, linestyle="--", linewidth=1.5, color="r")
    ax.set_title(
        f"{n_simulations} Δ=p1−p2 CIs ({method.title()}) | n1={n1}, n2={n2}, "
        f"CL={int((1 - alpha) * 100)}% | Fail={n_fail} (Coverage ≈ {coverage_pct:.1f}%)")
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Δ = p1 − p2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

---

## Key Points

- The confidence interval for $p_1 - p_2$ uses the normal approximation to the binomial distribution, assuming large sample sizes.
- The width depends on the sample proportions, sample sizes, and confidence level.
- If the confidence interval includes zero, there is no statistically significant difference between the two proportions at the given confidence level.
- The **Newcombe (Wilson-based)** method is recommended as the default for better coverage, especially with moderate sample sizes.

---

## Exercise

### Exercise: 95% CI for Difference Between Two Proportions

In two independent samples, 150 out of 200 prefer a brand in sample 1, and 120 out of 180 prefer the same brand in sample 2. Construct a 95% CI.

**Solution.**

$$
\hat{p}_1 = 0.75, \qquad \hat{p}_2 = 0.67
$$

$$
\text{SE} = \sqrt{\frac{0.75 \times 0.25}{200} + \frac{0.67 \times 0.33}{180}} \approx 0.047
$$

$$
\text{ME} = 1.96 \times 0.047 \approx 0.092
$$

$$
\boxed{(-0.012,\ 0.172)}
$$
