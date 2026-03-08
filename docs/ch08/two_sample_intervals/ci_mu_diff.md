# CI for μ₁ − μ₂

## Two-Sample Confidence Interval for the Difference of Means

When comparing two populations, we are often interested in the difference between their means. A confidence interval for $\mu_1 - \mu_2$ provides a range of plausible values for this difference, considering sampling variability.

---

## Formulas by Scenario

### Known Variances (z-Interval)

$$
(\bar{X}_1 - \bar{X}_2) \pm z_{\alpha/2} \times \sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}
$$

where $\sigma_1^2$ and $\sigma_2^2$ are the known population variances, $z_{\alpha/2}$ is the critical value satisfying $P(Z > z_{\alpha/2}) = \alpha/2$.

### Unknown, Unequal Variances — Welch's t-Interval

$$
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, \, \text{df}} \times \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$

where the degrees of freedom are computed using the **Welch–Satterthwaite equation**:

$$
\text{df} = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{1}{n_1 - 1}\left(\frac{s_1^2}{n_1}\right)^2 + \frac{1}{n_2 - 1}\left(\frac{s_2^2}{n_2}\right)^2}
$$

!!! tip "Default Choice"
    Prefer Welch's t-interval unless you have strong justification for equal variances.

### Unknown, Equal Variances — Pooled t-Interval

$$
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, \, n_1+n_2-2} \times \sqrt{s_p^2\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}
$$

where the **pooled variance** is

$$
s_p^2 = \frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}
$$

and $\text{df} = n_1 + n_2 - 2$.

### Large Sample Size (z-Interval with Sample Variances)

For $n_1 \ge 30$ and $n_2 \ge 30$, the normal approximation can be used even with unknown, unequal variances:

$$
(\bar{X}_1 - \bar{X}_2) \pm z_{\alpha/2} \times \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$

### Python Code

```python
import numpy as np
import scipy.stats as stats

n1, n2 = 30, 25
mean1, mean2 = 100, 90
s1, s2 = 15, 20
confidence_level = 0.95

# Standard error
standard_error = np.sqrt((s1**2 / n1) + (s2**2 / n2))

# Welch-Satterthwaite degrees of freedom
df = ((s1**2 / n1) + (s2**2 / n2))**2 / (
    ((s1**2 / n1)**2 / (n1 - 1)) + ((s2**2 / n2)**2 / (n2 - 1))
)

# Critical value and margin of error
t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, df)
margin_of_error = t_critical * standard_error

# Confidence interval
confidence_interval = (
    (mean1 - mean2) - margin_of_error,
    (mean1 - mean2) + margin_of_error,
)
print(f"{confidence_interval = }")
```

---

## Examples

### Example 1: 95% CI for Difference in Means (Welch)

Two independent samples: Sample 1 has $n_1 = 30$, $\bar{X}_1 = 100$, $s_1 = 15$; Sample 2 has $n_2 = 25$, $\bar{X}_2 = 90$, $s_2 = 20$.

**Solution.**

$$
\text{SE} = \sqrt{\frac{225}{30} + \frac{400}{25}} = \sqrt{7.5 + 16} = \sqrt{23.5} \approx 4.847
$$

Welch–Satterthwaite degrees of freedom:

$$
\text{df} = \frac{(7.5 + 16)^2}{\frac{7.5^2}{29} + \frac{16^2}{24}} \approx 48.35 \approx 48
$$

With $t_{0.025, 48} \approx 2.011$:

$$
\text{ME} = 2.011 \times 4.847 \approx 9.75
$$

$$
\boxed{(0.25,\ 19.75)}
$$

We are 95% confident that the true difference between the population means lies between 0.25 and 19.75.

---

## Simulation: Two-Sample Mean CI Coverage

```python
#!/usr/bin/env python3
"""
Two-sample mean CI simulation: Welch, pooled, z_known, z_plugin.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

rng_seed = None
n_simulations = 100
n1, n2 = 12, 10
mu1, mu2 = 0.0, 0.5
sigma1, sigma2 = 1.0, 1.5
alpha = 0.05
method = "welch"  # 'welch' | 'pooled' | 'z_known' | 'z_plugin'


def main():
    if rng_seed is not None:
        np.random.seed(rng_seed)

    delta_true = mu1 - mu2
    lowers = np.empty(n_simulations)
    uppers = np.empty(n_simulations)
    centers = np.empty(n_simulations)

    for i in range(n_simulations):
        x = np.random.normal(loc=mu1, scale=sigma1, size=n1)
        y = np.random.normal(loc=mu2, scale=sigma2, size=n2)
        xbar, ybar = x.mean(), y.mean()
        s1, s2 = x.std(ddof=1), y.std(ddof=1)
        diff_hat = xbar - ybar
        centers[i] = diff_hat

        if method == "welch":
            se = np.sqrt(s1**2 / n1 + s2**2 / n2)
            num = (s1**2 / n1 + s2**2 / n2) ** 2
            den = (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
            df = num / den
            crit = t.ppf(1 - alpha / 2.0, df=df)
        elif method == "pooled":
            df = n1 + n2 - 2
            sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df
            se = np.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
            crit = t.ppf(1 - alpha / 2.0, df=df)
        elif method == "z_known":
            se = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
            crit = norm.ppf(1 - alpha / 2.0)
        else:  # z_plugin
            se = np.sqrt(s1**2 / n1 + s2**2 / n2)
            crit = norm.ppf(1 - alpha / 2.0)

        lowers[i] = diff_hat - crit * se
        uppers[i] = diff_hat + crit * se

    covered = (lowers <= delta_true) & (delta_true <= uppers)
    n_fail = int((~covered).sum())
    coverage_pct = 100.0 * covered.mean()

    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(n_simulations):
        color = "k" if covered[i] else "r"
        ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
        ax.plot(centers[i], i, marker="o", ms=3, color=color)
    ax.axvline(delta_true, linestyle="--", linewidth=1.5)
    ax.set_title(
        f"{n_simulations} Two-Sample Mean CIs ({method}) | n1={n1}, n2={n2}, "
        f"CL={int((1 - alpha) * 100)}% | Fail={n_fail} (Coverage ≈ {coverage_pct:.1f}%)")
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Δ = μ₁ − μ₂")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

---

## Key Points

- When comparing two population means, we construct a confidence interval for $\mu_1 - \mu_2$.
- If the population variances are unknown and unequal, use **Welch's t-interval** (the default).
- If variances are assumed equal, the **pooled t-interval** uses a combined variance estimate.
- The width of the confidence interval depends on the sample sizes, sample variances, and confidence level.

---

## Exercises

### Exercise: Standard Error of the Difference

Two independent samples: population A with $\sigma_A = 15$ ($n_A = 36$) and population B with $\sigma_B = 20$ ($n_B = 49$). What is the standard error of $\bar{X}_A - \bar{X}_B$?

**Solution.**

$$
\sigma_{\bar{X}_A - \bar{X}_B} = \sqrt{\frac{15^2}{36} + \frac{20^2}{49}} \approx \boxed{3.80}
$$

### Exercise: 95% CI for Difference of Means

Sample 1: $\bar{X}_1 = 55$, $s_1 = 8$, $n_1 = 30$. Sample 2: $\bar{X}_2 = 50$, $s_2 = 10$, $n_2 = 35$. Construct a 95% CI.

**Solution.** Since both $n_1, n_2 \ge 30$, we can use the $z$-approximation:

$$
\text{SE} = \sqrt{\frac{64}{30} + \frac{100}{35}} = \sqrt{2.133 + 2.857} = \sqrt{4.99} \approx 2.233
$$

$$
\text{ME} = 1.96 \times 2.233 \approx 4.38
$$

$$
\boxed{(0.62,\ 9.38)}
$$
