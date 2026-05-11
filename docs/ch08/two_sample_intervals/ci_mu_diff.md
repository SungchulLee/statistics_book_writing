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

**Exercise 1.**
Two independent samples: $\sigma_A = 15, n_A = 36$; $\sigma_B = 20, n_B = 49$. Compute $\mathrm{SE}(\bar X_A - \bar X_B)$.

??? success "Solution to Exercise 1"
    $\mathrm{SE} = \sqrt{15^2/36 + 20^2/49} = \sqrt{6.25 + 8.16} = \sqrt{14.41} \approx 3.80$.

---

**Exercise 2.**
Sample 1: $\bar X_1 = 55, s_1 = 8, n_1 = 30$. Sample 2: $\bar X_2 = 50, s_2 = 10, n_2 = 35$. 95% CI for $\mu_1 - \mu_2$.

??? success "Solution to Exercise 2"
    $\mathrm{SE} = \sqrt{64/30 + 100/35} \approx 2.233$. With large $n$'s, use $z$: ME = $1.96 \cdot 2.233 \approx 4.38$.

    CI: $(5 - 4.38, 5 + 4.38) = (0.62, 9.38)$. Does *not* include 0 — evidence of a difference.

---

**Exercise 3.**
Welch's CI for two teaching methods: A ($n = 30, \bar X = 78, s = 8$), B ($n = 35, \bar X = 82, s = 10$). 95% CI for $\mu_A - \mu_B$.

??? success "Solution to Exercise 3"
    $\mathrm{SE} = \sqrt{64/30 + 100/35} \approx 2.234$.

    Welch's df: $\nu = (2.133 + 2.857)^2/[2.133^2/29 + 2.857^2/34] = 24.9/0.397 \approx 62.7$. Use $t_{62}$.

    $t_{0.975, 62} \approx 2.00$. CI: $(78 - 82) \pm 2.00 \cdot 2.234 = -4 \pm 4.47 = (-8.47, 0.47)$.

    Contains 0 — cannot conclude a significant difference at 5% level. Mostly negative, suggesting Method B may be better, but evidence inconclusive.

---

**Exercise 4.**
**Pooled vs Welch.** When to use each, and what assumption distinguishes them?

??? success "Solution to Exercise 4"
    **Pooled $t$-test:** assumes $\sigma_1 = \sigma_2$. Pools both samples to estimate the common $\sigma$. Df $= n_1 + n_2 - 2$.

    **Welch's $t$-test:** allows $\sigma_1 \ne \sigma_2$. Uses separate variances. Approximate df via Welch-Satterthwaite formula.

    **When to use pooled:** when variances are *known* to be equal (e.g., by experimental design). Modest efficiency gain.

    **When to use Welch:** default. Doesn't require equal variances; nearly as efficient as pooled when variances are actually equal.

    Modern recommendation: **always use Welch** unless equal-variance is structurally guaranteed. R's `t.test` defaults to Welch. The cost of "wrongly" using Welch when variances are equal is small; the cost of "wrongly" pooling when they aren't can be substantial.

---

**Exercise 5.**
**Paired vs independent.** $n = 50$ subjects measured pre/post treatment. Should the two-sample CI from this exercise apply?

??? success "Solution to Exercise 5"
    **No.** The pre/post measurements are *paired* — same subject twice. They are *not* independent. Using a two-sample CI would treat them as independent and ignore the within-subject correlation.

    **Correct approach:** compute differences $D_i = \mathrm{post}_i - \mathrm{pre}_i$ for each subject. One-sample CI on $D$ using $\bar D, s_D, n - 1$ degrees of freedom.

    **Why this matters:** if pre/post are positively correlated (typical), $\mathrm{Var}(D) < \mathrm{Var}(\mathrm{pre}) + \mathrm{Var}(\mathrm{post})$. Paired analysis has smaller SE, tighter CI, more power.

    The two-sample CI loses information about pairing and gives wider intervals than necessary. Always match the analysis to the design.

---

**Exercise 6.**
**Effect size.** In addition to the CI for $\mu_1 - \mu_2$, report **Cohen's $d$** = (effect)/(pooled SD). Interpret.

??? success "Solution to Exercise 6"
    For Exercise 3: pooled SD $= \sqrt{((29 \cdot 64) + (34 \cdot 100))/63} = \sqrt{(1856 + 3400)/63} = \sqrt{83.4} \approx 9.13$.

    Cohen's $d = (78 - 82)/9.13 \approx -0.44$.

    **Interpretation:**

    - $|d| = 0.2$: small effect.
    - $|d| = 0.5$: medium effect.
    - $|d| = 0.8$: large effect.

    A $d$ of $-0.44$ is a "medium-small" effect — Method B's mean is about 0.44 SDs above A's mean. The CI didn't quite reach significance, but the effect size is non-negligible.

    Always report both:

    - **Statistical significance** (CI excludes 0, $p < \alpha$).
    - **Practical significance** (effect size large enough to matter).

    Effect sizes are dimensionless and comparable across studies and disciplines — useful for meta-analysis. Significance tests are sample-size-dependent; effect sizes are not.
