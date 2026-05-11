# Robust Variance Tests Comparison

## Overview

When comparing variances across groups, the choice of test matters greatly depending on the underlying data distribution. Classical tests like Bartlett's are optimal under normality but break down under non-normality. This page compares several tests for equality of variances -- Bartlett's, Levene's (mean and median centered), and Fligner--Killeen -- in terms of Type I error control and power across different distributional settings.

## Tests Compared

The four main tests for homogeneity of variances are:

| Test | Assumption | Robustness |
|---|---|---|
| Bartlett | Normality required | Not robust to non-normality |
| Levene (mean) | None required | Moderately robust |
| Brown--Forsythe (Levene, median) | None required | Robust to skewness |
| Fligner--Killeen | None required | Highly robust (nonparametric) |

Under normality and the null hypothesis $H_0: \sigma_1^2 = \cdots = \sigma_k^2$, all four tests should reject at approximately rate $\alpha$. Under non-normality, Bartlett's test inflates its Type I error, while the others maintain control.

## Mathematical Framework

All four tests can be viewed through a common lens. Define transformed observations $Z_{ij}$ based on deviations from a group center, then test whether the group means of $Z_{ij}$ differ:

$$
Z_{ij} = |X_{ij} - c_i|,
$$

where $c_i$ is the group mean (Levene), median (Brown--Forsythe), or a rank-based center (Fligner--Killeen). Bartlett's test instead operates directly on the log-likelihood ratio.

## Code

The following simulation compares the false-positive rates of all four tests under a skewed (lognormal) distribution where all group variances are equal:

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n_sims, n, alpha = 2000, 30, 0.05
results = {"Bartlett": 0, "Levene (mean)": 0,
           "Brown-Forsythe": 0, "Fligner-Killeen": 0}

for _ in range(n_sims):
    g1 = rng.lognormal(0, 1, n)
    g2 = rng.lognormal(0, 1, n)
    g3 = rng.lognormal(0, 1, n)

    _, p = stats.bartlett(g1, g2, g3)
    if p < alpha:
        results["Bartlett"] += 1

    _, p = stats.levene(g1, g2, g3, center='mean')
    if p < alpha:
        results["Levene (mean)"] += 1

    _, p = stats.levene(g1, g2, g3, center='median')
    if p < alpha:
        results["Brown-Forsythe"] += 1

    _, p = stats.fligner(g1, g2, g3)
    if p < alpha:
        results["Fligner-Killeen"] += 1

for name, count in results.items():
    print(f"{name:20s}: false-positive rate = {count/n_sims:.3f}")
```

A power comparison under normality where the variances truly differ:

```python
results_power = {"Bartlett": 0, "Levene (mean)": 0,
                 "Brown-Forsythe": 0, "Fligner-Killeen": 0}

for _ in range(n_sims):
    g1 = rng.normal(0, 1.0, n)
    g2 = rng.normal(0, 1.5, n)
    g3 = rng.normal(0, 2.0, n)

    _, p = stats.bartlett(g1, g2, g3)
    if p < alpha:
        results_power["Bartlett"] += 1

    _, p = stats.levene(g1, g2, g3, center='mean')
    if p < alpha:
        results_power["Levene (mean)"] += 1

    _, p = stats.levene(g1, g2, g3, center='median')
    if p < alpha:
        results_power["Brown-Forsythe"] += 1

    _, p = stats.fligner(g1, g2, g3)
    if p < alpha:
        results_power["Fligner-Killeen"] += 1

for name, count in results_power.items():
    print(f"{name:20s}: power = {count/n_sims:.3f}")
```

## Interpretation

- **Under normality with equal variances**: all tests reject at approximately $\alpha = 0.05$.
- **Under non-normality (e.g., lognormal) with equal variances**: Bartlett's false-positive rate is heavily inflated (often 0.20+), Levene (mean) is moderately inflated, while Brown--Forsythe and Fligner--Killeen stay close to 0.05.
- **Under normality with unequal variances (power)**: Bartlett has the highest power (it is UMP under normality), followed by Levene (mean), Brown--Forsythe, and Fligner--Killeen.
- The trade-off is clear: Bartlett wins on power when normality holds but fails catastrophically otherwise. For general use, Brown--Forsythe or Fligner--Killeen offer the best balance of robustness and power.

## Exercises

**Exercise 1.** Run the false-positive simulation above with $n = 100$ instead of $n = 30$. Does increasing the sample size help Bartlett's test control its Type I error under lognormal data? Explain.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n_sims, n, alpha = 5000, 100, 0.05
    rej = 0

    for _ in range(n_sims):
        g1 = rng.lognormal(0, 1, n)
        g2 = rng.lognormal(0, 1, n)
        g3 = rng.lognormal(0, 1, n)
        _, p = stats.bartlett(g1, g2, g3)
        if p < alpha:
            rej += 1

    print(f"Bartlett false-positive rate (n=100): {rej/n_sims:.3f}")
    ```

    Increasing $n$ does **not** help Bartlett control its Type I error. The inflation persists (and may even worsen) because larger samples give the test more power to detect the non-normality-induced variability in $S^2$. The issue is a model assumption violation, not a finite-sample artifact.

---

**Exercise 2.** Add the $F$-test (two-group version) to the comparison for $k = 2$ groups. How does its Type I error rate compare to Bartlett's under lognormal data?

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n_sims, n, alpha = 5000, 30, 0.05
    rej_f, rej_b = 0, 0

    for _ in range(n_sims):
        g1 = rng.lognormal(0, 1, n)
        g2 = rng.lognormal(0, 1, n)
        F = np.var(g1, ddof=1) / np.var(g2, ddof=1)
        p_f = 2 * min(stats.f(n-1, n-1).cdf(F), stats.f(n-1, n-1).sf(F))
        _, p_b = stats.bartlett(g1, g2)
        if p_f < alpha:
            rej_f += 1
        if p_b < alpha:
            rej_b += 1

    print(f"F-test:   {rej_f/n_sims:.3f}")
    print(f"Bartlett: {rej_b/n_sims:.3f}")
    ```

    Both will have inflated rejection rates (well above 0.05) since they both assume normality. For $k = 2$, Bartlett's test and the F-test are closely related (see the Bartlett page), so their rejection rates will be similar.

---

**Exercise 3.** Design a simulation that estimates the power of each test when the data come from a $t(5)$ distribution with group standard deviations $\sigma_1 = 1$, $\sigma_2 = 1.5$, $\sigma_3 = 2$. Discuss which test performs best overall.

??? success "Solution to Exercise 3"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n_sims, n, alpha = 5000, 30, 0.05
    power = {"Bartlett": 0, "Levene": 0, "BF": 0, "FK": 0}

    for _ in range(n_sims):
        g1 = stats.t(df=5).rvs(n, random_state=rng) * 1.0
        g2 = stats.t(df=5).rvs(n, random_state=rng) * 1.5
        g3 = stats.t(df=5).rvs(n, random_state=rng) * 2.0
        _, p = stats.bartlett(g1, g2, g3)
        if p < alpha:
            power["Bartlett"] += 1
        _, p = stats.levene(g1, g2, g3, center='mean')
        if p < alpha:
            power["Levene"] += 1
        _, p = stats.levene(g1, g2, g3, center='median')
        if p < alpha:
            power["BF"] += 1
        _, p = stats.fligner(g1, g2, g3)
        if p < alpha:
            power["FK"] += 1

    for name, count in power.items():
        print(f"{name:10s}: {count/n_sims:.3f}")
    ```

    Bartlett will appear to have high "power," but this is misleading because its Type I error is also inflated. When properly calibrated, Levene (mean) typically offers the best power under heavy tails, with Brown--Forsythe close behind. Fligner--Killeen sacrifices some power for maximum robustness.

---

**Exercise 4.** Explain mathematically why the Brown--Forsythe test is more robust to skewness than the mean-centered Levene test.

??? success "Solution to Exercise 4"

    For a skewed distribution, the mean is pulled toward the tail, so the absolute deviations $Z_{ij} = |X_{ij} - \bar{X}_i|$ inherit the asymmetry: observations on the long-tail side produce systematically larger $Z_{ij}$ values. This inflates the within-group variability of $Z_{ij}$, which affects the ANOVA F-statistic.

    The median, being the 50th percentile, is not affected by the magnitude of extreme values. When centering on the median, the $Z_{ij}$ are more symmetrically distributed even when the original $X_{ij}$ are skewed. Formally, if $X$ has distribution $F$, then $E[|X - \text{median}|] \le E[|X - \mu|]$ (the median minimizes expected absolute deviation), so the median-centered deviations have smaller variance and are less sensitive to the tails. This translates to better calibration of the test statistic under $H_0$. $\square$

---

**Exercise 5.** Suppose you are analyzing clinical trial data where the outcome is known to be right-skewed (e.g., hospital length of stay). You need to check equal variances across three treatment arms before deciding on a downstream analysis. Which test would you recommend and why? Discuss at least two alternatives.

??? success "Solution to Exercise 5"

    For right-skewed clinical data such as length of stay, the recommended test is the **Brown--Forsythe test** (`scipy.stats.levene` with `center='median'`). It controls the Type I error rate well under skewness while retaining reasonable power to detect true variance differences.

    **Alternative 1: Fligner--Killeen test.** This is a nonparametric test based on ranks of absolute deviations from the median. It is the most robust option and appropriate when the data may contain outliers. The trade-off is slightly lower power under normality.

    **Alternative 2: Log-transform then Bartlett.** If the skewness is due to a log-normal-like mechanism, applying a log transformation may symmetrize the data, allowing the more powerful Bartlett test to be used validly. The risk is that the transformation may not fully normalize the data, and inference is conducted on the log scale.

    Bartlett's test and the F-test should be avoided directly on skewed data, as they will produce unreliable $p$-values.
