# Testing Spearman's rho

Spearman's rank correlation $r_s$ measures the strength of a monotonic association between two variables. To determine whether the observed $r_s$ is statistically significant, we test the null hypothesis that the population Spearman correlation is zero. This section covers the hypothesis test, its null distribution, and its connection to the Pearson t-test on ranks.

---

## Hypotheses

The standard test is:

$$
H_0\!: \rho_s = 0 \quad \text{vs} \quad H_1\!: \rho_s \neq 0
$$

where $\rho_s$ is the population Spearman rank correlation. The null hypothesis states that there is no monotonic association between $X$ and $Y$.

One-sided alternatives ($H_1\!: \rho_s > 0$ or $H_1\!: \rho_s < 0$) are used when the direction of the monotonic relationship is specified in advance.

---

## Test Statistic: t-Approximation

For moderate to large samples, the test statistic is

$$
t = r_s \sqrt{\frac{n - 2}{1 - r_s^2}}
$$

Under $H_0$ (no monotonic association), this statistic follows approximately a **t-distribution with $n - 2$ degrees of freedom**. This is the same formula used for testing Pearson's $r$, applied to the rank correlation.

The approximation is the same as performing a Pearson correlation t-test on the ranks of the data. Since Spearman's $r_s$ equals Pearson's $r$ computed on the ranks, the test procedures are algebraically equivalent.

---

## Exact Distribution for Small Samples

For small sample sizes (typically $n \le 20$), the exact null distribution of $r_s$ can be computed by considering all $n!$ possible permutations of the ranks. Under $H_0$, all permutations are equally likely, so the distribution of $r_s$ can be tabulated exactly.

Critical values from the exact distribution are available in tables for small $n$. Most statistical software uses the exact distribution for small samples and switches to the t-approximation for larger samples.

---

## Decision Rule

For a two-sided test at significance level $\alpha$:

- Using the t-approximation: reject $H_0$ if $|t| > t_{\alpha/2, \, n-2}$
- Using exact tables: reject $H_0$ if $|r_s|$ exceeds the critical value for the given $n$ and $\alpha$

The p-value is computed as

$$
p = 2 \cdot P(T_{n-2} > |t|)
$$

for the t-approximation, or from the exact permutation distribution for small samples.

---

## Example

A biologist ranks 12 animals by body mass ($X$) and metabolic rate ($Y$) and computes $r_s = 0.72$.

$$
t = 0.72 \sqrt{\frac{12 - 2}{1 - 0.72^2}} = 0.72 \sqrt{\frac{10}{0.4816}} = 0.72 \times 4.557 = 3.281
$$

With $n - 2 = 10$ degrees of freedom, the critical value for a two-sided test at $\alpha = 0.05$ is $t_{0.025, 10} = 2.228$. Since $|t| = 3.281 > 2.228$, we reject $H_0$ and conclude that there is a statistically significant monotonic association between body mass and metabolic rate.

---

## Assumptions

The test for Spearman's $r_s$ requires:

1. **Independence**: the observations are independent pairs.
2. **Ordinal or continuous data**: both variables must be at least ordinal (ranks must be meaningful).
3. **No specific distributional assumption**: unlike the Pearson t-test, the Spearman test does not assume normality. It is a **distribution-free** (nonparametric) test.

The distribution-free nature of Spearman's test makes it appropriate when:

- The data are non-normal or skewed.
- Outliers are present.
- The relationship is monotonic but not linear.

---

## Spearman vs Pearson Test: When to Choose

| Feature | Pearson t-test | Spearman t-test |
|:---|:---|:---|
| Detects | Linear association | Monotonic association |
| Distributional assumption | Bivariate normality | None (distribution-free) |
| Robustness to outliers | Low | High |
| Power under normality + linearity | Higher | Slightly lower |
| Power under non-normality | Lower | Higher |

If the data are bivariate normal and the relationship is linear, the Pearson test has slightly more power. In all other situations, the Spearman test is at least as powerful and often more powerful.

---

## Computation in Python

```python
import numpy as np
from scipy import stats

# Sample data
x = np.array([3, 1, 6, 4, 8, 2, 7, 5, 10, 9, 11, 12])
y = np.array([5, 2, 8, 3, 11, 1, 9, 6, 12, 7, 10, 4])

# Spearman test
r_s, p_value = stats.spearmanr(x, y)
print(f"Spearman r_s = {r_s:.4f}")
print(f"p-value      = {p_value:.4f}")

# Manual t-statistic for verification
n = len(x)
t_stat = r_s * np.sqrt((n - 2) / (1 - r_s**2))
p_manual = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
print(f"t-statistic  = {t_stat:.4f}")
print(f"Manual p     = {p_manual:.4f}")
```

The `scipy.stats.spearmanr` function uses the t-approximation for larger samples and can handle tied values.

---

## Summary

The hypothesis test for Spearman's $r_s$ uses the same t-statistic formula as the Pearson test, applied to the ranks of the data. The test is distribution-free and does not require normality, making it robust to outliers and applicable to ordinal data. For small samples, exact permutation-based p-values are available. The Spearman test is preferred over the Pearson test when the data are non-normal, contain outliers, or exhibit a monotonic but nonlinear relationship.
