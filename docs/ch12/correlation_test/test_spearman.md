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

## Exercises

**Exercise 1.**
Test whether Spearman's $r_s = 0.55$ from a sample of $n = 20$ is significantly different from zero at $\alpha = 0.05$.

??? success "Solution to Exercise 1"
    The test statistic is:

    $$
    t = r_s \sqrt{\frac{n-2}{1-r_s^2}} = 0.55\sqrt{\frac{18}{1 - 0.3025}} = 0.55\sqrt{\frac{18}{0.6975}} = 0.55\sqrt{25.806} = 0.55 \times 5.080 = 2.794
    $$

    With $df = n - 2 = 18$, the critical value $t_{0.025, 18} = 2.101$. Since $|t| = 2.794 > 2.101$, we reject $H_0$ at $\alpha = 0.05$. There is significant evidence of a monotonic association.

---

**Exercise 2.**
Explain why the t-distribution approximation for Spearman's test is more accurate for large $n$. What is the exact test for small $n$?

??? success "Solution to Exercise 2"
    The test statistic $t = r_s\sqrt{(n-2)/(1-r_s^2)}$ has an approximate $t_{n-2}$ distribution under $H_0$. This approximation works well for $n \geq 10$ and becomes exact in the limit.

    For small $n$, the exact distribution of $r_s$ under $H_0$ is discrete (because ranks can only take integer values) and is computed by enumerating all $n!$ permutations of one rank vector. Statistical software (e.g., `scipy.stats.spearmanr`) computes exact p-values for small $n$ and switches to the t-approximation for larger samples.

    The exact test is a permutation test: it compares the observed $r_s$ to the distribution of $r_s$ values obtained from all possible permutations of the $Y$-ranks, assuming independence.

---

**Exercise 3.**
A dataset of 50 countries shows Spearman's $r_s = 0.72$ between GDP per capita rank and life expectancy rank. Construct an approximate 95% confidence interval for the population $\rho_s$.

??? success "Solution to Exercise 3"
    Apply Fisher's z-transformation (which can also be used for Spearman's $r_s$ as an approximation):

    $$
    z_r = \frac{1}{2}\ln\frac{1 + 0.72}{1 - 0.72} = \frac{1}{2}\ln(6.143) = \frac{1}{2}(1.815) = 0.9076
    $$

    The standard error is approximately $1/\sqrt{n-3} = 1/\sqrt{47} = 0.1459$.

    The 95% CI for $z_\rho$ is $0.9076 \pm 1.96 \times 0.1459 = (0.6216, 1.1936)$.

    Back-transforming: $r = \frac{e^{2z}-1}{e^{2z}+1}$:

    - Lower: $\tanh(0.6216) = 0.553$
    - Upper: $\tanh(1.1936) = 0.832$

    The 95% CI for $\rho_s$ is approximately $(0.55, 0.83)$.

---

**Exercise 4.**
When is Spearman's test preferred over Pearson's test of correlation? List three scenarios.

??? success "Solution to Exercise 4"
    Spearman's test is preferred when:

    1. **The relationship is monotonic but nonlinear:** Pearson's $r$ measures only linear association and may underestimate the strength of a curved but monotonic relationship. Spearman's $r_s$ captures any monotonic pattern.

    2. **The data contain outliers:** Spearman's $r_s$ is based on ranks and is therefore robust to extreme values. A single outlier can dramatically change Pearson's $r$ but barely affects $r_s$.

    3. **The data are ordinal:** When measurements are on an ordinal scale (e.g., Likert ratings, rankings), the numerical values have no meaningful interval interpretation. Spearman's $r_s$ is appropriate because it uses only rank information, while Pearson's $r$ assumes interval-scale data.

    Additionally, Spearman's test does not require the bivariate normality assumption needed for exact inference with Pearson's $r$.
