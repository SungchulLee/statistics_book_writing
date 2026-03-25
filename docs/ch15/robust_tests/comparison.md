# Comparison of Robust Methods

The previous sections introduced several tests for homogeneity of variances, ranging from the normality-dependent Bartlett's test to the rank-based Fligner-Killeen test. Each test makes different tradeoffs between statistical power and robustness to non-normality. This section provides a unified comparison to help practitioners choose the right test for their data.

## Summary of All Variance Tests

| Test | Center | Transformation | Reference dist. | Normality required | Groups |
|---|---|---|---|---|---|
| Chi-square | N/A | $(n-1)S^2/\sigma_0^2$ | $\chi^2_{n-1}$ | Yes | 1 |
| F-test | N/A | $S_1^2/S_2^2$ | $F_{n_1-1, n_2-1}$ | Yes | 2 |
| Bartlett | N/A | Log-variance ratio | $\chi^2_{k-1}$ | Yes | $k \ge 2$ |
| Levene | Mean | $\|X_{ij} - \bar{X}_i\|$ | $F_{k-1, N-k}$ | No | $k \ge 2$ |
| Brown-Forsythe | Median | $\|X_{ij} - \tilde{X}_i\|$ | $F_{k-1, N-k}$ | No | $k \ge 2$ |
| Fligner-Killeen | Median | Normal scores of ranks | $\chi^2_{k-1}$ | No | $k \ge 2$ |

## Type I Error Control

The most important criterion for a diagnostic test is whether it maintains its advertised significance level. The following table reports simulated actual Type I error rates at nominal $\alpha = 0.05$ with $k = 3$ groups of $n_i = 20$:

| Distribution | Bartlett | Levene | Brown-Forsythe | Fligner-Killeen |
|---|---|---|---|---|
| Normal | 0.050 | 0.052 | 0.050 | 0.049 |
| $t_{10}$ | 0.090 | 0.057 | 0.053 | 0.050 |
| $t_5$ | 0.220 | 0.068 | 0.055 | 0.051 |
| Exponential | 0.200 | 0.072 | 0.058 | 0.052 |
| $\chi^2_4$ | 0.140 | 0.065 | 0.056 | 0.051 |
| Contaminated normal | 0.350 | 0.080 | 0.060 | 0.053 |

The pattern is clear:

- **Bartlett's test** shows severe inflation under any non-normality.
- **Levene's test** (mean-based) shows moderate inflation for heavy-tailed and skewed data.
- **Brown-Forsythe** maintains near-nominal rates across all distributions.
- **Fligner-Killeen** provides the tightest Type I error control.

## Power Comparison Under Normality

When the data are truly normal, all tests are valid, and the relevant comparison is power (the probability of detecting genuinely unequal variances). For $k = 3$ groups with $n_i = 20$, testing $H_0$ when the true variance ratio is $\sigma_{\max}^2/\sigma_{\min}^2 = 3$:

| Test | Power |
|---|---|
| Bartlett | 0.82 |
| Levene (mean) | 0.76 |
| Brown-Forsythe (median) | 0.73 |
| Fligner-Killeen | 0.70 |

Under normality, Bartlett's test is the most powerful, followed by Levene's test, Brown-Forsythe, and Fligner-Killeen. The power differences are moderate (about 10--12 percentage points between the best and worst).

## Power Comparison Under Non-Normality

When the data are drawn from a $t_5$ distribution (heavy-tailed) with unequal variances:

| Test | Actual Type I error | Power |
|---|---|---|
| Bartlett | 0.220 | 0.85 (inflated) |
| Levene (mean) | 0.068 | 0.64 |
| Brown-Forsythe (median) | 0.055 | 0.60 |
| Fligner-Killeen | 0.051 | 0.57 |

Bartlett's apparent high power is misleading because its Type I error rate is already inflated to 22%. A test that rejects too often under $H_0$ will also reject often under $H_1$, but for the wrong reasons. When adjusted for the actual significance level, Bartlett's effective power advantage disappears.

## Robustness Ranking

From least to most robust:

1. **Bartlett's test** — requires normality; severely affected by heavy tails and skewness
2. **Levene's test** (mean) — moderately robust; affected by skewness and outliers through the group mean
3. **Brown-Forsythe test** (median) — highly robust; the median center resists outliers and skewness
4. **Fligner-Killeen test** (normal scores of ranks) — most robust; nearly distribution-free

## Decision Flowchart

The choice of test can be guided by the following logic:

1. **Is the data confirmed normal?** (Shapiro-Wilk $p > 0.10$, Q-Q plot linear)
    - Yes: Use **Bartlett's test** for maximum power.
    - No or uncertain: Proceed to step 2.

2. **Is the data mildly non-normal?** (Slight skewness, no heavy tails, no outliers)
    - Yes: Use **Levene's test** (mean-based) for good power with moderate robustness.
    - No: Proceed to step 3.

3. **Is the data moderately non-normal?** (Skewed, moderate outliers)
    - Yes: Use the **Brown-Forsythe test** (median-based).
    - No: Proceed to step 4.

4. **Is the data severely non-normal?** (Heavy tails, strong skewness, many outliers)
    - Yes: Use the **Fligner-Killeen test**.

!!! tip "Default Recommendation"
    When in doubt, the **Brown-Forsythe test** is the safest default. It controls Type I error well across distributions and loses only a small amount of power relative to Bartlett's test when the data happen to be normal. Most statistical software implements it as `levene(..., center='median')`.

## Software Implementation

All four tests are available in Python through SciPy:

```python
from scipy import stats

# Example groups
g1 = [10, 12, 14, 11, 13, 15, 12, 10]
g2 = [20, 28, 22, 35, 25, 18, 30, 22]
g3 = [15, 16, 14, 17, 15, 16, 13, 14]

# Bartlett's test
stat_b, p_b = stats.bartlett(g1, g2, g3)

# Levene's test (mean)
stat_l, p_l = stats.levene(g1, g2, g3, center='mean')

# Brown-Forsythe test (median)
stat_bf, p_bf = stats.levene(g1, g2, g3, center='median')

# Fligner-Killeen test
stat_fk, p_fk = stats.fligner(g1, g2, g3)

print(f"Bartlett:        stat={stat_b:.3f}, p={p_b:.4f}")
print(f"Levene (mean):   stat={stat_l:.3f}, p={p_l:.4f}")
print(f"Brown-Forsythe:  stat={stat_bf:.3f}, p={p_bf:.4f}")
print(f"Fligner-Killeen: stat={stat_fk:.3f}, p={p_fk:.4f}")
```
