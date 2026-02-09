# Normal Distribution (Z)

## Overview

The **standard normal distribution** $Z \sim N(0, 1)$ is the most fundamental sampling distribution. It arises naturally whenever we standardize a normally distributed statistic, and — via the Central Limit Theorem — it serves as the large-sample approximation for a wide variety of estimators.

## Definition and Properties

A random variable $Z$ has the **standard normal distribution** if its PDF is:

$$
\varphi(z) = \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{z^2}{2}\right), \quad z \in \mathbb{R}.
$$

Key properties:

| Property | Value |
|----------|-------|
| Mean | $E[Z] = 0$ |
| Variance | $\text{Var}(Z) = 1$ |
| Symmetry | $\varphi(z) = \varphi(-z)$ |
| MGF | $M_Z(t) = \exp(t^2/2)$ |

## Role in Sampling Theory

### Standardization

If $X \sim N(\mu, \sigma^2)$, then:

$$
Z = \frac{X - \mu}{\sigma} \sim N(0, 1).
$$

More importantly, if $X_1, \dots, X_n$ are i.i.d. $N(\mu, \sigma^2)$, then the sample mean $\bar{X} \sim N(\mu, \sigma^2/n)$, and:

$$
Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim N(0, 1).
$$

This is an **exact** result when the population is normal and $\sigma$ is known.

### Central Limit Theorem (CLT)

For **any** population with finite variance $\sigma^2 < \infty$, the CLT guarantees:

$$
\frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0, 1) \quad \text{as } n \to \infty.
$$

This makes $Z$ the default reference distribution for large-sample inference, even when the underlying population is non-normal.

## When to Use the Z Distribution

The standard normal is appropriate when:

1. **Population is normal and $\sigma$ is known**: exact $Z$-statistic.
2. **Large sample size** ($n \geq 30$ as a rough guideline): CLT-based approximation, regardless of population shape.
3. **Proportions with large $n$**: The sample proportion $\hat{p}$ is approximately normal when $np \geq 5$ and $n(1-p) \geq 5$.

When $\sigma$ is unknown and $n$ is small, the **Student's $t$ distribution** replaces $Z$.

## Common Z-Based Pivotal Quantities

### Sample Mean (Known $\sigma$)

$$
Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim N(0, 1)
$$

**Confidence interval:**

$$
\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

### Sample Proportion (Large $n$)

$$
Z = \frac{\hat{p} - p}{\sqrt{p(1-p)/n}} \approx N(0, 1)
$$

**Confidence interval:**

$$
\hat{p} \pm z_{\alpha/2} \cdot \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

### Difference of Two Means (Known $\sigma_1, \sigma_2$)

$$
Z = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}} \sim N(0, 1)
$$

### Difference of Two Proportions (Large $n_1, n_2$)

$$
Z = \frac{(\hat{p}_1 - \hat{p}_2) - (p_1 - p_2)}{\sqrt{\frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}}} \approx N(0, 1)
$$

## Critical Values

Common critical values $z_{\alpha/2}$ for two-sided intervals:

| Confidence Level | $\alpha$ | $z_{\alpha/2}$ |
|-----------------|----------|----------------|
| 90% | 0.10 | 1.645 |
| 95% | 0.05 | 1.960 |
| 99% | 0.01 | 2.576 |

```python
from scipy import stats

for alpha in [0.10, 0.05, 0.01]:
    z = stats.norm.ppf(1 - alpha / 2)
    print(f"Confidence {1-alpha:.0%}: z* = {z:.3f}")
```

## Relationship to Other Distributions

The standard normal is the building block for other sampling distributions:

- **Chi-square**: If $Z_1, \dots, Z_k$ are i.i.d. $N(0,1)$, then $\sum Z_i^2 \sim \chi^2_k$.
- **Student's $t$**: $T = Z / \sqrt{V/k}$ where $V \sim \chi^2_k$ independent of $Z$.
- **$F$-distribution**: $F = (U/m) / (V/n)$ where $U \sim \chi^2_m$ and $V \sim \chi^2_n$ are independent.

## Summary

The standard normal distribution is the cornerstone of sampling theory. It provides exact results for normal populations with known variance, and approximate results for large samples from any finite-variance population via the CLT. Its simplicity and universality make it the first distribution to consider in any inferential problem.
