# Normal Distribution (Z)

## Overview

The **standard normal distribution** $Z \sim N(0, 1)$ is the most fundamental sampling distribution. It arises naturally whenever we standardize a normally distributed statistic, and — via the Central Limit Theorem — it serves as the large-sample approximation for a wide variety of estimators.

## Definition and Properties

A random variable $Z$ has the **standard normal distribution** if its PDF is:

$$
\varphi(z) = \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{z^2}{2}\right), \quad z \in \mathbb{R}
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
Z = \frac{X - \mu}{\sigma} \sim N(0, 1)
$$

More importantly, if $X_1, \dots, X_n$ are i.i.d. $N(\mu, \sigma^2)$, then the sample mean $\bar{X} \sim N(\mu, \sigma^2/n)$, and:

$$
Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim N(0, 1)
$$

This is an **exact** result when the population is normal and $\sigma$ is known.

### Central Limit Theorem (CLT)

For **any** population with finite variance $\sigma^2 < \infty$, the CLT guarantees:

$$
\frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0, 1) \quad \text{as } n \to \infty
$$

This makes $Z$ the default reference distribution for large-sample inference, even when the underlying population is non-normal.

## When to Use the Z Distribution

The standard normal is appropriate when:

1. **Population is normal and $\sigma$ is known**: exact $Z$-statistic.
2. **Large sample size** ($n \geq 30$ as a rough guideline): CLT-based approximation, regardless of population shape.
3. **Proportions with large $n$**: The sample proportion $\hat{p}$ is approximately normal when $np \geq 5$ and $n(1-p) \geq 5$.

When $\sigma$ is unknown and $n$ is small, the **Student's $t$ distribution** replaces $Z$.

## Common Z-Based Pivotal Quantities

### Sample Mean (Known sigma)

$$
Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim N(0, 1)
$$

**Confidence interval:**

$$
\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

### Sample Proportion (Large n)

$$
Z = \frac{\hat{p} - p}{\sqrt{p(1-p)/n}} \approx N(0, 1)
$$

**Confidence interval:**

$$
\hat{p} \pm z_{\alpha/2} \cdot \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

### Difference of Two Means (Known sigma_1, sigma_2)

$$
Z = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}} \sim N(0, 1)
$$

### Difference of Two Proportions (Large n_1, n_2)

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

## Exercises

**Exercise 1.**
Let $X_1, \ldots, X_n$ be i.i.d. $N(\mu, \sigma^2)$ with $\sigma$ known. Derive the distribution of $\bar{X}$ and the standardized statistic $Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}}$.

??? success "Solution to Exercise 1"
    Since each $X_i \sim N(\mu, \sigma^2)$ and the $X_i$ are independent, the sample mean $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ is a linear combination of independent normals. Therefore:

    $$
    \bar{X} \sim N\!\left(\mu, \frac{\sigma^2}{n}\right)
    $$

    Standardizing:

    $$
    Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0, 1)
    $$

    This follows because subtracting the mean and dividing by the standard deviation of a normal random variable always produces a standard normal.

---

**Exercise 2.**
A machine fills bottles with a mean of 500 mL and a known standard deviation of 5 mL (normally distributed). A sample of $n = 25$ bottles has a sample mean of 498 mL. Find the probability that the sample mean is 498 mL or less.

??? success "Solution to Exercise 2"
    Under the assumption $\mu = 500$ and $\sigma = 5$:

    $$
    Z = \frac{498 - 500}{5/\sqrt{25}} = \frac{-2}{1} = -2
    $$

    $$
    P(\bar{X} \leq 498) = P(Z \leq -2) = \mathcal{N}(-2) \approx 0.0228
    $$

    There is approximately a 2.28% chance of observing a sample mean of 498 mL or less if the true mean is 500 mL.

---

**Exercise 3.**
If $Z_1, Z_2, Z_3$ are independent standard normal variables, what is the distribution of $Z_1^2 + Z_2^2 + Z_3^2$? What are the mean and variance of this distribution?

??? success "Solution to Exercise 3"
    By definition, the sum of squares of $k$ independent standard normal variables follows a chi-square distribution with $k$ degrees of freedom:

    $$
    Z_1^2 + Z_2^2 + Z_3^2 \sim \chi^2_3
    $$

    The mean and variance of a $\chi^2_k$ distribution are $E[\chi^2_k] = k$ and $\text{Var}(\chi^2_k) = 2k$. Therefore:

    $$
    E[Z_1^2 + Z_2^2 + Z_3^2] = 3, \quad \text{Var}(Z_1^2 + Z_2^2 + Z_3^2) = 6
    $$

---

**Exercise 4.**
Explain why the normal distribution plays a central role in sampling theory even when the population is not normal. What theorem justifies this, and what are its limitations?

??? success "Solution to Exercise 4"
    The **Central Limit Theorem (CLT)** justifies the central role of the normal distribution. It states that for any population with finite mean $\mu$ and finite variance $\sigma^2$, the standardized sample mean $Z_n = \frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}}$ converges in distribution to $N(0,1)$ as $n \to \infty$, regardless of the shape of the original population.

    **Limitations:**

    - The CLT is an asymptotic result; for small $n$, the approximation may be poor, especially for highly skewed or heavy-tailed distributions.
    - The population must have a finite variance; for distributions with infinite variance (e.g., Cauchy), the CLT does not apply.
    - The Berry-Esseen theorem quantifies the rate of convergence: the approximation error is $O(1/\sqrt{n})$, and skewed distributions converge more slowly.
    - For inference about variance (not the mean), normality of the population is required for exact chi-square results; the CLT does not rescue variance-based inference in the same way.
