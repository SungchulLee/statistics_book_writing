# Sampling Distribution of Proportions

## Overview

The **sampling distribution of the sample proportion** $\hat{p}$ describes how the proportion of successes varies across repeated random samples from a binary population. It is the foundation for inference about population proportions — polls, quality control, clinical trials, and A/B tests all rely on it.

## Mathematical Definition

Let $X_1, \dots, X_n$ be i.i.d. $\text{Bernoulli}(p)$, where $X_i = 1$ (success) or $X_i = 0$ (failure). The sample proportion is:

$$
\hat{p} = \frac{1}{n}\sum_{i=1}^n X_i = \frac{\text{number of successes}}{n}.
$$

## Properties

### Expected Value (Unbiasedness)

$$
E[\hat{p}] = p.
$$

The sample proportion is an **unbiased estimator** of the population proportion.

### Variance and Standard Error

Since $\text{Var}(X_i) = p(1-p)$:

$$
\text{Var}(\hat{p}) = \frac{p(1-p)}{n}, \qquad
\text{SE}(\hat{p}) = \sqrt{\frac{p(1-p)}{n}}.
$$

!!! note
    Unlike $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$, the standard error of $\hat{p}$ depends on the parameter $p$ itself. In practice, $p$ is unknown, so we substitute $\hat{p}$:

    $$
    \widehat{\text{SE}}(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}.
    $$

### Shape (Normal Approximation)

By the CLT, for sufficiently large $n$:

$$
\frac{\hat{p} - p}{\sqrt{p(1-p)/n}} \xrightarrow{d} N(0, 1).
$$

The **rule of thumb** for the normal approximation to be valid:

$$
np \geq 5 \quad \text{and} \quad n(1-p) \geq 5.
$$

This ensures both success and failure counts are large enough for the bell-curve approximation.

## Example: Standard Error Computation

**Problem.** True proportion $p = 0.4$, sample size $n = 100$.

$$
\text{SE}(\hat{p}) = \sqrt{\frac{0.4 \times 0.6}{100}} = \sqrt{0.0024} \approx 0.049.
$$

Across repeated samples of size 100, $\hat{p}$ will typically vary about 0.049 around the true $p = 0.4$.

## Worked Examples

### Example 1: Brand Preference

**Problem.** In a population, 60% prefer brand A. For $n = 100$, find $P(\hat{p} > 0.65)$.

**Solution.**

$$
\text{SE} = \sqrt{\frac{0.60 \times 0.40}{100}} \approx 0.049
$$

$$
Z = \frac{0.65 - 0.60}{0.049} \approx 1.02
$$

$$
P(\hat{p} > 0.65) = P(Z > 1.02) \approx 0.154
$$

```python
from scipy import stats
print(f"P(p_hat > 0.65) = {stats.norm.sf(1.02):.4f}")
```

### Example 2: Small Sample — Exact vs Approximate

**Problem.** In a town, 30% prefer public transport. For $n = 10$, find $P(\hat{p} > 0.35)$.

**Exact Binomial.** Since $\hat{p} > 0.35$ means $X \geq 4$ (where $X \sim \text{Binomial}(10, 0.3)$):

$$
P(X \geq 4) = 1 - P(X \leq 3)
$$

$$
P(X = 0) = 0.0282, \quad P(X = 1) = 0.1211, \quad P(X = 2) = 0.2335, \quad P(X = 3) = 0.2668
$$

$$
P(X \geq 4) = 1 - 0.6496 = 0.3504
$$

**Normal Approximation.** Check conditions: $np = 3 < 5$ — the normal approximation is questionable.

$$
\text{SE} = \sqrt{\frac{0.3 \times 0.7}{10}} \approx 0.1449, \qquad
Z = \frac{0.35 - 0.30}{0.1449} \approx 0.345
$$

$$
P(\hat{p} > 0.35) \approx P(Z > 0.345) \approx 0.365
$$

**Comparison:**

| Method | Result |
|--------|--------|
| Exact binomial | 0.3504 |
| Normal approximation | 0.3650 |

The approximation is reasonably close despite the small sample, but the exact binomial is preferred when $np < 5$.

```python
from scipy import stats

# Exact
exact = 1 - stats.binom(n=10, p=0.3).cdf(3)
print(f"Exact: {exact:.4f}")

# Normal approximation
approx = stats.norm.sf(0.345)
print(f"Normal approx: {approx:.4f}")
```

## Difference of Two Proportions

For independent samples from two populations with proportions $p_1$ and $p_2$:

$$
Z = \frac{(\hat{p}_1 - \hat{p}_2) - (p_1 - p_2)}{\sqrt{\frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}}} \approx N(0, 1)
$$

**Confidence interval:**

$$
(\hat{p}_1 - \hat{p}_2) \pm z_{\alpha/2} \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}
$$

## Simulation: Sampling Distribution of $\hat{p}$

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(1)

population = stats.binom(n=1, p=0.4).rvs(100_000)
sample_size = 1_000
n_samples = 10_000

sample_proportions = [
    np.mean(np.random.choice(population, size=sample_size, replace=False))
    for _ in range(n_samples)
]

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))

ax0.hist(population, bins=3, density=True, alpha=0.5)
ax0.set_title('Population Distribution (Bernoulli, p = 0.4)', fontsize=16)

ax1.hist(sample_proportions, bins=50, density=True, alpha=0.5)
ax1.set_title(rf'Sampling Distribution of $\hat{{p}}$ (n = {sample_size})', fontsize=16)

for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

## Graduate-Level Notes

- For small samples or extreme proportions ($p$ near 0 or 1), the **binomial distribution** should be used directly.
- The **Wilson interval** is generally preferred over the Wald interval ($\hat{p} \pm z^* \cdot \widehat{\text{SE}}$) because it has better coverage properties, especially for small $n$ or extreme $p$.
- The **Agresti–Coull interval** adds 2 pseudo-successes and 2 pseudo-failures before computing the Wald interval, providing a simple fix with improved coverage.

## Summary

| Property | Result |
|----------|--------|
| $E[\hat{p}]$ | $p$ (unbiased) |
| $\text{Var}(\hat{p})$ | $p(1-p)/n$ |
| $\text{SE}(\hat{p})$ | $\sqrt{p(1-p)/n}$ |
| Normal approx. valid when | $np \geq 5$ and $n(1-p) \geq 5$ |
| Key difference from $\bar{X}$ | SE depends on the parameter itself |
| For small $n$ | Use exact binomial, not normal approximation |
