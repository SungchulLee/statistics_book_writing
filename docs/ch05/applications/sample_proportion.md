# Sampling Distribution of Proportions

## Overview

The **sampling distribution of the sample proportion** $\hat{p}$ describes how the proportion of successes varies across repeated random samples from a binary population. It is the foundation for inference about population proportions — polls, quality control, clinical trials, and A/B tests all rely on it.

## Mathematical Definition

Let $X_1, \dots, X_n$ be i.i.d. $\text{Bernoulli}(p)$, where $X_i = 1$ (success) or $X_i = 0$ (failure). The sample proportion is:

$$
\hat{p} = \frac{1}{n}\sum_{i=1}^n X_i = \frac{\text{number of successes}}{n}
$$

## Properties

### Expected Value (Unbiasedness)

$$
E[\hat{p}] = p
$$

The sample proportion is an **unbiased estimator** of the population proportion.

### Variance and Standard Error

Since $\text{Var}(X_i) = p(1-p)$:

$$
\text{Var}(\hat{p}) = \frac{p(1-p)}{n}, \qquad
\text{SE}(\hat{p}) = \sqrt{\frac{p(1-p)}{n}}
$$

!!! note
    Unlike $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$, the standard error of $\hat{p}$ depends on the parameter $p$ itself. In practice, $p$ is unknown, so we substitute $\hat{p}$:

    $$
    \widehat{\text{SE}}(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
    $$

### Shape (Normal Approximation)

By the CLT, for sufficiently large $n$:

$$
\frac{\hat{p} - p}{\sqrt{p(1-p)/n}} \xrightarrow{d} N(0, 1)
$$

The **rule of thumb** for the normal approximation to be valid:

$$
np \geq 5 \quad \text{and} \quad n(1-p) \geq 5
$$

This ensures both success and failure counts are large enough for the bell-curve approximation.

## Example: Standard Error Computation

**Problem.** True proportion $p = 0.4$, sample size $n = 100$.

$$
\text{SE}(\hat{p}) = \sqrt{\frac{0.4 \times 0.6}{100}} = \sqrt{0.0024} \approx 0.049
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

## Simulation: Sampling Distribution of p-hat
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

## Exercises

**Exercise 1.**
Population: $p = 0.60$. Sample $n = 100$. Compute $P(\hat p > 0.65)$.

??? success "Solution to Exercise 1"
    $\mathrm{SE}(\hat p) = \sqrt{p(1-p)/n} = \sqrt{0.24/100} \approx 0.049$. $Z = (0.65 - 0.60)/0.049 \approx 1.02$.

    $P(\hat p > 0.65) = 1 - \Phi(1.02) \approx 0.154$. About 15.4%.

    Conditions: $np = 60 \ge 10$ and $n(1-p) = 40 \ge 10$, so normal approximation is valid.

---

**Exercise 2.**
**Small-sample issue.** Population: $p = 0.30$. Sample $n = 10$. Compute $P(\hat p > 0.35)$ exactly and via normal approximation. Why does the normal approximation work or fail here?

??? success "Solution to Exercise 2"
    Exact: $\hat p > 0.35$ ⟺ $X \ge 4$ where $X \sim \mathrm{Binomial}(10, 0.3)$.

    $P(X < 4) = P(X = 0,1,2,3) = 0.028 + 0.121 + 0.233 + 0.267 = 0.650$. So $P(X \ge 4) = 0.350$.

    Normal approx: $\mathrm{SE} = \sqrt{0.21/10} \approx 0.145$. $Z = 0.05/0.145 \approx 0.345$. $P(Z > 0.345) \approx 0.365$.

    Difference: exact 0.350 vs. normal 0.365 — about 1.5 percentage points off. Normal works reasonably (the binomial is not too skewed at $p = 0.3$) but the small $np = 3$ violates the usual rule of thumb ($np \ge 10$). For better accuracy, apply continuity correction or use exact binomial.

---

**Exercise 3.**
**Prove $\hat p$ is unbiased and find its SE.** Show $\mathbb{E}[\hat p] = p$ and $\mathrm{Var}(\hat p) = p(1-p)/n$.

??? success "Solution to Exercise 3"
    $\hat p = X/n$ where $X = \sum X_i$ with $X_i \sim \mathrm{Bernoulli}(p)$ i.i.d.

    $\mathbb{E}[\hat p] = \mathbb{E}[X]/n = np/n = p$. Unbiased.

    $\mathrm{Var}(\hat p) = \mathrm{Var}(X)/n^2 = np(1-p)/n^2 = p(1-p)/n$.

    $\mathrm{SE}(\hat p) = \sqrt{p(1-p)/n}$.

    $\square$

    The SE is maximized at $p = 1/2$ (worst case). At $p = 1/2$, $\mathrm{SE} = 1/(2\sqrt n)$.

---

**Exercise 4.**
**Sample-size planning.** What sample size is needed to estimate $p$ with margin of error $\pm 3$ percentage points at 95% confidence, when $p$ is unknown?

??? success "Solution to Exercise 4"
    Margin of error: $\mathrm{ME} = z_{0.975} \cdot \mathrm{SE} = 1.96 \sqrt{p(1-p)/n} \le 0.03$.

    Worst case at $p = 1/2$: $\mathrm{SE} = 1/(2\sqrt n)$, so $1.96/(2\sqrt n) \le 0.03 \Rightarrow \sqrt n \ge 1.96/0.06 \approx 32.67 \Rightarrow n \ge 1068$.

    This is the origin of the "$n \approx 1000$" rule for opinion polls: with $n = 1000$, the margin of error at 95% confidence is at most $\pm 3.1$ pp regardless of $p$.

    If $p$ is suspected to be far from 0.5 (say $p \approx 0.1$), then $p(1-p) = 0.09$ instead of 0.25, requiring only $n \approx 0.09 \cdot 1068/0.25 \approx 385$. Knowledge of approximate $p$ reduces required $n$.

---

**Exercise 5.**
**Wilson score interval.** Why is the **Wilson score** confidence interval preferred to the standard Wald CI $\hat p \pm z \sqrt{\hat p(1-\hat p)/n}$ for binomial proportions?

??? success "Solution to Exercise 5"
    **Wald CI problems:**

    - Asymmetric coverage especially near $p = 0$ or $p = 1$.
    - Can produce intervals extending below 0 or above 1 (e.g., $\hat p = 0.05, n = 50$: CI = $0.05 \pm 0.06 = (-0.01, 0.11)$).
    - Coverage probability oscillates wildly with $n$ — far from nominal $1 - \alpha$.

    **Wilson score interval:**

    $$
    p_{\mathrm{Wilson}} = \frac{\hat p + z^2/(2n) \pm z\sqrt{\hat p(1-\hat p)/n + z^2/(4n^2)}}{1 + z^2/n}
    $$

    Solves $|p - \hat p| \le z\sqrt{p(1-p)/n}$ for $p$, inverting the test rather than substituting $\hat p$ for $p$ in the SE.

    **Advantages:** stays in $[0, 1]$, much better coverage near boundaries, recommended by modern practice (R's `prop.test`, Python's `statsmodels.stats.proportion.proportion_confint(method="wilson")`).

---

**Exercise 6.**
**Difference of proportions.** Two independent samples: $\hat p_1$ from $n_1$, $\hat p_2$ from $n_2$. Derive the SE of $\hat p_1 - \hat p_2$.

??? success "Solution to Exercise 6"
    By independence: $\mathrm{Var}(\hat p_1 - \hat p_2) = \mathrm{Var}(\hat p_1) + \mathrm{Var}(\hat p_2) = p_1(1-p_1)/n_1 + p_2(1-p_2)/n_2$.

    $\mathrm{SE}(\hat p_1 - \hat p_2) = \sqrt{p_1(1-p_1)/n_1 + p_2(1-p_2)/n_2}$.

    For inference (test $H_0: p_1 = p_2 = p$), substitute the pooled estimator $\hat p_{\text{pool}} = (X_1 + X_2)/(n_1 + n_2)$ in the SE.

    For confidence intervals (estimating $p_1 - p_2$ without assuming equality), substitute $\hat p_1$ and $\hat p_2$ separately:

    $$
    \mathrm{CI}: (\hat p_1 - \hat p_2) \pm z\sqrt{\hat p_1(1-\hat p_1)/n_1 + \hat p_2(1-\hat p_2)/n_2}
    $$

    This is the basis of two-proportion $z$-tests and CIs in A/B testing and clinical trials.
