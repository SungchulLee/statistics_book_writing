# Sampling Distribution of the Mean

## Overview

The **sampling distribution of the sample mean** $\bar{X}$ describes how $\bar{X}$ varies across repeated samples of size $n$ from a population. This is the single most important sampling distribution in statistics — it underpins confidence intervals for $\mu$, $t$-tests, and much of applied statistics.

## Mathematical Definition

Let $X_1, X_2, \dots, X_n$ be i.i.d. from a population with mean $\mu$ and variance $\sigma^2 < \infty$. The sample mean is:

$$
\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i.
$$

## Properties

### Expected Value (Unbiasedness)

$$
E[\bar{X}] = \mu.
$$

The sample mean is an **unbiased estimator** of the population mean: on average, it neither overestimates nor underestimates $\mu$.

### Variance and Standard Error

$$
\text{Var}(\bar{X}) = \frac{\sigma^2}{n}, \qquad
\text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}}.
$$

As $n$ increases, the standard error decreases — larger samples yield more precise estimates of $\mu$.

### Shape (Central Limit Theorem)

By the CLT, for sufficiently large $n$:

$$
\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1),
$$

so that approximately:

$$
\bar{X} \sim N\!\left(\mu, \frac{\sigma^2}{n}\right).
$$

This holds regardless of the population's shape, as long as $\sigma^2 < \infty$.

- For **normal** populations, this is exact for **all** $n$.
- For **skewed or heavy-tailed** populations, larger $n$ is needed.

## Standardized Forms

| Scenario | Standardized Statistic | Distribution |
|----------|----------------------|--------------|
| Normal pop., $\sigma$ known | $\frac{\bar{X} - \mu}{\sigma/\sqrt{n}}$ | $N(0, 1)$ exactly |
| Normal pop., $\sigma$ unknown | $\frac{\bar{X} - \mu}{S/\sqrt{n}}$ | $t_{n-1}$ exactly |
| Any pop., large $n$, $\sigma$ known | $\frac{\bar{X} - \mu}{\sigma/\sqrt{n}}$ | $N(0, 1)$ approximately |
| Any pop., large $n$, $\sigma$ unknown | $\frac{\bar{X} - \mu}{S/\sqrt{n}}$ | $N(0, 1)$ or $t_{n-1}$ approximately |

## Example: Standard Error Computation

**Problem.** Population has $\mu = 100$, $\sigma = 4$. For $n = 25$:

$$
\text{SE}(\bar{X}) = \frac{4}{\sqrt{25}} = 0.8.
$$

If we repeatedly draw samples of size 25, the sample means will cluster around 100 with a typical deviation of 0.8.

## Worked Examples

### Example 1: Apple Weights

**Problem.** Apple weights are $N(150, 20^2)$. For $n = 25$, find $P(\bar{X} > 155)$.

**Solution.**

$$
\text{SE} = \frac{20}{\sqrt{25}} = 4, \qquad
Z = \frac{155 - 150}{4} = 1.25
$$

$$
P(\bar{X} > 155) = P(Z > 1.25) = 1 - \Phi(1.25) \approx 0.1056
$$

```python
from scipy import stats
print(f"P(X_bar > 155) = {stats.norm.sf(1.25):.4f}")
```

### Example 2: Sleep Duration

**Problem.** Average sleep is 7 hours, $\sigma = 1.5$. For $n = 49$, find $P(6.8 < \bar{X} < 7.2)$.

**Solution.**

$$
\text{SE} = \frac{1.5}{\sqrt{49}} = 0.2143
$$

$$
Z_1 = \frac{6.8 - 7}{0.2143} \approx -0.93, \qquad
Z_2 = \frac{7.2 - 7}{0.2143} \approx 0.93
$$

$$
P(6.8 < \bar{X} < 7.2) = \Phi(0.93) - \Phi(-0.93) \approx 0.6476
$$

```python
from scipy import stats
print(f"P(6.8 < X_bar < 7.2) = {stats.norm.cdf(0.93) - stats.norm.cdf(-0.93):.4f}")
```

### Example 3: Body Weight (Small Sample, Normal Population)

**Problem.** Weights are $N(70, 10^2)$. For $n = 5$, find $P(\bar{X} > 72)$.

**Solution.** Because the population is normal, the result is exact even for $n = 5$:

$$
\text{SE} = \frac{10}{\sqrt{5}} \approx 4.47, \qquad
Z = \frac{72 - 70}{4.47} \approx 0.447
$$

$$
P(\bar{X} > 72) \approx 0.3274
$$

```python
from scipy import stats
print(f"P(X_bar > 72) = {stats.norm.sf(0.447):.4f}")
```

### Example 4: Running Out of Water

**Problem.** Average water consumption is 2 L ($\sigma = 0.7$ L). For 50 men on a trip with 110 L total, find $P(\text{run out})$.

**Solution.** Running out means $\bar{X} > 110/50 = 2.2$:

$$
\text{SE} = \frac{0.7}{\sqrt{50}} \approx 0.0990, \qquad
Z = \frac{2.2 - 2}{0.0990} \approx 2.020
$$

$$
P(\bar{X} > 2.2) = P(Z > 2.020) \approx 0.0217
$$

### Example 5: Lightbulbs (No Normality Assumption)

**Problem.** Lightbulb lifespan has $\mu = 800$, $\sigma = 100$. For $n = 5$, find $P(\bar{X} > 810)$ without assuming normality.

**Solution.** With $n = 5$ and no normality assumption, the CLT does not reliably apply. We **cannot** determine this probability without additional information about the population's shape.

### Example 6: Skewed Sales (Large Sample)

**Problem.** Daily sales are right-skewed with $\mu = 2000$, $\sigma = 500$. For $n = 100$, find $P(\bar{X} > 2100)$.

**Solution.** Even though the population is skewed, $n = 100$ is large enough for the CLT:

$$
\text{SE} = \frac{500}{\sqrt{100}} = 50, \qquad
Z = \frac{2100 - 2000}{50} = 2
$$

$$
P(\bar{X} > 2100) = P(Z > 2) \approx 0.0228
$$

```python
from scipy import stats
print(f"P(X_bar > 2100) = {stats.norm.sf(2):.4f}")
```

## Sampling Distribution of Two Means

### Known Variances or Large Samples

$$
Z = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{\sigma_1^2/n_1 + \sigma_2^2/n_2}} \sim N(0, 1)
$$

### Example: Two Cupcake Shifts

**Problem.** Shift A: $\mu_A = 130$g, $\sigma_A = 4$g. Shift B: $\mu_B = 125$g, $\sigma_B = 3$g. With $n_A = n_B = 40$, find $P(|\bar{X}_A - \bar{X}_B| > 6)$.

**Solution.**

$$
\text{SE} = \sqrt{\frac{16}{40} + \frac{9}{40}} = \sqrt{0.625} \approx 0.7906
$$

$$
P(\bar{X}_A - \bar{X}_B > 6): \quad Z = \frac{6 - 5}{0.7906} \approx 1.265, \quad P = 0.1030
$$

$$
P(\bar{X}_A - \bar{X}_B < -6): \quad Z = \frac{-6 - 5}{0.7906} \approx -13.91, \quad P \approx 0
$$

$$
P(|\bar{X}_A - \bar{X}_B| > 6) \approx 0.1030
$$

```python
import numpy as np
from scipy import stats

se = np.sqrt(16/40 + 9/40)
z_upper = (6 - 5) / se
z_lower = (-6 - 5) / se
prob = stats.norm.sf(z_upper) + stats.norm.cdf(z_lower)
print(f"P(|X_bar_A - X_bar_B| > 6) = {prob:.4f}")
```

## Effect of Sample Size on Standard Error

| $n$ | SE (with $\sigma = 50$) |
|-----|------------------------|
| 25 | 10 |
| 100 | 5 |
| 400 | 2.5 |

Quadrupling $n$ halves the standard error, since $\text{SE} \propto 1/\sqrt{n}$.

## Simulation: Sampling Distribution of $\bar{X}$

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(1)

population = stats.norm().rvs(100_000)
sample_size = 10
n_samples = 10_000

sample_means = [
    np.mean(np.random.choice(population, size=sample_size, replace=False))
    for _ in range(n_samples)
]

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))

ax0.hist(population, bins=100, density=True, alpha=0.5)
ax0.set_title('Population Distribution (Normal)', fontsize=16)

ax1.hist(sample_means, bins=100, density=True, alpha=0.5)
ax1.set_title(rf'Sampling Distribution of $\bar{{X}}$ (n = {sample_size})', fontsize=16)

for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

## Graduate-Level Notes

- The efficiency of $\bar{X}$ is linked to the rate $\text{Var}(\bar{X}) = \sigma^2/n$ — it achieves the Cramér–Rao lower bound under normality.
- For populations with **infinite variance** (e.g., Cauchy), the CLT does not apply and $\bar{X}$ may not converge.
- The **Berry–Esseen theorem** quantifies the rate of CLT convergence: $\sup_z |P(Z_n \leq z) - \Phi(z)| \leq C \cdot \rho / (\sigma^3 \sqrt{n})$, where $\rho = E[|X - \mu|^3]$.

## Summary

| Property | Result |
|----------|--------|
| $E[\bar{X}]$ | $\mu$ (unbiased) |
| $\text{Var}(\bar{X})$ | $\sigma^2/n$ |
| $\text{SE}(\bar{X})$ | $\sigma/\sqrt{n}$ |
| Shape | Normal (exact if pop. is normal; approximate via CLT for large $n$) |
| Key insight | Larger $n$ → smaller SE → more precise estimate |
