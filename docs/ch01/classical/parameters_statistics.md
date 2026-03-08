# Parameters vs. Statistics

## Overview

A core distinction in statistical inference is between **parameters**—fixed but unknown quantities that describe a population—and **statistics**—computable quantities derived from a sample that serve as estimates of those parameters. Every inferential procedure (confidence intervals, hypothesis tests, regression) rests on this distinction.

## Definitions

| Term | Scope | Notation (typical) | Known? |
|---|---|---|---|
| **Parameter** | Population | $\mu,\; \sigma^2,\; p,\; \beta$ | Usually unknown |
| **Statistic** | Sample | $\bar{x},\; s^2,\; \hat{p},\; \hat{\beta}$ | Computable from data |

A **parameter** is a numerical characteristic of a population—for example, the true average return of all stocks listed on the NYSE in a given year. A **statistic** is the corresponding quantity computed from a sample—for example, the average return of 50 randomly selected NYSE stocks.

## Why the Distinction Matters

Because we almost never observe the full population, we rely on sample statistics to **estimate** population parameters. The quality of that estimation depends on properties such as bias, consistency, and efficiency, which are discussed in later chapters.

## Common Parameter–Statistic Pairs

### Mean

$$
\text{Parameter: } \mu = \frac{1}{N}\sum_{i=1}^{N} x_i
\qquad\longleftrightarrow\qquad
\text{Statistic: } \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

### Variance

$$
\text{Parameter: } \sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2
\qquad\longleftrightarrow\qquad
\text{Statistic: } s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

The sample variance uses $n-1$ (Bessel's correction) so that $E[s^2] = \sigma^2$, making it an **unbiased** estimator.

### Proportion

$$
\text{Parameter: } p
\qquad\longleftrightarrow\qquad
\text{Statistic: } \hat{p} = \frac{\text{number of successes}}{n}
$$

## Sampling Variability

Because a statistic is computed from a random sample, it is itself a **random variable**. If we drew a different sample, we would get a different value for $\bar{x}$. The distribution of a statistic over all possible samples of a given size is called its **sampling distribution**. Two key properties of that distribution are:

- **Standard error**: the standard deviation of the statistic's sampling distribution (e.g., $\text{SE}(\bar{x}) = \sigma / \sqrt{n}$).
- **Bias**: the difference $E[\hat{\theta}] - \theta$, where $\hat{\theta}$ is the estimator and $\theta$ is the parameter.

## Python Example

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# True population parameter
mu_true = 100
sigma_true = 15
population = np.random.normal(mu_true, sigma_true, size=500_000)

# Repeated sampling: compute sample means
n = 50
num_samples = 2_000
sample_means = [np.random.choice(population, n, replace=False).mean()
                for _ in range(num_samples)]

# The sampling distribution of x-bar
print(f"True μ:                {mu_true}")
print(f"Mean of sample means:  {np.mean(sample_means):.2f}")
print(f"Theoretical SE:        {sigma_true / np.sqrt(n):.2f}")
print(f"Observed SE:           {np.std(sample_means, ddof=1):.2f}")

plt.figure(figsize=(8, 3))
plt.hist(sample_means, bins=40, density=True, alpha=0.7, edgecolor="black")
plt.axvline(mu_true, color="red", linestyle="--", label=f"μ = {mu_true}")
plt.xlabel("Sample Mean")
plt.ylabel("Density")
plt.title("Sampling Distribution of the Sample Mean")
plt.legend()
plt.tight_layout()
plt.show()
```

## Key Takeaways

- Parameters describe populations; statistics describe samples.
- Statistics are random variables whose distributions (sampling distributions) are the bridge from data to inference.
- An estimator's usefulness is judged by its bias, variance, and consistency—topics developed fully in Chapter 6.
