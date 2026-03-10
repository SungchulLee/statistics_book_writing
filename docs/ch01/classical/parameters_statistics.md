# Parameters vs Statistics

Every inferential procedure bridges the gap between fixed but unknown population parameters and computable sample statistics. This distinction is the starting point of all estimation and testing.

## Definition

A **parameter** is a numerical characteristic of a population (e.g., $\mu$, $\sigma^2$, $p$, $\beta$). A **statistic** is the corresponding quantity computed from a sample (e.g., $\bar{x}$, $s^2$, $\hat{p}$, $\hat{\beta}$). Because a statistic depends on a random sample, it is itself a random variable whose distribution over all possible samples is called its **sampling distribution**.

## Explanation

Common parameter-statistic pairs:

- **Mean**: $\mu \longleftrightarrow \bar{x} = \frac{1}{n}\sum x_i$
- **Variance**: $\sigma^2 \longleftrightarrow s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$ (Bessel's correction ensures $E[s^2] = \sigma^2$)
- **Proportion**: $p \longleftrightarrow \hat{p} = \text{successes}/n$

The **standard error** $\text{SE}(\bar{x}) = \sigma/\sqrt{n}$ measures the spread of the sampling distribution. The **bias** is $E[\hat{\theta}] - \theta$. An estimator is useful when it has low bias, low variance, and converges to the true parameter as $n \to \infty$ (consistency). These properties are developed in Chapter 6.

## Examples

```python
import numpy as np

np.random.seed(0)
mu_true = 100
sigma_true = 15
population = np.random.normal(mu_true, sigma_true, size=500_000)

n = 50
n_samples = 5_000
sample_means = np.array([
    np.random.choice(population, n, replace=False).mean()
    for _ in range(n_samples)
])

print(f"True mu:              {mu_true}")
print(f"Mean of sample means: {sample_means.mean():.2f}")
print(f"Theoretical SE:       {sigma_true / np.sqrt(n):.2f}")
print(f"Observed SE:          {sample_means.std(ddof=1):.2f}")
```
