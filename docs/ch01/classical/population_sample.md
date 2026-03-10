# Populations and Samples

The distinction between population and sample is the foundation of all inferential statistics. Every confidence interval, hypothesis test, and regression model rests on using a sample to learn about a population.

## Definition

The **population** is the complete set of individuals or observations of interest. A **sample** is a subset of the population selected for measurement. Population quantities are **parameters** (denoted $\mu$, $\sigma^2$, $p$); sample quantities are **statistics** (denoted $\bar{x}$, $s^2$, $\hat{p}$).

$$
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i \qquad \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i
$$

$$
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2 \qquad s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

## Explanation

We sample because studying the entire population is typically too costly, time-consuming, or infeasible. The goal is a **representative** sample where every unit has a known nonzero probability of selection (random sampling).

**Sampling error** is the unavoidable difference between a sample statistic and the population parameter. It shrinks with larger $n$ but never vanishes unless $n = N$. The sample variance uses $n - 1$ (Bessel's correction) to produce an unbiased estimate of $\sigma^2$.

| | Population | Sample |
|---|---|---|
| Size | $N$ | $n$ |
| Mean | $\mu$ | $\bar{x}$ |
| Variance divisor | $N$ | $n - 1$ |
| Goal | Describe | Estimate and infer |

## Examples

```python
import numpy as np

np.random.seed(42)
population = np.random.normal(loc=170, scale=10, size=100_000)

mu = population.mean()
sigma2 = population.var()  # divides by N

n = 100
sample = np.random.choice(population, size=n, replace=False)
x_bar = sample.mean()
s2 = sample.var(ddof=1)  # divides by n-1

print(f"Population mean (mu):    {mu:.2f}")
print(f"Sample mean (x_bar):     {x_bar:.2f}")
print(f"Sampling error:          {abs(x_bar - mu):.4f}")
print(f"Population variance:     {sigma2:.2f}")
print(f"Sample variance (s^2):   {s2:.2f}")
```
