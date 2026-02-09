# Populations and Samples

## Overview

In statistics, understanding the difference between **population** and **sample** is crucial for any analysis, including regression, ANOVA, or any other statistical test. These two concepts form the foundation of inferential statistics, where we make conclusions about a larger group based on data from a smaller group.

## Population

The **population** refers to the entire set of individuals, objects, or events that share some common characteristic and are of interest in a study. It is the complete collection of all possible data points we would like to understand or make inferences about.

- If we want to study the average height of adults in a country, the population would be **all adults** in that country.
- Populations can be **finite** (all adults in a particular country) or **infinite** (the number of fish in a vast ocean, which is impractical to measure exhaustively).

## Sample

A **sample** is a subset of the population that is selected for actual measurement, observation, or experimentation. It is used to draw inferences about the population when it is impractical or impossible to collect data from the entire population.

- Instead of measuring the height of every adult in the country, we may take a sample of 1,000 adults and estimate the average height based on this smaller group.
- Samples should ideally be **random** and **representative** of the population to avoid biases. Non-random or biased samples may lead to incorrect conclusions about the population.

## Why Sample Instead of Studying the Whole Population?

Studying the entire population is often:

- **Costly**: It is expensive to gather data from every individual or object in the population.
- **Time-consuming**: Collecting and analyzing data from the entire population can be impractical.
- **Infeasible**: In some cases, the population is too large or inaccessible to be fully studied.

## The Importance of Random Sampling

To ensure that a sample is representative of the population, **random sampling** is used. This method gives each individual or unit in the population an equal chance of being selected, reducing the risk of bias. If the sample is not random, the conclusions drawn from it may not accurately reflect the population.

## Population Parameters vs. Sample Statistics

**Population parameters** are characteristics of the entire population, such as the population mean ($\mu$) or population variance ($\sigma^2$):

$$
\mu = \frac{1}{N}\sum_{i=1}^{N} x_i, \qquad
\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2
$$

where $N$ is the total number of individuals in the population.

**Sample statistics** are values calculated from the sample, such as the sample mean ($\bar{x}$) or sample variance ($s^2$). These are used to **estimate** the population parameters:

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i, \qquad
s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2
$$

where $n$ is the number of samples. Note that the sample variance uses $n-1$ (Bessel's correction) rather than $n$ to provide an unbiased estimate of the population variance.

## Sampling Error

**Sampling error** refers to the difference between a sample statistic and the actual population parameter. This error occurs because the sample is only a subset of the population and may not perfectly represent it. Larger samples generally reduce sampling error, leading to more accurate estimates of the population parameters.

## Python Example

```python
import numpy as np

# Simulate a population
np.random.seed(42)
population = np.random.normal(loc=170, scale=10, size=100_000)  # heights in cm

# Population parameters
mu = population.mean()
sigma2 = population.var()  # uses N
print(f"Population mean (μ):       {mu:.2f}")
print(f"Population variance (σ²):  {sigma2:.2f}")

# Draw a random sample
n = 1_000
sample = np.random.choice(population, size=n, replace=False)

# Sample statistics
x_bar = sample.mean()
s2 = sample.var(ddof=1)  # uses n-1 (Bessel's correction)
print(f"\nSample mean (x̄):          {x_bar:.2f}")
print(f"Sample variance (s²):      {s2:.2f}")

# Sampling error
print(f"\nSampling error (mean):     {abs(x_bar - mu):.4f}")
```

## Key Takeaways

| Concept | Population | Sample |
|---|---|---|
| **Definition** | Entire group of interest | Subset selected for study |
| **Size notation** | $N$ | $n$ |
| **Mean notation** | $\mu$ | $\bar{x}$ |
| **Variance divisor** | $N$ | $n - 1$ |
| **Goal** | Describe | Estimate and infer |

Understanding the distinction between populations and samples is essential before moving on to any inferential procedure—whether it is a confidence interval, hypothesis test, or regression model.
