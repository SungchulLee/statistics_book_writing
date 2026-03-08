# Repeated Sampling Concept

## Overview

A **sampling distribution** is the probability distribution of a given statistic based on a random sample. When we draw multiple random samples from the same population and calculate a statistic (such as the sample mean or sample proportion) for each sample, the resulting values form a distribution. This distribution is known as the **sampling distribution** of that statistic.

$$
\left.
\begin{array}{ccccc}
\text{Population} &\rightarrow& \text{Sample } \mathbf{x}_1 &\rightarrow& \hat{\theta}(\mathbf{x}_1) \\
\\
\text{Population} &\rightarrow& \text{Sample } \mathbf{x}_2 &\rightarrow& \hat{\theta}(\mathbf{x}_2) \\
&\vdots& & & \\
\text{Population} &\rightarrow& \text{Sample } \mathbf{x}_n &\rightarrow& \hat{\theta}(\mathbf{x}_n) \\
&\vdots& & &
\end{array}
\right\}
\;\;
\begin{array}{c}
\text{Sampling Distribution:} \\
\text{Distribution of } \hat{\theta}(\mathbf{x}_1), \hat{\theta}(\mathbf{x}_2), \cdots, \hat{\theta}(\mathbf{x}_n), \cdots \\
\text{or Distribution of } \hat{\theta}(\mathbf{x})
\end{array}
$$

## Why Are Sampling Distributions Important?

Sampling distributions are fundamental to inferential statistics, which involves making conclusions about a population based on a sample. By understanding the behavior of a statistic across multiple samples, we can:

- **Estimate population parameters** (e.g., mean, variance) using statistics from the sample.
- **Calculate the standard error** to understand the variability of an estimator.
- **Formulate confidence intervals** to quantify the uncertainty of our estimates.
- **Perform hypothesis testing** to make informed decisions about population parameters.

## Three Distributions to Distinguish

### Population Distribution, Sample Distribution, and Sampling Distribution

The **Population Distribution** represents the distribution of all possible values of a variable in the entire population. This is the underlying distribution that characterizes the population from which we draw samples.

The **Sample Distribution** refers to the distribution of values within a specific sample drawn from the population. A sample is a subset of the population, and we use it to make inferences about the entire population.

The **Sampling Distribution** describes the distribution of a statistic or estimator (such as the sample mean or sample proportion) computed from multiple samples of the same size drawn from the population. It helps us understand the variability of a statistic and is central to statistical inference.

$$
\begin{array}{ccccccc}
\text{Population}
&\rightarrow&
\text{Sample } \mathbf{x}
&\rightarrow&
\text{Estimate } \hat{\theta}(\mathbf{x}) \\
\uparrow && \uparrow && \uparrow \\
\text{Population Distribution:} && \text{Sample Distribution:} && \text{Sampling Distribution:} \\
\text{Distribution of} && \text{Distribution of} && \text{Distribution of} \\
\text{Whole Population} && \text{Numbers in Particular Sample } \mathbf{x} && \text{Infinitely Many Estimates } \hat{\theta}(\mathbf{x}_i)
\end{array}
$$

## Simulation 1: Uniform Population

```python
import matplotlib.pyplot as plt
import numpy as np

# Set a random seed for reproducibility
np.random.seed(1)

# Define parameters for the population, sample size, and number of samples for the simulation
sample_size = 5        # Size of a single random sample
n_samples = 10_000     # Number of samples to draw for the sampling distribution
n_population = 10_000  # Size of the population to simulate

def plot_distributions():
    """
    Generates a plot showing the population distribution, sample distribution,
    and sampling distribution.
    """
    # Generate a large population from a uniform distribution
    population = np.random.uniform(size=(n_population,))

    # Generate a single random sample from the population
    single_sample = np.random.choice(population, size=sample_size, replace=False)

    # Generate multiple samples and compute their means
    sample_means = [
        np.mean(np.random.choice(population, size=sample_size, replace=False))
        for _ in range(n_samples)
    ]

    # Create a 3-row subplot
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot the population distribution
    ax0.hist(population, bins=np.linspace(0, 1, 100))
    ax0.set_title('Population Distribution', fontsize=20)

    # Plot the sample distribution (scatter plot)
    ax1.scatter(single_sample, np.zeros_like(single_sample), s=100)
    ax1.set_title(f'Sample Distribution of {sample_size} Samples', fontsize=20)

    # Plot the sampling distribution (histogram of sample means)
    ax2.hist(sample_means, bins=np.linspace(0, 1, 100))
    ax2.set_title('Sampling Distribution of $\\bar{X}$', fontsize=20)

    # Adjust the aesthetics
    for ax in (ax0, ax1, ax2):
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_distributions()
```

**Observation.** Even though the population is uniform (flat), the sampling distribution of $\bar{X}$ is bell-shaped and much more concentrated — a preview of the Central Limit Theorem.

## Simulation 2: Exponential Population

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Set a random seed for reproducibility
np.random.seed(1)

# Define parameters
sample_size = 30
n_samples = 10_000
n_population = 10_000

def plot_distributions():
    """
    Generates a plot showing the population distribution, sample distribution,
    and sampling distribution for an exponential population.
    """
    # Generate a large population from an exponential distribution
    population = stats.expon().rvs((n_population,))

    # Generate a single random sample from the population
    single_sample = np.random.choice(population, size=sample_size, replace=False)

    # Generate multiple samples and compute their means
    sample_means = [
        np.mean(np.random.choice(population, size=sample_size, replace=False))
        for _ in range(n_samples)
    ]

    # Create a 3-row subplot
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot the population distribution
    _, bins, _ = ax0.hist(population, bins=100)
    ax0.set_title('Population Distribution', fontsize=20)

    # Plot the sample distribution (scatter plot)
    ax1.scatter(single_sample, np.zeros_like(single_sample), s=100)
    ax1.set_title(f'Sample Distribution of {sample_size} Samples', fontsize=20)

    # Plot the sampling distribution (histogram of sample means)
    ax2.hist(sample_means, bins=bins)
    ax2.set_title('Sampling Distribution of $\\bar{X}$', fontsize=20)

    # Adjust the aesthetics
    for ax in (ax0, ax1, ax2):
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_distributions()
```

**Observation.** Although the exponential population is heavily right-skewed, the sampling distribution of $\bar{X}$ with $n = 30$ is approximately normal — the Central Limit Theorem at work.

## Simulation 3: Bernoulli Population

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Set a random seed for reproducibility
np.random.seed(1)

# Define parameters
sample_size = 30
n_samples = 10_000
n_population = 10_000

def plot_distributions():
    """
    Generates a plot showing the population distribution, sample distribution,
    and sampling distribution for a Bernoulli population.
    """
    # Generate a large population from a Bernoulli distribution
    population = stats.binom(n=1, p=0.3).rvs((n_population,))

    # Generate a single random sample from the population
    single_sample = np.random.choice(population, size=sample_size, replace=False)

    # Generate multiple samples and compute their means
    sample_means = [
        np.mean(np.random.choice(population, size=sample_size, replace=False))
        for _ in range(n_samples)
    ]

    # Create a 3-row subplot
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot the population distribution
    _, bins, _ = ax0.hist(population, bins=100)
    ax0.set_title('Population Distribution', fontsize=20)

    # Plot the sample distribution (scatter plot)
    ax1.scatter(single_sample, np.zeros_like(single_sample), s=100)
    ax1.set_title(f'Sample Distribution of {sample_size} Samples', fontsize=20)

    # Plot the sampling distribution (histogram of sample means)
    ax2.hist(sample_means, bins=10)
    ax2.set_title('Sampling Distribution of $\\bar{X}$', fontsize=20)

    # Adjust the aesthetics
    for ax in (ax0, ax1, ax2):
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_distributions()
```

## Example: Sampling Distribution of Two Balls Drawn from Three Balls

> **Reference:** [Khan Academy — Introduction to Sampling Distributions](https://www.khanacademy.org/math/ap-statistics/sampling-distribution-ap/what-is-sampling-distribution/v/introduction-to-sampling-distributions)

**Problem.** There are three balls in an urn, numbered one, two, and three. The population mean is $\mu = 2$. We choose two balls with replacement and compute their mean. Compute the distribution of this sample mean — that is, the sampling distribution of $\bar{X}$.

**Solution.** There are $3^2 = 9$ equally likely outcomes:

```python
import itertools as it
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():
    sample_space = np.array([1, 2, 3])

    columns = ["first", "second", "average"]
    df = pd.DataFrame(columns=columns)
    for first, second in it.product(sample_space, repeat=2):
        dg = pd.DataFrame([[first, second, (first + second) / 2]], columns=columns)
        df = pd.concat([df, dg], ignore_index=True)
    print(df, end="\n\n")

    fig, ax = plt.subplots(figsize=(12, 3))
    bins = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]) - 0.25
    ax.hist(df.average, bins=bins, density=True, alpha=0.7)
    ax.set_title(r"Sampling Distribution of $\bar{X}$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()
```

The sampling distribution has possible values $\{1.0, 1.5, 2.0, 2.5, 3.0\}$ with probabilities $\{1/9, 2/9, 3/9, 2/9, 1/9\}$. Its mean is $E[\bar{X}] = 2 = \mu$, confirming that $\bar{X}$ is unbiased.
