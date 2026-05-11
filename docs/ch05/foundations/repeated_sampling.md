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

## Exercises

**Exercise 1.**
A population has mean 100 and SD 20. As sample size $n$ increases, what happens to (a) the mean of the sampling distribution of $\bar X$, (b) the SE, (c) a single sample mean $\bar x$?

??? success "Solution to Exercise 1"
    (a) **Mean of sampling distribution** = population mean = 100, for every $n$. The sampling distribution is centered on $\mu$ regardless of sample size — $\bar X$ is *unbiased*.

    (b) **Standard error** $= \sigma/\sqrt n = 20/\sqrt n$. Shrinks as $n$ grows: $n = 4 \to \mathrm{SE} = 10$; $n = 100 \to \mathrm{SE} = 2$; $n = 10000 \to \mathrm{SE} = 0.2$.

    (c) **A single sample mean** $\bar x$ converges in probability to 100 by the WLLN, and almost surely by the SLLN. The narrowing sampling distribution means that an individual $\bar x$ is increasingly likely to be close to 100.

---

**Exercise 2.**
**Standard error vs. standard deviation.** Explain the distinction with examples. Why is the standard error always $\sigma / \sqrt n$, not $\sigma$?

??? success "Solution to Exercise 2"
    **Standard deviation $\sigma$:** measures spread of the *population* (or a single sample). Describes how much individual observations vary.

    **Standard error $\sigma/\sqrt n$:** measures spread of a *statistic's sampling distribution* (typically the sample mean). Describes how much $\bar X$ varies from sample to sample.

    Two distinct quantities. Reporting "SD = 5" describes the data. Reporting "SE = 0.5" describes uncertainty about an estimate.

    **Why $\sigma/\sqrt n$ for the mean?** Recall $\mathrm{Var}(\bar X) = \mathrm{Var}((1/n)\sum X_i) = (1/n^2) \cdot n\sigma^2 = \sigma^2/n$ for i.i.d. data. The square root gives SE $= \sigma/\sqrt n$. The $\sqrt n$ rate is the "law of square-root improvement": quadrupling the sample halves the SE.

---

**Exercise 3.**
**Sampling without replacement.** For a finite population of size $N$ with mean $\mu$ and variance $\sigma^2$, derive the variance of $\bar X$ when sampling $n$ items *without replacement*.

??? success "Solution to Exercise 3"
    $\mathrm{Var}(\bar X) = \frac{\sigma^2}{n} \cdot \frac{N - n}{N - 1}$ — the **finite-population correction (FPC)** factor.

    Derivation: the $X_i$'s are no longer independent (the second draw depends on the first), but they are exchangeable. Compute the variance of the sum:

    $\mathrm{Var}(\sum X_i) = n\sigma^2 + n(n-1) \mathrm{Cov}(X_1, X_2)$. By symmetry, $\sum_{i \ne j} \mathrm{Cov}(X_i, X_j) = -\mathrm{Var}(\sum X_i^{\text{total}})/(N-1)$, giving $\mathrm{Cov}(X_1, X_2) = -\sigma^2/(N-1)$.

    Therefore $\mathrm{Var}(\sum X_i) = n\sigma^2(N-n)/(N-1)$, and dividing by $n^2$ gives the FPC formula.

    **Practical:** when $n/N \le 0.05$, the FPC is close to 1 and can be ignored. When $n/N$ is substantial (auditing, recounting), FPC matters and standard-error formulas must include it.

---

**Exercise 4.**
**Asymptotic vs. exact distribution.** For a sample of size $n = 5$ from $\mathrm{Uniform}(0, 1)$, the exact distribution of $\bar X$ is known (Irwin-Hall scaled). Sketch the shape and compare with the CLT-predicted normal approximation.

??? success "Solution to Exercise 4"
    **Exact:** sum of 5 i.i.d. Uniform(0, 1) is the Irwin-Hall distribution on $[0, 5]$ — piecewise polynomial of degree 4, bell-shaped, symmetric around 2.5. Scaling by $1/5$ gives $\bar X$ on $[0, 1]$, peaked at $1/2$.

    **CLT approximation:** $\bar X \approx N(1/2, 1/(12 \cdot 5)) = N(0.5, 0.0167)$, SD $\approx 0.129$.

    **Comparison:** the exact distribution is bounded on $[0, 1]$; the normal extends to $\pm \infty$. At the center the two are nearly identical; in the tails the exact has zero probability outside $[0, 1]$ while the normal has small positive probability there (about 0.001 below 0 or above 1).

    For $n = 5$, the approximation is already quite good — the uniform's symmetry and bounded support give very fast CLT convergence. By contrast, an Exponential(1) sample of size 5 would have visibly skewed $\bar X$ distribution, far from normal.

---

**Exercise 5.**
**Convergence in the CLT** is **in distribution**, not pointwise. Define convergence in distribution and explain why this is the right notion for the CLT.

??? success "Solution to Exercise 5"
    **Convergence in distribution:** $X_n \xrightarrow{d} X$ iff $F_{X_n}(x) \to F_X(x)$ at every continuity point $x$ of $F_X$.

    This is a statement about *distributions*, not about random variables themselves. The $X_n$ and $X$ need not be defined on the same probability space — they just need to have CDFs converging.

    **Why the right notion for CLT:** the CLT compares the distribution of $\sqrt n (\bar X_n - \mu)/\sigma$ to a fixed normal. We are not claiming the realized values of $\bar X_n$ converge to a particular normal random variable — only that their *distribution* approaches normal. Different realizations have different limit behavior; the distribution is what stabilizes.

    Distinction from almost-sure convergence: $\bar X_n \to \mu$ a.s. (each path converges), but $\sqrt n (\bar X_n - \mu)$ does *not* converge a.s. anywhere — it keeps fluctuating, while its *distribution* approaches normal. This is convergence in distribution without any stronger form.

---

**Exercise 6.**
**Simulation experiment.** Describe how to *visualize* the sampling distribution of $\bar X$ from any population. What three plots would you make for $n = 5, 30, 100$?

??? success "Solution to Exercise 6"
    **Procedure:** Draw a sample of size $n$ from the population. Compute $\bar X$. Repeat $B$ times (e.g., $B = 10000$). The resulting $B$ values approximate the sampling distribution.

    **Three plots** for $n = 5, 30, 100$:

    1. **Histogram** of the $B$ values of $\bar X$, with theoretical normal $N(\mu, \sigma^2/n)$ overlaid. Visualizes the CLT approximation quality.
    2. **Q-Q plot** against the theoretical normal: if CLT is well-approximated, points fall on a line.
    3. **Triple comparison** of the three histograms side-by-side (or in a stacked layout): visual confirmation of (a) center stays at $\mu$, (b) spread shrinks as $\sigma/\sqrt n$, (c) shape becomes increasingly normal.

    **Key observations from such simulations:**

    - For symmetric populations (uniform, normal), $n = 5$ may suffice for approximate normality.
    - For skewed populations (exponential, lognormal), $n = 30$ is the conventional threshold but may still show visible skew.
    - For heavy-tailed populations (Cauchy, $t_1$), no $n$ produces normality — the CLT requires finite variance.

    This simulation-based diagnostic is more reliable than blindly applying "$n \ge 30$" rules of thumb.
