# Central Limit Theorem

## Overview

The **Central Limit Theorem (CLT)** is one of the most important results in all of probability and statistics. It states that the sampling distribution of the sample mean of a sufficiently large number of i.i.d. random variables is approximately normal, **regardless** of the original distribution, provided the population has finite mean and variance.

---

## Statement of the CLT

If $X_1, X_2, \ldots, X_n$ are i.i.d. random variables with mean $\mu$ and variance $\sigma^2$, then the standardized sample mean converges in distribution to a standard normal:

$$
\frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0, 1) \quad \text{as } n \to \infty
$$

Equivalently, the sample mean is approximately normally distributed for large $n$:

$$
\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)
$$

Or in terms of the sum $S_n = \sum_{i=1}^n X_i$:

$$
S_n \approx N(n\mu, \, n\sigma^2)
$$

---

## From LLN to CLT

The Law of Large Numbers tells us **where** the sample mean converges: $\bar{X} \to \mu$. The CLT tells us **how fast** and **in what shape** the fluctuations around $\mu$ behave:

$$
\frac{\sqrt{n}}{\sigma}(\bar{X} - \mu) = \frac{S_n - n\mu}{\sqrt{n\sigma^2}} \xrightarrow{d} N(0, 1)
$$

The LLN says the deviation $\bar{X} - \mu \to 0$. By rescaling by $\sqrt{n}$, the CLT reveals that these deviations have a non-trivial normal structure.

---

## Practical Guidelines

### Minimum Sample Size ($n \geq 30$)

A commonly cited rule of thumb is that $n \geq 30$ is "large enough" for the CLT approximation to hold:

- If the population is approximately symmetric, even $n \approx 15$–20 may suffice.
- If the population is skewed or heavy-tailed, $n \geq 40$–50 may be needed.
- The number 30 is a **convention**, not a theorem.

### Normal Approximation for Proportions

When dealing with sample proportions (binomial data), the CLT requires:

$$
np \geq 5 \quad \text{and} \quad n(1-p) \geq 5
$$

Some textbooks use the stricter rule $np \geq 10$ and $n(1-p) \geq 10$.

### The 10% Condition for Independence

When sampling without replacement from a finite population of size $N$, draws are not independent. The finite population correction is:

$$
\text{Var}(\bar{X}_n) = \frac{\sigma^2}{n} \cdot \frac{N - n}{N - 1}
$$

The correction is negligible if the sampling fraction is small:

$$
\frac{n}{N} \leq 10\%
$$

If this holds, we can safely treat the sample as i.i.d.

---

## Normal Approximation

### Approximation to the Binomial

For $X \sim \text{Binomial}(n, p)$ with large $n$:

$$
X \approx N(np, \, np(1-p))
$$

With **continuity correction**:

$$
P(X \leq k) \approx P\left(Z \leq \frac{k + 0.5 - np}{\sqrt{np(1-p)}}\right)
$$

### Approximation to the Poisson

For $X \sim \text{Poisson}(\lambda)$ with large $\lambda$:

$$
X \approx N(\lambda, \lambda)
$$

---

## CLT in Action: Uniform and Exponential Distributions

The CLT works regardless of the original distribution. However, the **rate of convergence** depends on the distribution's shape:

- **Symmetric distributions** (e.g., uniform) converge to normality very quickly.
- **Skewed distributions** (e.g., exponential) converge more slowly—extreme values have a more pronounced effect, requiring larger sample sizes.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def demonstrate_clt(distribution_type, sample_size, n_simulations=10_000):
    """Demonstrate CLT convergence for a given distribution."""
    np.random.seed(0)

    if distribution_type == 'uniform':
        data = np.mean(stats.uniform().rvs((sample_size, n_simulations)), axis=0)
        label = 'Uniform(0,1)'
    elif distribution_type == 'exponential':
        data = np.mean(stats.expon().rvs((sample_size, n_simulations)), axis=0)
        label = 'Exponential(1)'

    mu, sigma = data.mean(), data.std()

    fig, ax = plt.subplots(figsize=(12, 3))
    _, bins, _ = ax.hist(data, bins=100, density=True, alpha=0.3, color='blue',
                         label=f'Sample Means (n={sample_size})')
    ax.plot(bins, stats.norm(mu, sigma).pdf(bins), '--r', lw=2, label='Normal PDF')
    ax.set_title(f'CLT: Sample Means from {label}')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.show()

# Demonstrate with both distributions
demonstrate_clt('uniform', sample_size=5)
demonstrate_clt('exponential', sample_size=5)
```

---

## Applications

The CLT underpins many core statistical procedures:

- **Hypothesis testing:** $z$-tests and $t$-tests assume the sampling distribution of the test statistic is approximately normal.
- **Confidence intervals:** Constructed using normal quantiles, justified by the CLT.
- **Quality control:** Assessing whether sample means of product measurements meet standards.

---

## Putting It All Together

Before applying a normal approximation, check these conditions:

| Condition | Rule of Thumb |
|:---|:---|
| Sample size | $n \geq 30$ (unless population is nearly normal) |
| Proportions | $np \geq 5$ and $n(1-p) \geq 5$ |
| Finite population sampling | $n/N \leq 10\%$ |

These are not strict theorems but widely adopted **practical guidelines** that bridge the ideal mathematical world and real data analysis.

---

## Key Takeaways

- The CLT guarantees that sample means are approximately normal for large $n$, regardless of the population distribution.
- The rate of convergence depends on the skewness of the original distribution.
- Practical conditions ($n \geq 30$, success/failure counts, 10% rule) ensure the approximation is reliable.
- The CLT is the theoretical backbone of confidence intervals, hypothesis tests, and much of applied statistics.
