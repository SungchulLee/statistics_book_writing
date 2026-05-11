# Sampling Distribution of X-bar (Exponential)

## Overview

This page examines the sampling distribution of the sample mean $\bar{X}$ when the underlying population follows an Exponential distribution. The Exponential distribution is strongly right-skewed, making it an excellent test case for the Central Limit Theorem: despite the skewness of the population, the distribution of $\bar{X}$ becomes approximately normal as the sample size grows. With small $n$, the skewness is still visible in the sampling distribution.

## Population Model

Suppose each observation is drawn from an Exponential distribution with rate parameter $\lambda = 1$:

$$
X \sim \text{Exp}(1), \qquad f(x) = e^{-x}, \quad x \ge 0
$$

The population mean and variance are:

$$
\mu = E[X] = \frac{1}{\lambda} = 1, \qquad \sigma^2 = \text{Var}(X) = \frac{1}{\lambda^2} = 1
$$

## Sampling Distribution Theory

For a random sample $X_1, \ldots, X_n$ drawn independently from $\text{Exp}(\lambda)$:

$$
E[\bar{X}] = \mu = \frac{1}{\lambda}, \qquad \text{Var}(\bar{X}) = \frac{\sigma^2}{n} = \frac{1}{n\lambda^2}
$$

!!! info "Exact Distribution"
    The sum $S_n = \sum_{i=1}^n X_i$ follows a Gamma distribution: $S_n \sim \text{Gamma}(n, \lambda)$. Therefore $\bar{X} = S_n/n \sim \text{Gamma}(n, n\lambda)$, which has shape $n$ and rate $n\lambda$. As $n \to \infty$, the CLT guarantees:

    $$
    \bar{X} \;\dot{\sim}\; N\!\left(\frac{1}{\lambda},\; \frac{1}{n\lambda^2}\right)
    $$

## Simulation

The code below draws 10,000 samples of size $n = 5$ from an $\text{Exp}(1)$ population and visualizes the population, a single sample, and the sampling distribution of $\bar{X}$.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

sample_size = 5
n_samples = 10_000
n_population = 10_000

# Generate a large population from Exp(1)
population = np.random.exponential(size=(n_population,))

# Draw a single sample
single_sample = np.random.choice(population, size=sample_size, replace=False)

# Simulate the sampling distribution of X-bar
sample_means = [
    np.mean(np.random.choice(population, size=sample_size, replace=False))
    for _ in range(n_samples)
]

# Plot
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

_, bins, _ = ax0.hist(population, bins=100)
ax0.set_title("Population Distribution (Exponential)")

ax1.scatter(single_sample, np.zeros_like(single_sample), s=100)
ax1.set_title(f"Sample Distribution (n = {sample_size})")

ax2.hist(sample_means, bins=bins)
ax2.set_title("Sampling Distribution of X-bar")

plt.tight_layout()
plt.show()
```

## Interpretation

!!! note "Key Observations"

    1. The **population distribution** is strongly right-skewed with a long tail extending to large values.
    2. The **sampling distribution** of $\bar{X}$ is already much more concentrated than the population, even with $n = 5$.
    3. With $n = 5$, the sampling distribution retains some right skew. Increasing $n$ would make it progressively more symmetric and more closely approximate a normal distribution.
    4. The standard error is $\text{SE}(\bar{X}) = 1/\sqrt{5} \approx 0.447$, compared to the population standard deviation of $\sigma = 1$.

!!! warning "Small Samples from Skewed Populations"
    The CLT convergence rate depends on how skewed the population is. For the Exponential distribution (skewness $= 2$), you may need $n \ge 30$ or more before the normal approximation becomes reliable for inference.

## Exercises

**Exercise 1.** Derive the mean and variance of $X \sim \text{Exp}(\lambda)$ using the moment generating function $M_X(t) = \lambda / (\lambda - t)$ for $t < \lambda$.

??? success "Solution to Exercise 1"
    The moment generating function is $M_X(t) = \frac{\lambda}{\lambda - t}$ for $t < \lambda$.

    First moment:

    $$
    M_X'(t) = \frac{\lambda}{(\lambda - t)^2}, \qquad E[X] = M_X'(0) = \frac{\lambda}{\lambda^2} = \frac{1}{\lambda}
    $$

    Second moment:

    $$
    M_X''(t) = \frac{2\lambda}{(\lambda - t)^3}, \qquad E[X^2] = M_X''(0) = \frac{2\lambda}{\lambda^3} = \frac{2}{\lambda^2}
    $$

    Variance:

    $$
    \text{Var}(X) = E[X^2] - (E[X])^2 = \frac{2}{\lambda^2} - \frac{1}{\lambda^2} = \frac{1}{\lambda^2}
    $$

    $\square$

---

**Exercise 2.** Show that if $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Exp}(\lambda)$, then $S_n = \sum_{i=1}^n X_i \sim \text{Gamma}(n, \lambda)$.

??? success "Solution to Exercise 2"
    The MGF of $X_i \sim \text{Exp}(\lambda)$ is $M_{X_i}(t) = \frac{\lambda}{\lambda - t}$.

    By independence, the MGF of the sum is:

    $$
    M_{S_n}(t) = \prod_{i=1}^n M_{X_i}(t) = \left(\frac{\lambda}{\lambda - t}\right)^n
    $$

    This is the MGF of a $\text{Gamma}(n, \lambda)$ distribution (shape $n$, rate $\lambda$). Since the MGF uniquely determines the distribution, we conclude $S_n \sim \text{Gamma}(n, \lambda)$. $\square$

---

**Exercise 3.** For $n = 5$ and $\lambda = 1$, compute the exact probability $P(\bar{X} > 2)$ using the Gamma distribution, and compare it with the normal approximation.

??? success "Solution to Exercise 3"
    Since $\bar{X} = S_5 / 5$ where $S_5 \sim \text{Gamma}(5, 1)$, we have $P(\bar{X} > 2) = P(S_5 > 10)$.

    Using Python:

    ```python
    from scipy import stats
    # Exact (Gamma)
    p_exact = 1 - stats.gamma.cdf(10, a=5, scale=1)
    # Normal approximation: mean=1, se=1/sqrt(5)
    p_normal = 1 - stats.norm.cdf(2, loc=1, scale=1/5**0.5)
    print(f"Exact (Gamma):       {p_exact:.6f}")
    print(f"Normal approximation: {p_normal:.6f}")
    ```

    The exact value is approximately 0.0293, while the normal approximation gives approximately 0.0127. The normal approximation significantly underestimates the right-tail probability because $n = 5$ is too small for the CLT to fully correct for the Exponential's skewness. $\square$

---

**Exercise 4.** The skewness of the Exponential distribution is $\gamma_1 = 2$. Show that the skewness of $\bar{X}$ is $\gamma_1(\bar{X}) = 2/\sqrt{n}$. At what sample size does the skewness of $\bar{X}$ drop below 0.5?

??? success "Solution to Exercise 4"
    For i.i.d. random variables with skewness $\gamma_1$, the skewness of $\bar{X} = \frac{1}{n}\sum X_i$ is:

    $$
    \gamma_1(\bar{X}) = \frac{\gamma_1}{\sqrt{n}}
    $$

    This follows because the third central moment of $\bar{X}$ is $\mu_3 / n^2$ (from summing $n$ independent copies and dividing by $n$), and $(\text{Var}(\bar{X}))^{3/2} = (\sigma^2/n)^{3/2}$, so:

    $$
    \gamma_1(\bar{X}) = \frac{n \cdot \mu_3 / n^3}{(\sigma^2 / n)^{3/2}} = \frac{\mu_3}{n^2} \cdot \frac{n^{3/2}}{\sigma^3} = \frac{\mu_3}{\sigma^3 \sqrt{n}} = \frac{\gamma_1}{\sqrt{n}}
    $$

    For $\text{Exp}(1)$, $\gamma_1 = 2$, so $\gamma_1(\bar{X}) = 2/\sqrt{n}$.

    Setting $2/\sqrt{n} < 0.5$:

    $$
    \sqrt{n} > 4 \implies n > 16
    $$

    So $n \ge 17$ is needed for the skewness to drop below 0.5. $\square$

---

**Exercise 5.** Repeat the simulation with $n = 50$ instead of $n = 5$. Overlay a normal density $N(1, 1/50)$ on the histogram of sample means. Qualitatively describe the fit.

??? success "Solution to Exercise 5"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    np.random.seed(1)
    population = np.random.exponential(size=10_000)

    sample_means = [
        np.mean(np.random.choice(population, size=50, replace=False))
        for _ in range(10_000)
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(sample_means, bins=60, density=True, alpha=0.5, label="Simulated")

    x = np.linspace(min(sample_means), max(sample_means), 200)
    ax.plot(x, stats.norm.pdf(x, loc=1, scale=1/np.sqrt(50)), "r--", lw=2,
            label="N(1, 1/50)")
    ax.legend()
    ax.set_title("Sampling Distribution of X-bar, n=50")
    plt.show()
    ```

    With $n = 50$, the histogram of sample means is nearly symmetric and closely follows the $N(1, 1/50)$ density. The skewness of $\bar{X}$ is $2/\sqrt{50} \approx 0.28$, which is small enough that the normal approximation provides an excellent fit. $\square$
