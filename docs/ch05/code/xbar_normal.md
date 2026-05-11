# Sampling Distribution of X-bar (Normal)

## Overview

When the population itself is normally distributed, the sampling distribution of the sample mean $\bar{X}$ is **exactly** normal for every sample size $n$ -- no Central Limit Theorem approximation is needed. This page demonstrates this exact result through theory and simulation. The normal-population case serves as the foundation for many classical inference procedures, including $t$-tests and confidence intervals.

## Population Model

Suppose the population follows a standard Normal distribution:

$$
X \sim N(\mu, \sigma^2) = N(0, 1)
$$

with $\mu = 0$ and $\sigma^2 = 1$.

## Exact Sampling Distribution

For a random sample $X_1, X_2, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$, the sample mean has the exact distribution:

$$
\bar{X} \sim N\!\left(\mu, \frac{\sigma^2}{n}\right)
$$

!!! info "Why This is Exact"
    A linear combination of independent normal random variables is itself normal. Since $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ is a linear combination of i.i.d. normal random variables, $\bar{X}$ is exactly normal. No asymptotic argument is required.

The standardized version is:

$$
Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim N(0, 1)
$$

## Simulation

The following code draws 10,000 samples of size $n = 5$ from a $N(0, 1)$ population and compares the population, a single sample, and the sampling distribution of $\bar{X}$.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

sample_size = 5
n_samples = 10_000
n_population = 10_000

# Generate a large population from N(0, 1)
population = np.random.normal(loc=0, scale=1, size=n_population)

# Draw a single sample
single_sample = np.random.choice(population, size=sample_size, replace=False)

# Simulate the sampling distribution of X-bar
sample_means = [
    np.mean(np.random.choice(population, size=sample_size, replace=False))
    for _ in range(n_samples)
]

# Plot
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

ax0.hist(population, bins=100, edgecolor="white")
ax0.set_title("Population Distribution N(0, 1)")

ax1.scatter(single_sample, np.zeros_like(single_sample), s=100)
ax1.set_title(f"Sample Distribution (n = {sample_size})")

ax2.hist(sample_means, bins=100, edgecolor="white")
ax2.set_title("Sampling Distribution of X-bar")

plt.tight_layout()
plt.show()
```

## Interpretation

!!! note "Key Observations"

    1. The **population distribution** is bell-shaped (normal).
    2. The **sampling distribution** of $\bar{X}$ is also bell-shaped, centred at the same mean $\mu = 0$.
    3. The sampling distribution is **narrower** than the population by a factor of $\sqrt{n}$. For $n = 5$, the standard error is $\sigma/\sqrt{5} \approx 0.447$ compared to $\sigma = 1$.
    4. Unlike the Uniform or Exponential cases, the normality of the sampling distribution here is **exact**, not approximate.

### Comparison of Spreads

| Distribution | Standard deviation |
|---|---|
| Population $X$ | $\sigma = 1$ |
| $\bar{X}$ with $n = 5$ | $\sigma / \sqrt{5} \approx 0.447$ |
| $\bar{X}$ with $n = 25$ | $\sigma / \sqrt{25} = 0.200$ |
| $\bar{X}$ with $n = 100$ | $\sigma / \sqrt{100} = 0.100$ |

## Proof of Exact Normality

!!! abstract "Theorem"
    If $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$, then $\bar{X} \sim N(\mu, \sigma^2/n)$.

**Proof.** The moment generating function (MGF) of $X_i$ is:

$$
M_{X_i}(t) = \exp\!\left(\mu t + \frac{\sigma^2 t^2}{2}\right)
$$

Since the $X_i$ are independent:

$$
M_{S_n}(t) = \prod_{i=1}^n M_{X_i}(t) = \exp\!\left(n\mu t + \frac{n\sigma^2 t^2}{2}\right)
$$

The MGF of $\bar{X} = S_n / n$ is:

$$
M_{\bar{X}}(t) = M_{S_n}(t/n) = \exp\!\left(\mu t + \frac{\sigma^2 t^2}{2n}\right)
$$

This is the MGF of $N(\mu, \sigma^2/n)$. Since the MGF uniquely determines the distribution, $\bar{X} \sim N(\mu, \sigma^2/n)$. $\square$

## Exercises

**Exercise 1.** If $X_1, \ldots, X_{25} \overset{\text{iid}}{\sim} N(100, 16)$, find $P(\bar{X} > 102)$.

??? success "Solution to Exercise 1"
    Here $\mu = 100$, $\sigma^2 = 16$, $n = 25$.

    $$
    \bar{X} \sim N\!\left(100, \frac{16}{25}\right) = N(100,\; 0.64)
    $$

    Standardize:

    $$
    Z = \frac{102 - 100}{\sqrt{0.64}} = \frac{2}{0.8} = 2.5
    $$

    $$
    P(\bar{X} > 102) = P(Z > 2.5) = 1 - \mathcal{N}(2.5) \approx 1 - 0.9938 = 0.0062
    $$

    $\square$

---

**Exercise 2.** Prove that if $X \sim N(\mu_X, \sigma_X^2)$ and $Y \sim N(\mu_Y, \sigma_Y^2)$ are independent, then $aX + bY \sim N(a\mu_X + b\mu_Y,\; a^2\sigma_X^2 + b^2\sigma_Y^2)$.

??? success "Solution to Exercise 2"
    Let $W = aX + bY$. The MGF of $W$ is:

    $$
    M_W(t) = E[e^{t(aX + bY)}] = E[e^{taX}] \cdot E[e^{tbY}]
    $$

    where the factorisation uses independence. Substituting the normal MGFs:

    $$
    M_W(t) = \exp\!\left(a\mu_X t + \frac{a^2\sigma_X^2 t^2}{2}\right) \cdot \exp\!\left(b\mu_Y t + \frac{b^2\sigma_Y^2 t^2}{2}\right)
    $$

    $$
    = \exp\!\left((a\mu_X + b\mu_Y)t + \frac{(a^2\sigma_X^2 + b^2\sigma_Y^2)t^2}{2}\right)
    $$

    This is the MGF of $N(a\mu_X + b\mu_Y,\; a^2\sigma_X^2 + b^2\sigma_Y^2)$. $\square$

---

**Exercise 3.** A machine fills bottles to a mean of 500 ml with a standard deviation of 4 ml, and the fill amounts are normally distributed. Quality control samples 16 bottles. What is the probability that the sample mean is within 2 ml of the target?

??? success "Solution to Exercise 3"
    Let $X_i \sim N(500, 16)$ with $n = 16$.

    $$
    \bar{X} \sim N\!\left(500, \frac{16}{16}\right) = N(500, 1)
    $$

    We need $P(498 < \bar{X} < 502) = P(-2 < Z < 2)$ where $Z = (\bar{X} - 500)/1$.

    $$
    P(-2 < Z < 2) = \mathcal{N}(2) - \mathcal{N}(-2) = 2\mathcal{N}(2) - 1 \approx 2(0.9772) - 1 = 0.9544
    $$

    There is approximately a 95.44% probability that the sample mean is within 2 ml of the target. $\square$

---

**Exercise 4.** Explain why the Central Limit Theorem is not needed when the population is normal, but is essential when the population is Exponential or Uniform. What changes about the sampling distribution in the non-normal case as $n$ increases?

??? success "Solution to Exercise 4"
    When the population is normal, $\bar{X}$ is exactly normal for every $n$ because a linear combination of independent normal random variables is normal. This is a direct property of the normal distribution's MGF (or equivalently, its closure under convolution).

    For non-normal populations (e.g., Exponential, Uniform), $\bar{X}$ is **not** exactly normal for finite $n$. Its distribution depends on $n$ and the specific population shape. However, as $n \to \infty$, the CLT guarantees that the standardized $\bar{X}$ converges in distribution to $N(0,1)$.

    As $n$ increases for a non-normal population:

    - The sampling distribution becomes more symmetric (skewness decreases as $\gamma_1/\sqrt{n}$).
    - The excess kurtosis shrinks toward 0 (at rate $1/n$).
    - The distribution becomes progressively closer to normal in shape.

    The rate of convergence depends on how "non-normal" the population is; heavily skewed or heavy-tailed populations require larger $n$. $\square$

---

**Exercise 5.** Using the simulation code above, increase $n$ from 5 to 100. Overlay the theoretical density $N(0, 1/100)$ on the histogram. Verify that the empirical standard deviation of the 10,000 sample means is close to $1/\sqrt{100} = 0.1$.

??? success "Solution to Exercise 5"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    np.random.seed(1)
    population = np.random.normal(loc=0, scale=1, size=10_000)

    sample_means = [
        np.mean(np.random.choice(population, size=100, replace=False))
        for _ in range(10_000)
    ]

    empirical_se = np.std(sample_means)
    theoretical_se = 1 / np.sqrt(100)

    print(f"Theoretical SE: {theoretical_se:.4f}")
    print(f"Empirical SE:   {empirical_se:.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(sample_means, bins=60, density=True, alpha=0.5, label="Simulated")
    x = np.linspace(-0.5, 0.5, 200)
    ax.plot(x, stats.norm.pdf(x, 0, theoretical_se), "r--", lw=2,
            label=f"N(0, {theoretical_se**2:.4f})")
    ax.legend()
    ax.set_title("Sampling Distribution of X-bar (n = 100)")
    plt.show()
    ```

    The empirical standard error will be close to 0.1, and the histogram will match the $N(0, 0.01)$ density essentially perfectly, confirming the exact normality result. $\square$
