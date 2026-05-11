# Sampling Distribution of X-bar (Uniform)

## Overview

When we repeatedly draw random samples of size $n$ from a population and compute the sample mean $\bar{X}$ each time, the resulting distribution of those means is called the **sampling distribution of $\bar{X}$**. This page explores this concept using a Uniform population. Even though the population distribution is flat, the sampling distribution of $\bar{X}$ concentrates around the population mean and, by the Central Limit Theorem, becomes approximately normal as $n$ grows.

## Population Model

Suppose the population follows a continuous Uniform distribution on the interval $[0, 1]$:

$$
X \sim \text{Uniform}(0, 1)
$$

The population mean and variance are:

$$
\mu = E[X] = \frac{1}{2}, \qquad \sigma^2 = \text{Var}(X) = \frac{1}{12}
$$

## Sampling Distribution of the Sample Mean

For a random sample $X_1, X_2, \ldots, X_n$ drawn independently from this population, the sample mean is:

$$
\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i
$$

By the properties of expectation and variance for independent random variables:

$$
E[\bar{X}] = \mu = \frac{1}{2}, \qquad \text{Var}(\bar{X}) = \frac{\sigma^2}{n} = \frac{1}{12n}
$$

By the **Central Limit Theorem**, for large $n$:

$$
\bar{X} \;\dot{\sim}\; N\!\left(\frac{1}{2},\; \frac{1}{12n}\right)
$$

## Simulation

The following code draws 10,000 samples of size $n = 5$ from a Uniform(0, 1) population and plots three distributions side by side: the population, a single sample, and the sampling distribution of $\bar{X}$.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

sample_size = 5
n_samples = 10_000
n_population = 10_000

# Generate a large population from Uniform(0, 1)
population = np.random.uniform(size=(n_population,))

# Draw a single sample
single_sample = np.random.choice(population, size=sample_size, replace=False)

# Simulate the sampling distribution of X-bar
sample_means = [
    np.mean(np.random.choice(population, size=sample_size, replace=False))
    for _ in range(n_samples)
]

# Plot
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

ax0.hist(population, bins=np.linspace(0, 1, 100))
ax0.set_title("Population Distribution")

ax1.scatter(single_sample, np.zeros_like(single_sample), s=100)
ax1.set_title(f"Sample Distribution (n = {sample_size})")

ax2.hist(sample_means, bins=np.linspace(0, 1, 100))
ax2.set_title("Sampling Distribution of X-bar")

plt.tight_layout()
plt.show()
```

## Interpretation

!!! note "Key Observations"

    1. The **population distribution** is flat (uniform) across $[0, 1]$.
    2. A **single sample** of size 5 is just a handful of scattered points -- it cannot reveal the shape of the population on its own.
    3. The **sampling distribution** of $\bar{X}$ is bell-shaped and centred at $\mu = 0.5$, even though the population is not bell-shaped. This is the Central Limit Theorem in action.
    4. The spread of the sampling distribution is narrower than the population distribution by a factor of $1/\sqrt{n}$.

The standard error of $\bar{X}$ for $n = 5$ is:

$$
\text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}} = \frac{1/\sqrt{12}}{\sqrt{5}} \approx 0.129
$$

As $n$ increases, this standard error shrinks, and the sampling distribution concentrates more tightly around $\mu$.

## Exercises

**Exercise 1.** For $X \sim \text{Uniform}(0, 1)$, derive $E[X]$ and $\text{Var}(X)$ from first principles using integration.

??? success "Solution to Exercise 1"
    The pdf of $X \sim \text{Uniform}(0, 1)$ is $f(x) = 1$ for $x \in [0, 1]$.

    $$
    E[X] = \int_0^1 x \cdot 1 \, dx = \left[\frac{x^2}{2}\right]_0^1 = \frac{1}{2}
    $$

    $$
    E[X^2] = \int_0^1 x^2 \cdot 1 \, dx = \left[\frac{x^3}{3}\right]_0^1 = \frac{1}{3}
    $$

    $$
    \text{Var}(X) = E[X^2] - (E[X])^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}
    $$

    $\square$

---

**Exercise 2.** Prove that $E[\bar{X}] = \mu$ and $\text{Var}(\bar{X}) = \sigma^2 / n$ for any population with mean $\mu$ and variance $\sigma^2$, assuming the observations are independent.

??? success "Solution to Exercise 2"
    Let $X_1, \ldots, X_n$ be i.i.d. with $E[X_i] = \mu$ and $\text{Var}(X_i) = \sigma^2$.

    $$
    E[\bar{X}] = E\!\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \frac{1}{n} \cdot n\mu = \mu
    $$

    By independence:

    $$
    \text{Var}(\bar{X}) = \text{Var}\!\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}
    $$

    $\square$

---

**Exercise 3.** If you increase the sample size from $n = 5$ to $n = 20$, by what factor does the standard error of $\bar{X}$ decrease? What sample size would you need to cut the standard error in half relative to $n = 5$?

??? success "Solution to Exercise 3"
    The standard error is $\text{SE} = \sigma / \sqrt{n}$.

    The ratio of standard errors is:

    $$
    \frac{\text{SE}(n=5)}{\text{SE}(n=20)} = \frac{\sigma/\sqrt{5}}{\sigma/\sqrt{20}} = \sqrt{\frac{20}{5}} = \sqrt{4} = 2
    $$

    So the standard error decreases by a factor of 2 when going from $n = 5$ to $n = 20$.

    To cut the standard error in half relative to $n = 5$, we need:

    $$
    \frac{\sigma}{\sqrt{n}} = \frac{1}{2} \cdot \frac{\sigma}{\sqrt{5}} \implies \sqrt{n} = 2\sqrt{5} \implies n = 20
    $$

    $\square$

---

**Exercise 4.** Modify the simulation to use $n = 50$ instead of $n = 5$. Compute the theoretical standard error and compare it to the empirical standard deviation of the 10,000 simulated means.

??? success "Solution to Exercise 4"
    The theoretical standard error for $n = 50$ is:

    $$
    \text{SE} = \frac{1/\sqrt{12}}{\sqrt{50}} = \frac{1}{\sqrt{600}} \approx 0.0408
    $$

    In code:

    ```python
    import numpy as np
    np.random.seed(1)

    population = np.random.uniform(size=10_000)
    sample_means = [
        np.mean(np.random.choice(population, size=50, replace=False))
        for _ in range(10_000)
    ]
    empirical_se = np.std(sample_means)
    print(f"Theoretical SE: {1 / np.sqrt(600):.4f}")
    print(f"Empirical SE:   {empirical_se:.4f}")
    ```

    The empirical standard error should be close to 0.0408, confirming the theoretical formula. $\square$

---

**Exercise 5.** The Irwin--Hall distribution is the distribution of the sum $S_n = X_1 + X_2 + \cdots + X_n$ where $X_i \sim \text{Uniform}(0,1)$ independently. Show that $\bar{X} = S_n / n$ and use the Irwin--Hall pdf for $n = 2$ to find the exact pdf of $\bar{X}$ when $n = 2$.

??? success "Solution to Exercise 5"
    By definition, $\bar{X} = S_n / n$. For $n = 2$, the Irwin--Hall pdf of $S_2 = X_1 + X_2$ is the triangular distribution:

    $$
    f_{S_2}(s) = \begin{cases} s & 0 \le s \le 1 \\ 2 - s & 1 < s \le 2 \\ 0 & \text{otherwise} \end{cases}
    $$

    Since $\bar{X} = S_2 / 2$, we apply the change-of-variables formula. Let $y = s/2$, so $s = 2y$ and $ds = 2\,dy$:

    $$
    f_{\bar{X}}(y) = f_{S_2}(2y) \cdot 2 = \begin{cases} 4y & 0 \le y \le \tfrac{1}{2} \\ 4(1 - y) & \tfrac{1}{2} < y \le 1 \\ 0 & \text{otherwise} \end{cases}
    $$

    This is a symmetric triangular distribution on $[0, 1]$ with peak at $y = 1/2$, confirming that even for $n = 2$ the sampling distribution of $\bar{X}$ is already unimodal and symmetric (though not yet normal). $\square$
