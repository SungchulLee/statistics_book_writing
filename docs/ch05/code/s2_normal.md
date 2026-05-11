# Sampling Distribution of S-squared (Normal)

## Overview

While the sampling distribution of $\bar{X}$ is central to estimation of the mean, the sampling distribution of the sample variance $S^2$ is essential for inference about variability. When the population is normal, the distribution of $S^2$ is exactly linked to the chi-squared distribution. For non-normal populations, the chi-squared relationship is only approximate, but improves with larger sample sizes. This page explores both the exact theory and simulation evidence.

## Definitions

Given a random sample $X_1, \ldots, X_n$ from a population with mean $\mu$ and variance $\sigma^2$, the **sample variance** is:

$$
S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2
$$

The factor $n - 1$ (rather than $n$) makes $S^2$ an unbiased estimator of $\sigma^2$:

$$
E[S^2] = \sigma^2
$$

## Exact Result for Normal Populations

!!! abstract "Theorem"
    If $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$, then:

    $$
    \frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)
    $$

    Moreover, $\bar{X}$ and $S^2$ are independent.

From this result, the distribution of $S^2$ itself can be written as:

$$
S^2 \sim \frac{\sigma^2}{n-1} \cdot \chi^2(n-1)
$$

The mean and variance of $S^2$ (for the normal case) are:

$$
E[S^2] = \sigma^2, \qquad \text{Var}(S^2) = \frac{2\sigma^4}{n-1}
$$

## Non-Normal Populations

For non-normal populations, the chi-squared relationship does not hold exactly. However:

- $S^2$ remains unbiased: $E[S^2] = \sigma^2$.
- The variance of $S^2$ depends on the population kurtosis $\kappa$:

$$
\text{Var}(S^2) = \frac{1}{n}\left(\kappa - \frac{n-3}{n-1}\right)\sigma^4
$$

where $\kappa = E[(X - \mu)^4] / \sigma^4$ is the kurtosis. For normal distributions, $\kappa = 3$, and this simplifies to $2\sigma^4/(n-1)$.

## Simulation

The code below simulates the sampling distribution of $S^2$ from four different populations (Normal, Exponential, Chi-squared, Uniform), each with $n = 100$, and overlays the theoretical chi-squared density.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

np.random.seed(1)

n_population = 10_000
n_sample = 100
n_sim = 1_000

# Build populations from different distributions
populations = {
    "Normal(0,1)":  stats.norm().rvs(n_population, random_state=1),
    "Exp(1)":       stats.expon().rvs(n_population, random_state=2),
    "Chi-sq(2)":    stats.chi2(df=2).rvs(n_population, random_state=3),
    "Uniform(0,1)": stats.uniform().rvs(n_population, random_state=4),
}

fig, axes = plt.subplots(1, len(populations), figsize=(16, 3.5))

for ax, (name, population) in zip(axes, populations.items()):
    # Simulate sampling distribution of S^2
    s2_sims = np.array([
        np.random.choice(population, size=n_sample, replace=False).var(ddof=1)
        for _ in range(n_sim)
    ])

    # Histogram
    _, bins, _ = ax.hist(s2_sims, density=True, bins=30,
                         alpha=0.5, edgecolor="white",
                         label=r"simulated $S^2$")

    # Theoretical chi-squared density scaled to S^2 units
    df = n_sample - 1
    sigma2 = population.var()
    c = df / sigma2
    x_grid = np.linspace(bins[0], bins[-1], 300)
    pdf = stats.chi2(df).pdf(x_grid * c) * c
    ax.plot(x_grid, pdf, "--r", lw=2, alpha=0.7,
            label=r"$\chi^2$-based PDF")

    ax.set_title(name)
    ax.set_xlabel(r"$S^2$")

axes[0].set_ylabel("Density")
axes[-1].legend(fontsize=8)
plt.tight_layout()
plt.show()
```

## Interpretation

!!! note "Key Observations"

    1. **Normal population**: The simulated distribution of $S^2$ matches the chi-squared-based density almost perfectly. This confirms the exact theoretical result.
    2. **Exponential and Chi-squared populations**: These are right-skewed, so the sampling distribution of $S^2$ is more spread out and the chi-squared overlay is only an approximation. The fit improves as $n$ increases.
    3. **Uniform population**: The Uniform distribution has light tails (kurtosis $< 3$), so $S^2$ has **less** variability than the chi-squared model predicts. The fit is reasonable but not exact.
    4. With $n = 100$, the chi-squared approximation is serviceable for all four populations, illustrating the CLT-like convergence for $S^2$.

## Exercises

**Exercise 1.** If $X_1, \ldots, X_{10} \overset{\text{iid}}{\sim} N(0, 4)$, find $P(S^2 > 6)$.

??? success "Solution to Exercise 1"
    Here $\sigma^2 = 4$, $n = 10$, so $\frac{(n-1)S^2}{\sigma^2} = \frac{9S^2}{4} \sim \chi^2(9)$.

    We need:

    $$
    P(S^2 > 6) = P\!\left(\frac{9S^2}{4} > \frac{9 \cdot 6}{4}\right) = P(\chi^2(9) > 13.5)
    $$

    Using Python:

    ```python
    from scipy import stats
    p = 1 - stats.chi2.cdf(13.5, df=9)
    print(f"P(S^2 > 6) = {p:.4f}")
    ```

    This gives $P(\chi^2(9) > 13.5) \approx 0.1415$. $\square$

---

**Exercise 2.** Prove that $E[S^2] = \sigma^2$ for any population (not just normal).

??? success "Solution to Exercise 2"
    Starting from the definition:

    $$
    S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2
    $$

    Expand the sum:

    $$
    \sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n X_i^2 - n\bar{X}^2
    $$

    Taking expectations:

    $$
    E\!\left[\sum_{i=1}^n X_i^2\right] = n E[X_i^2] = n(\sigma^2 + \mu^2)
    $$

    $$
    E[n\bar{X}^2] = n\!\left(\text{Var}(\bar{X}) + (E[\bar{X}])^2\right) = n\!\left(\frac{\sigma^2}{n} + \mu^2\right) = \sigma^2 + n\mu^2
    $$

    Therefore:

    $$
    E\!\left[\sum_{i=1}^n (X_i - \bar{X})^2\right] = n(\sigma^2 + \mu^2) - \sigma^2 - n\mu^2 = (n-1)\sigma^2
    $$

    Dividing by $n - 1$:

    $$
    E[S^2] = \sigma^2
    $$

    $\square$

---

**Exercise 3.** Show that $\text{Var}(S^2) = 2\sigma^4/(n-1)$ when the population is normal, using the fact that $\text{Var}(\chi^2(k)) = 2k$.

??? success "Solution to Exercise 3"
    From the theorem, $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)$.

    Let $Q = \frac{(n-1)S^2}{\sigma^2}$. Then $S^2 = \frac{\sigma^2 Q}{n-1}$ and:

    $$
    \text{Var}(S^2) = \frac{\sigma^4}{(n-1)^2} \cdot \text{Var}(Q) = \frac{\sigma^4}{(n-1)^2} \cdot 2(n-1) = \frac{2\sigma^4}{n-1}
    $$

    $\square$

---

**Exercise 4.** For an Exponential population with $\lambda = 1$, the kurtosis is $\kappa = 9$. Compute the theoretical $\text{Var}(S^2)$ for $n = 100$ and compare it with $2\sigma^4/(n-1)$ (the normal-theory value).

??? success "Solution to Exercise 4"
    For $\text{Exp}(1)$: $\sigma^2 = 1$, $\kappa = 9$.

    The general formula for the variance of $S^2$ is:

    $$
    \text{Var}(S^2) = \frac{1}{n}\left(\kappa - \frac{n-3}{n-1}\right)\sigma^4
    $$

    With $n = 100$:

    $$
    \text{Var}(S^2) = \frac{1}{100}\left(9 - \frac{97}{99}\right) \cdot 1 = \frac{1}{100}\left(9 - 0.9798\right) = \frac{8.0202}{100} = 0.08020
    $$

    The normal-theory value would be:

    $$
    \frac{2\sigma^4}{n-1} = \frac{2}{99} \approx 0.02020
    $$

    The Exponential population gives a variance of $S^2$ about 4 times larger than the normal-theory value. This is because the Exponential's heavier tails (excess kurtosis $= 6$) cause $S^2$ to be more variable. $\square$

---

**Exercise 5.** Construct a 95% confidence interval for $\sigma^2$ using the chi-squared distribution. Suppose $n = 25$ observations from a normal population yield $S^2 = 12$.

??? success "Solution to Exercise 5"
    Since $(n-1)S^2/\sigma^2 \sim \chi^2(n-1)$, we have:

    $$
    P\!\left(\chi^2_{0.025}(24) \le \frac{24 S^2}{\sigma^2} \le \chi^2_{0.975}(24)\right) = 0.95
    $$

    Rearranging for $\sigma^2$:

    $$
    \frac{24 S^2}{\chi^2_{0.975}(24)} \le \sigma^2 \le \frac{24 S^2}{\chi^2_{0.025}(24)}
    $$

    From chi-squared tables (or Python):

    ```python
    from scipy import stats
    lower_chi2 = stats.chi2.ppf(0.025, df=24)  # approximately 12.40
    upper_chi2 = stats.chi2.ppf(0.975, df=24)  # approximately 39.36
    ```

    $$
    \frac{24 \times 12}{39.36} \le \sigma^2 \le \frac{24 \times 12}{12.40}
    $$

    $$
    7.32 \le \sigma^2 \le 23.23
    $$

    The 95% confidence interval for $\sigma^2$ is approximately $(7.32, 23.23)$. Note that this interval is not symmetric around $S^2 = 12$ because the chi-squared distribution is skewed. $\square$
