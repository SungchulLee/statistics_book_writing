# Chi-Square Distribution

## Overview

The chi-square distribution is one of the most fundamental distributions in statistical inference. It arises naturally as the distribution of a sum of squared independent standard normal random variables. The chi-square distribution underpins tests for variance, goodness-of-fit tests, and tests of independence in contingency tables, and it appears in the sampling distribution of the sample variance from a normal population.

## Definition

If $Z_1, Z_2, \ldots, Z_d$ are independent standard normal random variables, then

$$
Q = \sum_{i=1}^{d} Z_i^2 \sim \chi^2(d),
$$

where $d$ is the degrees-of-freedom parameter.

## Properties

The mean and variance of a $\chi^2(d)$ random variable are

$$
E[Q] = d, \qquad \operatorname{Var}(Q) = 2d.
$$

The probability density function for $x > 0$ is

$$
f(x; d) = \frac{1}{2^{d/2}\,\Gamma(d/2)}\, x^{d/2 - 1}\, e^{-x/2}.
$$

Key additional properties:

- **Additivity**: If $Q_1 \sim \chi^2(d_1)$ and $Q_2 \sim \chi^2(d_2)$ are independent, then $Q_1 + Q_2 \sim \chi^2(d_1 + d_2)$.
- **Relation to sample variance**: If $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$, then $(n-1)S^2/\sigma^2 \sim \chi^2(n-1)$.
- **CLT approximation**: For large $d$, $\chi^2(d) \approx N(d, 2d)$.

## Code

### PDF and CDF

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

df = 5
x = np.linspace(0, 30, 300)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(x, stats.chi2(df=df).pdf(x), label="PDF")
ax.plot(x, stats.chi2(df=df).cdf(x), label="CDF")
ax.legend()
ax.set_title(f"PDF and CDF of chi-squared({df})")
plt.tight_layout()
plt.show()
```

### Sampling and Construction from Normals

The following code verifies the definition by comparing direct sampling from $\chi^2(d)$ with the sum of $d$ squared standard normals:

```python
df, seed = 5, 1

# Direct sampling
data_direct = stats.chi2(df=df).rvs(10_000, random_state=seed)

# Construction: sum of d squared standard normals
z = stats.norm().rvs(size=(df, 10_000), random_state=seed)
data_constructed = np.sum(z ** 2, axis=0)

# Both histograms should match the theoretical PDF
bins = np.linspace(0, 25, 80)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
for ax, data, title in [
    (ax1, data_direct, "Direct sampling"),
    (ax2, data_constructed, "Sum of Z^2 construction"),
]:
    ax.hist(data, bins=bins, density=True, alpha=0.7)
    ax.plot(bins, stats.chi2(df=df).pdf(bins), "r--", lw=2)
    ax.set_title(title)
plt.tight_layout()
plt.show()
```

## Interpretation

- As the degrees of freedom $d$ increase, the distribution shifts right and becomes more symmetric, approaching a normal distribution.
- The mode of $\chi^2(d)$ is $\max(d - 2, 0)$, so for small $d$ the distribution is heavily right-skewed.
- The construction from squared normals provides geometric intuition: $Q$ measures the squared distance of a $d$-dimensional standard normal vector from the origin.

## Exercises

**Exercise 1.** Let $Q \sim \chi^2(10)$. Compute $P(Q > 18.307)$ and $P(3.247 < Q < 20.483)$ using Python.

??? success "Solution to Exercise 1"

    ```python
    import scipy.stats as stats

    rv = stats.chi2(df=10)
    p1 = rv.sf(18.307)
    p2 = rv.cdf(20.483) - rv.cdf(3.247)
    print(f"P(Q > 18.307) = {p1:.4f}")
    print(f"P(3.247 < Q < 20.483) = {p2:.4f}")
    ```

    $P(Q > 18.307) \approx 0.05$ (this is the $\chi^2_{0.95}(10)$ critical value), and $P(3.247 < Q < 20.483) \approx 0.95$ (a 95% central interval).

---

**Exercise 2.** Prove the additivity property: if $Q_1 \sim \chi^2(d_1)$ and $Q_2 \sim \chi^2(d_2)$ are independent, then $Q_1 + Q_2 \sim \chi^2(d_1 + d_2)$.

??? success "Solution to Exercise 2"

    Write $Q_1 = \sum_{i=1}^{d_1} Z_i^2$ and $Q_2 = \sum_{j=1}^{d_2} W_j^2$ where all $Z_i, W_j \overset{\text{iid}}{\sim} N(0,1)$ are independent (by the independence of $Q_1$ and $Q_2$). Then

    $$
    Q_1 + Q_2 = \sum_{i=1}^{d_1} Z_i^2 + \sum_{j=1}^{d_2} W_j^2 = \sum_{l=1}^{d_1 + d_2} U_l^2,
    $$

    which is the sum of $d_1 + d_2$ independent squared standard normals, hence $\chi^2(d_1 + d_2)$ by definition. $\square$

---

**Exercise 3.** Write a simulation that draws $n = 50$ samples from $N(3, 4)$ (i.e., $\mu = 3$, $\sigma^2 = 4$), computes $(n-1)S^2/\sigma^2$, and repeats 10,000 times. Plot the histogram and overlay the $\chi^2(49)$ PDF to verify the theoretical result.

??? success "Solution to Exercise 3"

    ```python
    import numpy as np
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    n, mu, sigma2 = 50, 3, 4
    n_sims = 10000

    chi2_stats = []
    for _ in range(n_sims):
        x = rng.normal(mu, np.sqrt(sigma2), n)
        s2 = np.var(x, ddof=1)
        chi2_stats.append((n - 1) * s2 / sigma2)

    chi2_stats = np.array(chi2_stats)
    bins = np.linspace(20, 80, 80)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(chi2_stats, bins=bins, density=True, alpha=0.7, label="Simulated")
    ax.plot(bins, stats.chi2(df=n - 1).pdf(bins), "r--", lw=2, label="chi2(49) PDF")
    ax.legend()
    ax.set_title("Sampling distribution of (n-1)S^2/sigma^2")
    plt.tight_layout()
    plt.show()
    ```

    The histogram will closely match the $\chi^2(49)$ curve, confirming the theory.

---

**Exercise 4.** Derive the moment generating function (MGF) of $\chi^2(d)$ and use it to obtain $E[Q]$ and $\operatorname{Var}(Q)$.

??? success "Solution to Exercise 4"

    If $Z \sim N(0,1)$, its MGF is $M_Z(t) = (1-2t)^{-1/2}$ for $t < 1/2$. Since $Q = \sum_{i=1}^d Z_i^2$ with independent $Z_i$,

    $$
    M_Q(t) = \prod_{i=1}^d M_{Z_i^2}(t) = (1 - 2t)^{-d/2}, \quad t < \tfrac{1}{2}.
    $$

    Taking derivatives:

    $$
    M_Q'(t) = d(1-2t)^{-d/2 - 1}, \quad E[Q] = M_Q'(0) = d.
    $$

    $$
    M_Q''(t) = d(d+2)(1-2t)^{-d/2 - 2}, \quad E[Q^2] = M_Q''(0) = d(d+2).
    $$

    Therefore

    $$
    \operatorname{Var}(Q) = E[Q^2] - (E[Q])^2 = d(d+2) - d^2 = 2d. \quad \square
    $$

---

**Exercise 5.** For $d = 1, 5, 10, 30, 100$, compute the skewness of $\chi^2(d)$ using the formula $\gamma_1 = \sqrt{8/d}$ and verify numerically using samples. Explain why the normal approximation improves with increasing $d$.

??? success "Solution to Exercise 5"

    ```python
    import numpy as np
    import scipy.stats as stats

    for d in [1, 5, 10, 30, 100]:
        theoretical_skew = np.sqrt(8 / d)
        samples = stats.chi2(df=d).rvs(100000, random_state=42)
        empirical_skew = stats.skew(samples)
        print(f"df={d:3d}: theoretical={theoretical_skew:.4f}, "
              f"empirical={empirical_skew:.4f}")
    ```

    The skewness $\gamma_1 = \sqrt{8/d}$ decreases as $d$ grows: for $d = 1$ it is $2\sqrt{2} \approx 2.83$, but for $d = 100$ it is only $0.283$. By the central limit theorem, $Q = \sum Z_i^2$ is a sum of i.i.d. terms, so its standardized distribution converges to $N(0,1)$. In practice, the normal approximation $\chi^2(d) \approx N(d, 2d)$ is already quite accurate for $d \gtrsim 30$.
