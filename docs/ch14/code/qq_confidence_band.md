# Quantile-Quantile Plot Confidence Band Simulation

## Overview

A bare Q-Q plot can be difficult to interpret because sampling variability causes points to scatter around the reference line even under perfect normality. Pointwise confidence bands, constructed by simulation, provide a visual envelope: points within the band are consistent with the null hypothesis of normality, while points outside suggest genuine departures. This page explains the simulation-based construction and demonstrates it on skewed data.

## Construction via Parametric Bootstrap

Given an observed sample $x_1, \ldots, x_n$, the algorithm proceeds as follows.

1. **Estimate parameters.** Compute $\hat{\mu} = \bar{x}$ and $\hat{\sigma} = s$ (sample standard deviation with Bessel correction).

2. **Compute theoretical quantiles.** Using the plotting positions $p_i = (i - 0.5)/n$ for $i = 1, \ldots, n$, set

    $$
    q_i = \mathcal{N}^{-1}(p_i).
    $$

3. **Sort the observed data.** Let $x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(n)}$ denote the order statistics.

4. **Simulate under the null.** For $b = 1, \ldots, B$:
    - Draw $x_1^{*(b)}, \ldots, x_n^{*(b)} \overset{\text{iid}}{\sim} \mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$.
    - Sort to obtain the simulated order statistics $x_{(1)}^{*(b)} \leq \cdots \leq x_{(n)}^{*(b)}$.

5. **Compute the envelope.** For each rank $i$, take the 2.5th and 97.5th percentiles of $\{x_{(i)}^{*(1)}, \ldots, x_{(i)}^{*(B)}\}$:

    $$
    L_i = Q_{0.025}\bigl(x_{(i)}^{*(1)}, \ldots, x_{(i)}^{*(B)}\bigr), \qquad U_i = Q_{0.975}\bigl(x_{(i)}^{*(1)}, \ldots, x_{(i)}^{*(B)}\bigr).
    $$

6. **Plot.** Display $(q_i, x_{(i)})$ as scatter points, the fitted line $y = \hat{\mu} + \hat{\sigma}\, q$, and the shaded region $[L_i, U_i]$.

### Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def qq_with_band(x, B=800, seed=42):
    x = np.asarray(x, dtype=float)
    n = x.size
    mu, sd = x.mean(), x.std(ddof=1)

    # Theoretical quantiles
    p = (np.arange(1, n + 1) - 0.5) / n
    q_theor = stats.norm.ppf(p)

    # Observed order statistics
    x_sorted = np.sort(x)

    # Simulate under fitted normal
    rng = np.random.default_rng(seed)
    sims = np.sort(rng.normal(mu, sd, size=(B, n)), axis=1)
    lo = np.percentile(sims, 2.5, axis=0)
    hi = np.percentile(sims, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(q_theor, x_sorted, s=15)
    ax.plot(q_theor, mu + sd * q_theor, linestyle="--")
    ax.fill_between(q_theor, lo, hi, alpha=0.15,
                    label="95% pointwise band")
    ax.set_title("Q-Q Plot with Simulated 95% Band")
    ax.set_xlabel("Theoretical quantiles (Normal)")
    ax.set_ylabel("Ordered data")
    ax.legend()
    plt.tight_layout()
    plt.show()

rng = np.random.default_rng(123)
x = rng.lognormal(mean=0.0, sigma=0.6, size=300)
qq_with_band(x, B=600, seed=7)
```

## Pointwise vs Simultaneous Bands

The band described above is *pointwise*: for each individual rank $i$, there is a 95% probability that a normal order statistic falls within $[L_i, U_i]$. However, the probability that *all* $n$ points simultaneously fall within their respective intervals is less than 95%. A *simultaneous* band (analogous to a Bonferroni correction) would be wider. The pointwise band is nevertheless standard practice because it provides useful visual guidance without being overly conservative.

## Interpretation

For the lognormal example above, the upper-tail points will escape the confidence band, curving upward beyond the shaded region. This visually confirms the right skewness that formal tests would also detect. Data drawn from a true normal distribution should have most points (roughly 95% at each rank) within the band, with occasional excursions expected by chance.

## Exercises

**Exercise 1.** Generate $n = 200$ standard normal observations. Produce the Q-Q plot with a 95% pointwise band using $B = 1000$ simulations. Verify that nearly all points fall within the band.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=200)

    n = x.size
    mu, sd = x.mean(), x.std(ddof=1)
    p = (np.arange(1, n + 1) - 0.5) / n
    q = stats.norm.ppf(p)
    x_sorted = np.sort(x)

    sims = np.sort(rng.normal(mu, sd, size=(1000, n)), axis=1)
    lo = np.percentile(sims, 2.5, axis=0)
    hi = np.percentile(sims, 97.5, axis=0)

    outside = np.sum((x_sorted < lo) | (x_sorted > hi))
    print(f"Points outside band: {outside} / {n}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(q, x_sorted, s=15)
    ax.plot(q, mu + sd * q, linestyle="--")
    ax.fill_between(q, lo, hi, alpha=0.15, label="95% band")
    ax.legend()
    ax.set_title("Q-Q Plot with 95% Band (Normal Data)")
    plt.tight_layout()
    plt.show()
    ```

    Under normality, approximately $0.05 \times 200 = 10$ points are expected outside the band. The actual count will vary but should be in that neighbourhood. $\square$

---

**Exercise 2.** Repeat Exercise 1 with $n = 200$ observations from a $t_4$ distribution. Identify which portions of the Q-Q plot escape the band and explain why.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(1)
    x = rng.standard_t(df=4, size=200)

    n = x.size
    mu, sd = x.mean(), x.std(ddof=1)
    p = (np.arange(1, n + 1) - 0.5) / n
    q = stats.norm.ppf(p)
    x_sorted = np.sort(x)

    sims = np.sort(rng.normal(mu, sd, size=(1000, n)), axis=1)
    lo = np.percentile(sims, 2.5, axis=0)
    hi = np.percentile(sims, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(q, x_sorted, s=15)
    ax.plot(q, mu + sd * q, linestyle="--")
    ax.fill_between(q, lo, hi, alpha=0.15, label="95% band")
    ax.legend()
    ax.set_title("Q-Q Plot with 95% Band (t4 Data)")
    plt.tight_layout()
    plt.show()
    ```

    The $t_4$ distribution has heavier tails than the normal. On the Q-Q plot, the lowest order statistics fall below the lower edge of the band (more negative than expected) and the highest order statistics rise above the upper edge (more positive than expected), producing the characteristic S-shape outside the envelope. $\square$

---

**Exercise 3.** Explain mathematically why the confidence band is wider at the tails (extreme quantiles) than near the centre.

??? success "Solution to Exercise 3"

    The variance of the $i$-th order statistic from $\mathcal{N}(\mu, \sigma^2)$ is approximately

    $$
    \text{Var}(X_{(i)}) \approx \frac{p_i(1 - p_i)}{n\, [\phi(\mathcal{N}^{-1}(p_i))]^2}\, \sigma^2,
    $$

    where $p_i = i/(n+1)$ and $\phi$ is the standard normal density. Near the centre ($p_i \approx 0.5$), $\phi(\mathcal{N}^{-1}(0.5)) = \phi(0) = 1/\sqrt{2\pi}$ is at its maximum, making the denominator large and the variance small. In the tails ($p_i$ near 0 or 1), $\phi(\mathcal{N}^{-1}(p_i))$ becomes very small (the normal density decays rapidly), so the variance grows. Consequently, the simulated order statistics have larger spread in the tails, producing a wider confidence band. $\square$

---

**Exercise 4.** Modify the simulation to produce a 99% pointwise band. How does the width compare to the 95% band?

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=200)

    n = x.size
    mu, sd = x.mean(), x.std(ddof=1)
    p = (np.arange(1, n + 1) - 0.5) / n
    q = stats.norm.ppf(p)
    x_sorted = np.sort(x)

    sims = np.sort(rng.normal(mu, sd, size=(1000, n)), axis=1)
    lo95 = np.percentile(sims, 2.5, axis=0)
    hi95 = np.percentile(sims, 97.5, axis=0)
    lo99 = np.percentile(sims, 0.5, axis=0)
    hi99 = np.percentile(sims, 99.5, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(q, lo99, hi99, alpha=0.10, label="99% band")
    ax.fill_between(q, lo95, hi95, alpha=0.15, label="95% band")
    ax.scatter(q, x_sorted, s=15, zorder=3)
    ax.plot(q, mu + sd * q, linestyle="--")
    ax.legend()
    ax.set_title("95% vs 99% Pointwise Bands")
    plt.tight_layout()
    plt.show()
    ```

    The 99% band uses the 0.5th and 99.5th percentiles of the simulated order statistics, so it is wider than the 95% band at every rank. The ratio of widths is approximately $\mathcal{N}^{-1}(0.995)/\mathcal{N}^{-1}(0.975) \approx 2.576/1.960 \approx 1.31$, so the 99% band is roughly 31% wider. $\square$

---

**Exercise 5.** Describe how to convert the pointwise band into an approximate simultaneous band using a Bonferroni correction. What confidence level should each individual interval use if the nominal overall level is 95% and $n = 100$?

??? success "Solution to Exercise 5"

    For a simultaneous band at overall level $1 - \alpha$, the Bonferroni correction requires each of the $n$ pointwise intervals to have level $1 - \alpha/n$. With $\alpha = 0.05$ and $n = 100$, each interval must cover probability $1 - 0.05/100 = 0.9995$. The simulation would use the $0.025\%$ and $99.975\%$ percentiles of the simulated order statistics instead of $2.5\%$ and $97.5\%$. This produces a much wider band (the normal quantile changes from $z_{0.975} = 1.96$ to $z_{0.99975} \approx 3.48$). While conservative, it guarantees that under the null hypothesis all $n$ points fall within the band with at least 95% probability simultaneously. In practice, the Bonferroni band is overly conservative for large $n$; more refined simultaneous bands (e.g., based on the Kolmogorov-Smirnov distribution) are preferred. $\square$
