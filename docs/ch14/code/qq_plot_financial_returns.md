# Quantile-Quantile Plot Financial Returns

## Overview

Financial asset returns are one of the most important practical settings where normality assumptions fail. This page demonstrates how Q-Q plots can diagnose non-normality in simulated daily returns, comparing normally distributed returns with heavy-tailed returns generated from a Student-$t$ distribution. The characteristic S-shaped Q-Q pattern reveals that real-world returns have fatter tails than the normal distribution predicts, with critical implications for risk management.

## Simulating Financial Returns

Daily log-returns are often modelled as

$$
r_t = \mu + \sigma\, \varepsilon_t,
$$

where $\mu$ is the expected daily return, $\sigma$ is the daily volatility, and $\varepsilon_t$ is a standardised innovation. Under the normal model, $\varepsilon_t \sim \mathcal{N}(0,1)$. Under a heavy-tailed model, $\varepsilon_t \sim t_\nu$ with $\nu$ degrees of freedom. Empirical studies of stock returns typically find $\nu$ in the range 3--8, indicating substantially heavier tails than the normal.

### Code

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n = 2000

# Normal returns: daily mean ~0.05%, volatility ~1.5%
normal_returns = np.random.normal(loc=0.0005, scale=0.015, size=n)

# Heavy-tailed returns: Student's t with df=6
heavy_returns = stats.t.rvs(df=6, loc=0.0005, scale=0.015, size=n)

for name, r in [("Normal", normal_returns), ("Heavy-tailed", heavy_returns)]:
    g1 = stats.skew(r)
    g2 = stats.kurtosis(r)
    _, p_jb = stats.jarque_bera(r)
    print(f"{name}: mean={r.mean():.6f}, std={r.std():.6f}, "
          f"skew={g1:.4f}, excess_kurt={g2:.4f}, JB p={p_jb:.6f}")
```

## Q-Q Plot Comparison

The diagnostic power of Q-Q plots becomes clear when we compare the two return series.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
normal_returns = np.random.normal(0.0005, 0.015, 2000)
heavy_returns = stats.t.rvs(df=6, loc=0.0005, scale=0.015, size=2000)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

stats.probplot(normal_returns, dist="norm", plot=axes[0])
axes[0].set_title("Q-Q Plot: Normal Returns")
axes[0].grid(True, alpha=0.3)

stats.probplot(heavy_returns, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot: Heavy-Tailed Returns")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Normal returns:** The points lie along the diagonal, confirming the distributional assumption.

**Heavy-tailed returns:** The Q-Q plot exhibits a characteristic S-shape. The lower-left points bend below the line (more extreme losses than predicted by the normal) and the upper-right points curve above the line (more extreme gains). This S-pattern is the hallmark of leptokurtic (fat-tailed) distributions.

## Tail Risk Implications

The discrepancy between normal and heavy-tailed models becomes critical at extreme quantiles. For a portfolio, the Value at Risk (VaR) at level $\alpha$ is

$$
\text{VaR}_\alpha = -\mu - \sigma\, q_\alpha,
$$

where $q_\alpha$ is the $\alpha$-quantile of the innovation distribution. For the normal, $q_{0.01} = \mathcal{N}^{-1}(0.01) = -2.326$. For $t_6$:

$$
q_{0.01}^{(t_6)} = t_6^{-1}(0.01) \approx -3.143.
$$

The heavy-tailed model predicts a 1% VaR that is roughly 35% larger. Under-estimating tail risk by using the normal model can lead to insufficient capital reserves.

## Interpretation

The Q-Q plot is arguably the single most important diagnostic for financial return data. The S-shaped pattern has been documented in equity returns, foreign exchange, commodities, and fixed income. Practitioners should:

- Always check normality with Q-Q plots before applying normal-based risk models.
- Consider the Jarque-Bera test as a formal companion diagnostic.
- Use heavy-tailed distributions ($t_\nu$, generalised hyperbolic) or non-parametric methods when the Q-Q plot departs from the diagonal.

## Exercises

**Exercise 1.** Simulate 2,000 normal returns and 2,000 $t_6$ returns. For each, compute the sample skewness, excess kurtosis, and Jarque-Bera $p$-value. Interpret the results.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(42)
    normal_r = np.random.normal(0.0005, 0.015, 2000)
    heavy_r = stats.t.rvs(df=6, loc=0.0005, scale=0.015, size=2000)

    for name, r in [("Normal", normal_r), ("t(6)", heavy_r)]:
        g1 = stats.skew(r, bias=False)
        g2 = stats.kurtosis(r, fisher=True, bias=False)
        _, p = stats.jarque_bera(r)
        print(f"{name}: skew={g1:.4f}, kurt={g2:.4f}, JB p={p:.6f}")
    ```

    The normal returns should have $g_1 \approx 0$, $g_2 \approx 0$, and a non-significant JB $p$-value. The $t_6$ returns should show $g_2 \gg 0$ (excess kurtosis around 1--3) and a very small JB $p$-value, confirming the heavier tails. $\square$

---

**Exercise 2.** Compute the empirical 1st, 5th, 95th, and 99th percentiles of both return series. Compare with the theoretical normal quantiles $\mu + \sigma\,\mathcal{N}^{-1}(p)$.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(42)
    normal_r = np.random.normal(0.0005, 0.015, 2000)
    heavy_r = stats.t.rvs(df=6, loc=0.0005, scale=0.015, size=2000)

    percentiles = [1, 5, 95, 99]
    mu, sigma = 0.0005, 0.015
    print(f"{'Pct':>4} {'Normal':>10} {'Heavy':>10} {'Theo Normal':>12}")
    for p in percentiles:
        n_q = np.percentile(normal_r, p)
        h_q = np.percentile(heavy_r, p)
        t_q = mu + sigma * stats.norm.ppf(p / 100)
        print(f"{p:>4} {n_q:>10.5f} {h_q:>10.5f} {t_q:>12.5f}")
    ```

    At the 1st and 99th percentiles, the heavy-tailed returns extend much further from the mean than both the normal returns and the theoretical normal quantiles. This quantifies the tail risk underestimation. $\square$

---

**Exercise 3.** Explain why the S-shaped Q-Q plot pattern arises for heavy-tailed data. Relate the shape to the relationship between $F^{-1}(p)$ for the $t$ distribution and $\mathcal{N}^{-1}(p)$ for the normal.

??? success "Solution to Exercise 3"

    On the Q-Q plot, the $x$-axis shows $\mathcal{N}^{-1}(p_i)$ (normal theoretical quantiles) and the $y$-axis shows $X_{(i)}$ (ordered data). If the data come from a $t_\nu$ distribution with heavier tails, then for small $p$ (left tail):

    $$
    t_\nu^{-1}(p) < \mathcal{N}^{-1}(p) \quad \text{(more negative)},
    $$

    so the data points fall *below* the reference line. For large $p$ (right tail):

    $$
    t_\nu^{-1}(p) > \mathcal{N}^{-1}(p) \quad \text{(more positive)},
    $$

    so the data points fall *above* the line. Near the centre ($p \approx 0.5$), both quantile functions are similar, so the points stay on the line. This creates the S-shape: concave in the left tail and convex in the right tail. The severity of the S-shape increases as $\nu$ decreases (heavier tails). $\square$

---

**Exercise 4.** Compute the theoretical excess kurtosis of a $t_\nu$ distribution as a function of $\nu$ (for $\nu > 4$). At what value of $\nu$ does the excess kurtosis drop below 1?

??? success "Solution to Exercise 4"

    The excess kurtosis of a $t_\nu$ distribution (for $\nu > 4$) is

    $$
    \gamma_2 = \frac{6}{\nu - 4}.
    $$

    Setting $\gamma_2 < 1$:

    $$
    \frac{6}{\nu - 4} < 1 \implies \nu - 4 > 6 \implies \nu > 10.
    $$

    So for $\nu > 10$, the excess kurtosis is below 1. At $\nu = 10$, $\gamma_2 = 1$ exactly. For $\nu \leq 4$, the fourth moment does not exist and $\gamma_2 = \infty$. In practice, financial returns typically exhibit $\gamma_2$ between 1 and 10, corresponding roughly to $t_\nu$ with $\nu \in [4, 10]$. $\square$

---

**Exercise 5.** A risk manager uses the normal distribution to compute the 1% Value at Risk. Show that the ratio of the $t_\nu$ VaR to the normal VaR at the 1% level approaches 1 as $\nu \to \infty$, and compute the ratio for $\nu = 5$.

??? success "Solution to Exercise 5"

    The VaR ratio at level $\alpha$ is

    $$
    R(\nu) = \frac{t_\nu^{-1}(\alpha)}{\mathcal{N}^{-1}(\alpha)}.
    $$

    As $\nu \to \infty$, $t_\nu \to \mathcal{N}(0,1)$, so $t_\nu^{-1}(\alpha) \to \mathcal{N}^{-1}(\alpha)$ and $R(\nu) \to 1$.

    For $\nu = 5$ and $\alpha = 0.01$:

    ```python
    from scipy import stats
    q_t5 = stats.t.ppf(0.01, df=5)
    q_norm = stats.norm.ppf(0.01)
    print(f"t(5) quantile: {q_t5:.4f}")
    print(f"Normal quantile: {q_norm:.4f}")
    print(f"Ratio: {q_t5 / q_norm:.4f}")
    ```

    $\mathcal{N}^{-1}(0.01) = -2.326$ and $t_5^{-1}(0.01) \approx -3.365$, giving $R(5) \approx 3.365/2.326 \approx 1.447$. The heavy-tailed VaR is about 45% larger than the normal VaR. A risk manager using the normal model would underestimate the true 1% loss by nearly half, potentially leading to catastrophic under-provisioning of capital. $\square$
