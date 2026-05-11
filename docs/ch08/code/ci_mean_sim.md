# Mean Confidence Interval Coverage Simulation

## Overview

This page explores the coverage properties of one-sample confidence intervals for a population mean $\mu$ through Monte Carlo simulation. Three methods are compared: the $z$-interval with known $\sigma$, the $z$-interval with the plug-in sample standard deviation $s$, and the $t$-interval. The simulation reveals why the $t$-interval is the correct default when $\sigma$ is unknown, especially for small sample sizes.

## Three Interval Methods

### z-Interval with Known Variance

When the population standard deviation $\sigma$ is known, the exact $(1-\alpha)100\%$ confidence interval is

$$
\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

### z-Interval with Plug-in s

A common teaching variant replaces $\sigma$ by the sample standard deviation $s$:

$$
\bar{X} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}
$$

This interval **under-covers** for small $n$ because it ignores the extra variability introduced by estimating $\sigma$ with $s$.

### t-Interval (Default in Practice)

The $t$-interval accounts for the estimation of $\sigma$:

$$
\bar{X} \pm t_{\alpha/2,\,n-1} \cdot \frac{s}{\sqrt{n}}
$$

Since $t_{\alpha/2,\,n-1} > z_{\alpha/2}$ for finite $n$, this interval is wider and achieves the nominal coverage.

## Finite Population Correction

When sampling without replacement from a finite population of size $N$, multiply the standard error by the finite population correction (FPC) factor:

$$
\text{FPC} = \sqrt{\frac{N - n}{N - 1}}
$$

This correction is negligible when $n \le 0.10 N$.

## Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

def finite_population_correction(n, N):
    """Return FPC factor if N is provided; else 1.0."""
    if N is None:
        return 1.0
    return float(np.sqrt((N - n) / (N - 1)))

def compute_intervals(xbar, s, n, alpha, method, sigma_known=None, N=None):
    """Compute CI lower/upper for z_known, z_plugin, or t method."""
    fpc = finite_population_correction(n, N)
    if method == "z_known":
        z_star = norm.ppf(1 - alpha / 2)
        se = sigma_known / np.sqrt(n) * fpc
        moe = z_star * se
    elif method == "z_plugin":
        z_star = norm.ppf(1 - alpha / 2)
        se = (s / np.sqrt(n)) * fpc
        moe = z_star * se
    else:  # t
        df = n - 1
        t_star = t.ppf(1 - alpha / 2, df=df)
        se = (s / np.sqrt(n)) * fpc
        moe = t_star * se
    return xbar - moe, xbar + moe

# Simulation parameters
rng = np.random.default_rng(42)
n_sim, n, mu, sigma, alpha = 100, 10, 0.0, 1.0, 0.05

X = rng.normal(loc=mu, scale=sigma, size=(n_sim, n))
xbar = X.mean(axis=1)
s = X.std(axis=1, ddof=1)

lower, upper = compute_intervals(xbar, s, n, alpha, method="t")
covered = (lower <= mu) & (mu <= upper)
coverage_pct = 100.0 * covered.mean()

print(f"t-interval coverage: {coverage_pct:.1f}%")
```

### Plotting the Intervals

```python
fig, ax = plt.subplots(figsize=(12, 12))
for i in range(n_sim):
    color = "k" if covered[i] else "r"
    ax.plot([lower[i], upper[i]], [i, i], lw=2, color=color)
    ax.plot(xbar[i], i, marker="o", ms=3, color=color)

ax.axvline(mu, linestyle="--", linewidth=1.5, color="r")
n_fail = int((~covered).sum())
ax.set_title(f"{n_sim} t CIs | n={n}, CL=95% | Fail={n_fail} (Coverage ~ {coverage_pct:.1f}%)")
ax.set_yticks([])
ax.set_xlabel("Mean value")
plt.tight_layout()
plt.show()
```

## Interpretation

- **z-known** achieves exactly $(1-\alpha)100\%$ coverage because the standard error involves no estimation.
- **z-plugin** uses $s$ in place of $\sigma$ but retains the normal critical value, so it under-covers for small $n$. The under-coverage is most pronounced for $n < 15$.
- **t-interval** compensates for the extra uncertainty in $s$ by using the heavier-tailed $t_{n-1}$ distribution. For all sample sizes, the empirical coverage stays close to the nominal level.
- For large $n$, all three methods converge because $t_{\alpha/2,\,n-1} \to z_{\alpha/2}$ and $s \xrightarrow{P} \sigma$.

## Exercises

**Exercise 1.** Run the simulation with $n = 5$ and $n_{\text{sim}} = 10{,}000$. Report the empirical coverage for each of the three methods at the 95 % confidence level. Which method(s) achieve the nominal rate?

??? success "Solution to Exercise 1"

    ```python
    rng = np.random.default_rng(0)
    n, n_sim, mu, sigma, alpha = 5, 10_000, 0.0, 1.0, 0.05
    X = rng.normal(mu, sigma, (n_sim, n))
    xbar = X.mean(axis=1)
    s = X.std(axis=1, ddof=1)

    for method in ["z_known", "z_plugin", "t"]:
        lo, hi = compute_intervals(xbar, s, n, alpha, method, sigma_known=sigma)
        cov = ((lo <= mu) & (mu <= hi)).mean()
        print(f"{method}: {100*cov:.1f}%")
    ```

    Typical output: **z_known** $\approx$ 95.0 %, **z_plugin** $\approx$ 92 %, **t** $\approx$ 95.0 %. Only the $z$-known and $t$ intervals achieve the nominal 95 %. The plug-in $z$ under-covers because $s$ is highly variable for $n = 5$. $\square$

---

**Exercise 2.** Explain mathematically why the plug-in $z$-interval under-covers for small $n$. Specifically, show that if $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$, then the pivot $(\bar{X} - \mu)/(s/\sqrt{n})$ does not follow $N(0,1)$.

??? success "Solution to Exercise 2"

    Under normality, $\bar{X} \sim N(\mu, \sigma^2/n)$ and $(n-1)s^2/\sigma^2 \sim \chi^2_{n-1}$, independently. The pivot

    $$
    T = \frac{\bar{X} - \mu}{s / \sqrt{n}} = \frac{(\bar{X} - \mu)/(\sigma/\sqrt{n})}{s/\sigma}
    = \frac{Z}{\sqrt{\chi^2_{n-1}/(n-1)}}
    $$

    where $Z \sim N(0,1)$. By definition, this ratio follows a $t_{n-1}$ distribution, not $N(0,1)$. The $t_{n-1}$ distribution has heavier tails than $N(0,1)$, so using $z_{\alpha/2}$ instead of $t_{\alpha/2,\,n-1}$ produces an interval that is too narrow, and the coverage probability $P(\mu \in \text{CI}) < 1 - \alpha$. $\square$

---

**Exercise 3.** Show that the finite population correction factor satisfies $\text{FPC} \to 1$ as $N \to \infty$ with $n$ fixed.

??? success "Solution to Exercise 3"

    $$
    \text{FPC} = \sqrt{\frac{N - n}{N - 1}} = \sqrt{\frac{1 - n/N}{1 - 1/N}}
    $$

    As $N \to \infty$ with $n$ fixed, $n/N \to 0$ and $1/N \to 0$, so

    $$
    \text{FPC} \to \sqrt{\frac{1 - 0}{1 - 0}} = 1
    $$

    Hence the correction has no effect for infinite (or very large) populations. $\square$

---

**Exercise 4.** Suppose the true population is not normal but instead follows an exponential distribution with rate $\lambda = 1$ (so $\mu = 1$, $\sigma = 1$). Design a simulation with $n = 10$ and $n_{\text{sim}} = 10{,}000$ to check whether the $t$-interval still achieves approximately 95 % coverage.

??? success "Solution to Exercise 4"

    ```python
    from scipy.stats import t as t_dist
    rng = np.random.default_rng(0)
    n, n_sim, mu, alpha = 10, 10_000, 1.0, 0.05
    covers = 0
    for _ in range(n_sim):
        sample = rng.exponential(scale=1.0, size=n)
        xbar = sample.mean()
        s = sample.std(ddof=1)
        t_crit = t_dist.ppf(1 - alpha / 2, df=n - 1)
        lo = xbar - t_crit * s / np.sqrt(n)
        hi = xbar + t_crit * s / np.sqrt(n)
        if lo <= mu <= hi:
            covers += 1
    print(f"Coverage: {100 * covers / n_sim:.1f}%")
    ```

    Typical result: coverage $\approx$ 91--93 %, below 95 %. The exponential distribution is strongly right-skewed, and for $n = 10$ the CLT approximation is not yet adequate. With $n = 30$ the coverage improves to $\approx$ 94 %, and by $n = 100$ it is close to 95 %. $\square$

---

**Exercise 5.** A quality inspector samples $n = 50$ parts from a production run of $N = 400$ parts. Compute the FPC factor and explain its practical effect on the width of the confidence interval for the mean part weight.

??? success "Solution to Exercise 5"

    $$
    \text{FPC} = \sqrt{\frac{400 - 50}{400 - 1}} = \sqrt{\frac{350}{399}} = \sqrt{0.8772} \approx 0.9366
    $$

    The standard error is multiplied by 0.9366, reducing it by about 6.3 %. The confidence interval becomes narrower because sampling 50 out of 400 (12.5 %) exhausts a non-trivial fraction of the population, so there is less uncertainty about the remaining units than simple random sampling with replacement would suggest. Note that $n/N = 0.125 > 0.10$, confirming that the FPC should not be ignored here. $\square$
