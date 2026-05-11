# Variance Confidence Interval Coverage Simulation

## Overview

This page demonstrates the chi-squared confidence interval for the population variance $\sigma^2$ and evaluates its coverage through simulation. The chi-squared variance interval is exact under normality but can fail when the underlying population is skewed or heavy-tailed. The simulation confirms the nominal coverage for normal data and highlights the sensitivity to the normality assumption.

## Chi-Squared Confidence Interval for Variance

### Pivotal Quantity

If $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$, then the statistic

$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}
$$

where $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$ is the unbiased sample variance.

### Confidence Interval

Inverting the pivot yields the $(1-\alpha)100\%$ CI for $\sigma^2$:

$$
\left(\frac{(n-1)S^2}{\chi^2_{1-\alpha/2,\,n-1}},\;\;
      \frac{(n-1)S^2}{\chi^2_{\alpha/2,\,n-1}}\right)
$$

Note that the **larger** chi-squared quantile appears in the **lower** endpoint because dividing by a larger number produces a smaller result.

A confidence interval for $\sigma$ is obtained by taking square roots:

$$
\left(\sqrt{\frac{(n-1)S^2}{\chi^2_{1-\alpha/2,\,n-1}}},\;\;
      \sqrt{\frac{(n-1)S^2}{\chi^2_{\alpha/2,\,n-1}}}\right)
$$

### Critical Normality Assumption

This interval is **exact only when the data are normally distributed**. Unlike confidence intervals for the mean (which benefit from the Central Limit Theorem), the chi-squared variance interval does not become robust to non-normality even for large $n$. For skewed or heavy-tailed data, consider a bootstrap confidence interval for $\sigma^2$.

## Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

n_simulations = 100
n_samples = 12
mu, sigma = 0.0, 2.0
alpha = 0.05

true_var = sigma**2
df = n_samples - 1
chi2_lo = chi2.ppf(alpha / 2, df=df)
chi2_hi = chi2.ppf(1 - alpha / 2, df=df)

lowers = np.empty(n_simulations)
uppers = np.empty(n_simulations)
centers = np.empty(n_simulations)

for i in range(n_simulations):
    x = np.random.normal(loc=mu, scale=sigma, size=n_samples)
    s2 = x.var(ddof=1)
    lowers[i] = df * s2 / chi2_hi
    uppers[i] = df * s2 / chi2_lo
    centers[i] = s2

covered = (lowers <= true_var) & (true_var <= uppers)
n_fail = int((~covered).sum())
coverage_pct = 100.0 * covered.mean()
print(f"Coverage: {coverage_pct:.1f}%, Failures: {n_fail}")
```

### Plotting the Intervals

```python
fig, ax = plt.subplots(figsize=(12, 12))
for i in range(n_simulations):
    color = "k" if covered[i] else "r"
    ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
    ax.plot(centers[i], i, marker="o", ms=3, color=color)

ax.axvline(true_var, linestyle="--", linewidth=1.5, color="r")
ax.set_title(f"{n_simulations} Chi-square Variance CIs | n={n_samples}, CL=95%")
ax.set_yticks([])
ax.set_xlabel("Variance")
plt.tight_layout()
plt.show()
```

## Interpretation

- When the data truly come from a normal distribution, the chi-squared interval achieves the stated coverage level. The simulation confirms empirical coverage close to 95 % for $n = 12$.
- The intervals are **asymmetric** because the chi-squared distribution is right-skewed. The upper bound tends to be farther from $S^2$ than the lower bound.
- If the population is non-normal (e.g., exponential, $t$ with low df, or log-normal), the chi-squared CI can drastically under- or over-cover. A bootstrap percentile or BCa interval for $\sigma^2$ is a more robust alternative.
- The interval can optionally be reported on the standard deviation scale by taking square roots of both endpoints.

## Exercises

**Exercise 1.** A sample of size $n = 20$ from a normal population yields $s^2 = 16$. Construct a 95 % confidence interval for $\sigma^2$ and for $\sigma$.

??? success "Solution to Exercise 1"

    With $\text{df} = 19$ and $\alpha = 0.05$:

    $$
    \chi^2_{0.025,\,19} = 8.907, \quad \chi^2_{0.975,\,19} = 32.852
    $$

    The 95 % CI for $\sigma^2$ is

    $$
    \left(\frac{19 \times 16}{32.852},\; \frac{19 \times 16}{8.907}\right) = \left(\frac{304}{32.852},\; \frac{304}{8.907}\right) = (9.25,\; 34.13)
    $$

    Taking square roots, the 95 % CI for $\sigma$ is

    $$
    (\sqrt{9.25},\; \sqrt{34.13}) = (3.04,\; 5.84)
    $$

    $\square$

---

**Exercise 2.** Prove that $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$ when $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$.

??? success "Solution to Exercise 2"

    Define $Z_i = (X_i - \mu)/\sigma$, so $Z_i \overset{\text{iid}}{\sim} N(0,1)$. Then

    $$
    \sum_{i=1}^n Z_i^2 = \sum_{i=1}^n \frac{(X_i - \mu)^2}{\sigma^2} \sim \chi^2_n
    $$

    Decompose via the identity $X_i - \mu = (X_i - \bar{X}) + (\bar{X} - \mu)$:

    $$
    \sum_{i=1}^n (X_i - \mu)^2 = \sum_{i=1}^n (X_i - \bar{X})^2 + n(\bar{X} - \mu)^2
    $$

    Dividing by $\sigma^2$:

    $$
    \chi^2_n = \frac{(n-1)S^2}{\sigma^2} + \frac{(\bar{X}-\mu)^2}{\sigma^2/n}
    $$

    The second term is $Z^2$ where $Z = (\bar{X}-\mu)/(\sigma/\sqrt{n}) \sim N(0,1)$, so it follows $\chi^2_1$. By Cochran's theorem (or the independence of $\bar{X}$ and $S^2$ under normality), the two terms are independent. Therefore

    $$
    \frac{(n-1)S^2}{\sigma^2} = \chi^2_n - \chi^2_1 \sim \chi^2_{n-1}
    $$

    by the additive property of independent chi-squared random variables. $\square$

---

**Exercise 3.** Modify the simulation to draw data from an exponential distribution with $\lambda = 1$ (so $\sigma^2 = 1$) and $n = 20$. Report the empirical coverage. Why does the chi-squared interval fail?

??? success "Solution to Exercise 3"

    ```python
    from scipy.stats import chi2
    np.random.seed(0)
    n, n_sim, true_var = 20, 10_000, 1.0
    df = n - 1
    chi2_lo = chi2.ppf(0.025, df)
    chi2_hi = chi2.ppf(0.975, df)
    covers = 0
    for _ in range(n_sim):
        x = np.random.exponential(1.0, size=n)
        s2 = x.var(ddof=1)
        lo = df * s2 / chi2_hi
        hi = df * s2 / chi2_lo
        if lo <= true_var <= hi:
            covers += 1
    print(f"Coverage: {100 * covers / n_sim:.1f}%")
    ```

    Typical result: coverage $\approx$ 87--90 %, well below 95 %. The exponential distribution is right-skewed with excess kurtosis 6, which means $(n-1)S^2/\sigma^2$ does not follow $\chi^2_{n-1}$. The chi-squared pivot relies on normality, and no CLT-based argument rescues the variance CI the way it does for the mean. $\square$

---

**Exercise 4.** Show that the chi-squared CI for $\sigma^2$ is not symmetric about $S^2$, and explain why geometrically.

??? success "Solution to Exercise 4"

    The lower endpoint is $L = (n-1)S^2 / \chi^2_{1-\alpha/2}$ and the upper endpoint is $U = (n-1)S^2 / \chi^2_{\alpha/2}$. The distances from $S^2$ are:

    $$
    S^2 - L = S^2\!\left(1 - \frac{n-1}{\chi^2_{1-\alpha/2}}\right), \quad U - S^2 = S^2\!\left(\frac{n-1}{\chi^2_{\alpha/2}} - 1\right)
    $$

    Since the $\chi^2_{n-1}$ distribution is right-skewed, $\chi^2_{\alpha/2} < n-1 < \chi^2_{1-\alpha/2}$ (the mean of $\chi^2_{n-1}$ is $n-1$, which lies closer to the upper quantile). This means $n-1/\chi^2_{\alpha/2}$ is larger than $n-1/\chi^2_{1-\alpha/2}$ is below 1, so $U - S^2 > S^2 - L$. The upper tail of the CI extends farther because the chi-squared distribution has a long right tail: small chi-squared values in the denominator produce large values of $\sigma^2$. $\square$

---

**Exercise 5.** If a 95 % CI for $\sigma^2$ is $(9.25, 34.13)$, can we claim that the population standard deviation is less than 6? Justify your answer.

??? success "Solution to Exercise 5"

    The 95 % CI for $\sigma$ is $(\sqrt{9.25}, \sqrt{34.13}) = (3.04, 5.84)$. The entire interval lies below 6, so at the 95 % confidence level, the data are consistent with $\sigma < 6$. Equivalently, the value $\sigma = 6$ (i.e., $\sigma^2 = 36$) falls outside the 95 % CI for $\sigma^2$, which is $(9.25, 34.13)$. Therefore we have sufficient evidence at the 5 % significance level to conclude that $\sigma < 6$. $\square$
