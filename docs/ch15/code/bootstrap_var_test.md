# Bootstrap Variance Test

## Overview

This page demonstrates a nonparametric bootstrap approach for testing whether two groups have equal variances. Instead of relying on distributional assumptions (as the $F$-test does), we resample from the observed data to build a sampling distribution of the variance ratio. The bootstrap provides both a confidence interval and an approximate $p$-value, making it a flexible alternative when normality is questionable.

---

## The Variance Ratio Statistic

Given two independent samples $\mathbf{x}_1 = (x_{11}, \ldots, x_{1,n_1})$ and $\mathbf{x}_2 = (x_{21}, \ldots, x_{2,n_2})$, the statistic of interest is the ratio of sample variances:

$$
\hat\theta = \frac{S_1^2}{S_2^2}, \qquad S_i^2 = \frac{1}{n_i - 1}\sum_{j=1}^{n_i}(x_{ij} - \bar{x}_i)^2
$$

Under the null hypothesis $H_0\colon \sigma_1^2 = \sigma_2^2$, the true ratio is $\theta = 1$.

---

## Bootstrap Procedure

The nonparametric bootstrap estimates the sampling distribution of $\hat\theta$ without assuming normality:

1. **Resample**: For each bootstrap replicate $b = 1, \ldots, B$, draw $n_1$ observations with replacement from $\mathbf{x}_1$ and $n_2$ from $\mathbf{x}_2$.
2. **Compute**: Calculate $\hat\theta^{(b)} = S_1^{2(b)} / S_2^{2(b)}$.
3. **Log-transform**: Work on the log scale $\log\hat\theta^{(b)}$ for better symmetry, since ratios are right-skewed.
4. **Confidence interval**: The percentile CI on the log scale is $(q_{0.025}, q_{0.975})$. Exponentiate to get the CI for $\theta$.
5. **$p$-value**: An approximate two-sided bootstrap $p$-value is:

$$
p = 2 \min\!\Big(\frac{1}{B}\sum_{b=1}^B \mathbf{1}(\log\hat\theta^{(b)} \le \log\hat\theta),\;\; \frac{1}{B}\sum_{b=1}^B \mathbf{1}(\log\hat\theta^{(b)} \ge \log\hat\theta)\Big)
$$

---

## Implementation

```python
import numpy as np

def variance_ratio(x1, x2):
    return np.var(x1, ddof=1) / np.var(x2, ddof=1)

def bootstrap_varratio(x1, x2, B=2000, seed=None):
    rng = np.random.default_rng(seed)
    n1, n2 = len(x1), len(x2)
    stat_obs = variance_ratio(x1, x2)
    log_obs = np.log(stat_obs)

    boots = np.empty(B)
    for b in range(B):
        b1 = rng.choice(x1, size=n1, replace=True)
        b2 = rng.choice(x2, size=n2, replace=True)
        boots[b] = np.log(variance_ratio(b1, b2))

    # Percentile CI on log scale, then exponentiate
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ci = (float(np.exp(lo)), float(np.exp(hi)))

    # Approximate two-sided p-value
    p_two = 2 * min(np.mean(boots <= log_obs), np.mean(boots >= log_obs))
    p_two = float(min(p_two, 1.0))

    return float(stat_obs), ci, p_two
```

---

## Worked Example

```python
x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)

theta_hat, ci, p = bootstrap_varratio(x1, x2, B=10000, seed=42)
print(f"Observed ratio: {theta_hat:.4f}")
print(f"95% Bootstrap CI: ({ci[0]:.4f}, {ci[1]:.4f})")
print(f"Bootstrap p-value: {p:.4f}")
```

If the 95% confidence interval contains 1, we fail to reject the null hypothesis of equal variances. The $p$-value provides a complementary measure of evidence.

---

## Interpretation

- The bootstrap makes no parametric assumption about the distribution of $S^2$. It is valid for non-normal, skewed, or heavy-tailed data where the $F$-test may fail.
- Working on the log scale improves the symmetry of the bootstrap distribution, producing more accurate percentile intervals.
- The bootstrap $p$-value is approximate. With $B = 10{,}000$ replicates, the Monte Carlo error is of order $1/\sqrt{B} \approx 0.01$.
- For small sample sizes, the bootstrap distribution can be coarse, and BCa (bias-corrected and accelerated) intervals may be preferred.

---

## Exercises

**Exercise 1.** Generate two samples of size 30 from $\mathcal{N}(0, 1)$ (equal variances). Run the bootstrap variance test with $B = 5000$. Does the 95% confidence interval contain 1? Repeat 500 times and estimate the coverage probability (the proportion of intervals that contain the true ratio of 1).

??? success "Solution to Exercise 1"

    ```python
    import numpy as np

    def variance_ratio(x1, x2):
        return np.var(x1, ddof=1) / np.var(x2, ddof=1)

    def bootstrap_varratio(x1, x2, B=5000, seed=None):
        rng = np.random.default_rng(seed)
        n1, n2 = len(x1), len(x2)
        boots = np.empty(B)
        for b in range(B):
            b1 = rng.choice(x1, size=n1, replace=True)
            b2 = rng.choice(x2, size=n2, replace=True)
            boots[b] = np.log(variance_ratio(b1, b2))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return np.exp(lo), np.exp(hi)

    rng = np.random.default_rng(0)
    covers = 0
    for trial in range(500):
        x1 = rng.normal(0, 1, 30)
        x2 = rng.normal(0, 1, 30)
        lo, hi = bootstrap_varratio(x1, x2, B=5000, seed=trial)
        if lo <= 1.0 <= hi:
            covers += 1

    print(f"Coverage: {covers/500:.3f}")
    ```

    The coverage should be close to 0.95, confirming that the percentile bootstrap CI has approximately correct coverage for the variance ratio under normality. $\square$

---

**Exercise 2.** Explain why the log transformation improves the bootstrap confidence interval for a variance ratio. Hint: consider the support and symmetry of $S_1^2 / S_2^2$ versus $\log(S_1^2 / S_2^2)$.

??? success "Solution to Exercise 2"

    The ratio $\theta = S_1^2/S_2^2$ takes values in $(0, \infty)$. Its distribution is right-skewed: the ratio can be arbitrarily large but not negative. The percentile method assumes the bootstrap distribution is approximately pivotal (symmetric about the true parameter), so applying it directly to $\theta$ produces intervals that are too narrow on the left and too wide on the right.

    The log transformation maps $(0, \infty) \to (-\infty, \infty)$. Under the null ($\theta = 1$), $\log\theta = 0$ sits at the center of the real line. The distribution of $\log(S_1^2/S_2^2)$ is much closer to symmetric, so the percentile interval on the log scale is more accurate. Exponentiating back gives an interval for $\theta$ that is asymmetric in the right way: narrower below 1 and wider above 1, matching the true skewness of the ratio. $\square$

---

**Exercise 3.** Compare the bootstrap variance test to the classical $F$-test on data drawn from an exponential distribution (which is highly right-skewed). Generate $n_1 = n_2 = 20$ from $\text{Exp}(1)$ and $\text{Exp}(1)$ (equal variances). Run both tests 2000 times at $\alpha = 0.05$ and report the false-positive rate of each.

??? success "Solution to Exercise 3"

    ```python
    import numpy as np
    from scipy.stats import f as f_dist

    def variance_ratio(x1, x2):
        return np.var(x1, ddof=1) / np.var(x2, ddof=1)

    def bootstrap_pvalue(x1, x2, B=2000, seed=None):
        rng = np.random.default_rng(seed)
        n1, n2 = len(x1), len(x2)
        log_obs = np.log(variance_ratio(x1, x2))
        boots = np.empty(B)
        for b in range(B):
            b1 = rng.choice(x1, size=n1, replace=True)
            b2 = rng.choice(x2, size=n2, replace=True)
            boots[b] = np.log(variance_ratio(b1, b2))
        p = 2 * min(np.mean(boots <= log_obs), np.mean(boots >= log_obs))
        return min(p, 1.0)

    rng = np.random.default_rng(42)
    rej_f, rej_boot = 0, 0
    n_sims = 2000

    for i in range(n_sims):
        x1 = rng.exponential(1, 20)
        x2 = rng.exponential(1, 20)
        F = variance_ratio(x1, x2)
        p_f = 2 * min(f_dist.cdf(F, 19, 19), 1 - f_dist.cdf(F, 19, 19))
        if p_f < 0.05:
            rej_f += 1
        p_b = bootstrap_pvalue(x1, x2, B=1000, seed=i)
        if p_b < 0.05:
            rej_boot += 1

    print(f"F-test false positive rate: {rej_f/n_sims:.3f}")
    print(f"Bootstrap false positive rate: {rej_boot/n_sims:.3f}")
    ```

    The $F$-test will have an inflated false-positive rate (often above 0.10) because the exponential distribution violates normality severely. The bootstrap test should be closer to the nominal 0.05 rate, since it does not assume any particular distributional form. $\square$

---

**Exercise 4.** The percentile bootstrap interval is the simplest variant. Implement the **BCa (bias-corrected and accelerated)** bootstrap interval for the log variance ratio. Apply both methods to the data `x1 = [12, 15, 14, 10, 13, 14, 12, 11]` and `x2 = [22, 25, 20, 18, 24, 23, 19, 21]` and compare the resulting intervals.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy import stats as sp_stats

    def variance_ratio(x1, x2):
        return np.var(x1, ddof=1) / np.var(x2, ddof=1)

    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)
    rng = np.random.default_rng(42)
    B = 10000
    n1, n2 = len(x1), len(x2)

    log_obs = np.log(variance_ratio(x1, x2))
    boots = np.array([np.log(variance_ratio(
        rng.choice(x1, n1, replace=True),
        rng.choice(x2, n2, replace=True))) for _ in range(B)])

    # Percentile
    lo_p, hi_p = np.percentile(boots, [2.5, 97.5])

    # BCa: bias correction
    z0 = sp_stats.norm.ppf(np.mean(boots < log_obs))

    # Acceleration (jackknife)
    jk = np.empty(n1)
    for i in range(n1):
        x1_jk = np.delete(x1, i)
        jk[i] = np.log(np.var(x1_jk, ddof=1) / np.var(x2, ddof=1))
    jk_mean = jk.mean()
    a_hat = np.sum((jk_mean - jk)**3) / (6 * np.sum((jk_mean - jk)**2)**1.5)

    alpha1 = sp_stats.norm.cdf(z0 + (z0 + sp_stats.norm.ppf(0.025)) /
                                (1 - a_hat * (z0 + sp_stats.norm.ppf(0.025))))
    alpha2 = sp_stats.norm.cdf(z0 + (z0 + sp_stats.norm.ppf(0.975)) /
                                (1 - a_hat * (z0 + sp_stats.norm.ppf(0.975))))
    lo_bca, hi_bca = np.percentile(boots, [100*alpha1, 100*alpha2])

    print(f"Percentile CI (log): ({lo_p:.3f}, {hi_p:.3f})")
    print(f"BCa CI (log):        ({lo_bca:.3f}, {hi_bca:.3f})")
    print(f"Percentile CI (ratio): ({np.exp(lo_p):.3f}, {np.exp(hi_p):.3f})")
    print(f"BCa CI (ratio):        ({np.exp(lo_bca):.3f}, {np.exp(hi_bca):.3f})")
    ```

    The BCa interval adjusts for both bias (the median of the bootstrap distribution may not equal $\hat\theta$) and skewness (the acceleration factor $\hat{a}$). For small samples, the BCa and percentile intervals can differ noticeably. The BCa interval generally has better coverage properties. $\square$

---

**Exercise 5.** Prove that the bootstrap percentile interval is transformation-invariant. That is, if $(L, U)$ is the percentile interval for $\theta$, then $(g(L), g(U))$ is the percentile interval for $g(\theta)$ for any monotone increasing function $g$.

??? success "Solution to Exercise 5"

    Let $\hat\theta_1^*, \ldots, \hat\theta_B^*$ be the bootstrap replicates. The $(100\alpha/2)$-th and $(100(1-\alpha/2))$-th percentiles of these replicates define the interval $(L, U)$:

    $$
    L = \hat\theta^*_{(\lfloor B\alpha/2 \rfloor)}, \qquad U = \hat\theta^*_{(\lceil B(1-\alpha/2) \rceil)}
    $$

    where $\hat\theta^*_{(k)}$ denotes the $k$-th order statistic. Now consider the transformed replicates $g(\hat\theta_1^*), \ldots, g(\hat\theta_B^*)$. Since $g$ is monotone increasing, the order statistics transform consistently:

    $$
    g(\hat\theta^*)_{(k)} = g(\hat\theta^*_{(k)})
    $$

    Therefore the percentile interval for $g(\theta)$ is:

    $$
    \bigl(g(\hat\theta^*_{(\lfloor B\alpha/2 \rfloor)}),\;\; g(\hat\theta^*_{(\lceil B(1-\alpha/2) \rceil)})\bigr) = (g(L),\; g(U))
    $$

    This is why computing the percentile interval on the log scale and exponentiating is equivalent to computing the percentile interval directly on the ratio scale. The transformation invariance is a key advantage of the percentile method over normal-approximation intervals. $\square$
