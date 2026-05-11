# Bootstrap Confidence Interval Methods

## Overview

Bootstrap confidence intervals provide a way to quantify uncertainty about a parameter estimate without relying on distributional assumptions. This page presents three bootstrap CI methods -- percentile, basic (reverse percentile), and BCa (bias-corrected and accelerated) -- applied to a non-normal (Poisson) sample where classical intervals may be unreliable. Each method exploits the empirical bootstrap distribution in a different way, offering trade-offs between simplicity and accuracy.

## The Bootstrap Principle

Given an observed sample $x_1, x_2, \ldots, x_n$, we approximate the sampling distribution of a statistic $\hat\theta = T(x_1, \ldots, x_n)$ by repeatedly resampling **with replacement** from the data. Each bootstrap replicate $\hat\theta^{*(b)}$, for $b = 1, \ldots, B$, is computed on a resample of size $n$ drawn uniformly from the original observations.

## Percentile Method

The percentile method reads the confidence limits directly from the quantiles of the bootstrap distribution. For a $100(1 - \alpha)\%$ CI:

$$
\text{CI}_{\text{pct}} = \bigl[\hat\theta^*_{\alpha/2},\;\hat\theta^*_{1 - \alpha/2}\bigr]
$$

where $\hat\theta^*_q$ denotes the $q$-th quantile of the bootstrap distribution.

```python
def bootstrap_percentile_ci(data, statistic, n_boot=10_000, alpha=0.05):
    """Percentile bootstrap CI."""
    n = len(data)
    boot_stats = np.array([
        statistic(data[np.random.randint(0, n, n)])
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_stats, 100 * alpha / 2)
    hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return lo, hi, boot_stats
```

The method is simple and intuitive but can under-cover when the bootstrap distribution is biased or skewed.

## Basic (Reverse Percentile) Method

The basic method uses the bootstrap distribution to estimate the spread of $\hat\theta - \theta$, then inverts the interval. Let $\hat\theta$ be the sample statistic. The interval is:

$$
\text{CI}_{\text{basic}} = \bigl[2\hat\theta - \hat\theta^*_{1 - \alpha/2},\;2\hat\theta - \hat\theta^*_{\alpha/2}\bigr]
$$

```python
def bootstrap_basic_ci(data, statistic, n_boot=10_000, alpha=0.05):
    """Basic (reverse-percentile) bootstrap CI."""
    theta_hat = statistic(data)
    n = len(data)
    boot_stats = np.array([
        statistic(data[np.random.randint(0, n, n)])
        for _ in range(n_boot)
    ])
    lo = 2 * theta_hat - np.percentile(boot_stats, 100 * (1 - alpha / 2))
    hi = 2 * theta_hat - np.percentile(boot_stats, 100 * alpha / 2)
    return lo, hi, boot_stats
```

The key idea is that if the bootstrap overestimates $\hat\theta$, we correct by reflecting the quantiles around $\hat\theta$.

## BCa Method (Bias-Corrected and Accelerated)

The BCa method adjusts for both **bias** and **skewness** in the bootstrap distribution. It introduces two correction factors:

- **Bias correction** $z_0$: measures how far the center of the bootstrap distribution is from $\hat\theta$.
- **Acceleration** $a$: captures the rate at which the standard error of $\hat\theta$ changes with respect to the true parameter, estimated via the jackknife.

The adjusted percentile levels are:

$$
\alpha_1 = \mathcal{N}\!\left(z_0 + \frac{z_0 + z_{\alpha/2}}{1 - a(z_0 + z_{\alpha/2})}\right), \qquad \alpha_2 = \mathcal{N}\!\left(z_0 + \frac{z_0 + z_{1 - \alpha/2}}{1 - a(z_0 + z_{1 - \alpha/2})}\right)
$$

where $\mathcal{N}$ is the standard normal CDF, $z_q = \mathcal{N}^{-1}(q)$, and:

$$
z_0 = \mathcal{N}^{-1}\!\left(\frac{1}{B}\sum_{b=1}^{B}\mathbf{1}(\hat\theta^{*(b)} < \hat\theta)\right)
$$

$$
a = \frac{\sum_{i=1}^{n}(\bar\theta_{(\cdot)} - \hat\theta_{(i)})^3}{6\left[\sum_{i=1}^{n}(\bar\theta_{(\cdot)} - \hat\theta_{(i)})^2\right]^{3/2}}
$$

Here $\hat\theta_{(i)}$ is the jackknife replicate with observation $i$ deleted, and $\bar\theta_{(\cdot)}$ is the mean of the jackknife replicates.

```python
def bootstrap_bca_ci(data, statistic, n_boot=10_000, alpha=0.05):
    """BCa (bias-corrected and accelerated) bootstrap CI."""
    n = len(data)
    theta_hat = statistic(data)

    boot_stats = np.array([
        statistic(data[np.random.randint(0, n, n)])
        for _ in range(n_boot)
    ])

    # Bias correction factor z0
    z0 = stats.norm.ppf(np.mean(boot_stats < theta_hat))

    # Acceleration factor a -- jackknife estimate
    jack = np.array([statistic(np.delete(data, i)) for i in range(n)])
    jack_mean = jack.mean()
    a_num = np.sum((jack_mean - jack) ** 3)
    a_den = 6 * np.sum((jack_mean - jack) ** 2) ** 1.5
    a = a_num / a_den if a_den != 0 else 0.0

    # Adjusted percentiles
    z_alpha = stats.norm.ppf(alpha / 2)
    z_1alpha = stats.norm.ppf(1 - alpha / 2)

    p_lo = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    p_hi = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    lo = np.percentile(boot_stats, 100 * p_lo)
    hi = np.percentile(boot_stats, 100 * p_hi)
    return lo, hi, boot_stats
```

## Applying the Methods to Poisson Data

The example draws $n = 80$ observations from a Poisson distribution with true rate $\lambda = 3.5$. The Poisson distribution is discrete and right-skewed, making it a good candidate for bootstrap methods.

```python
lam_true = 3.5
n = 80
data = stats.poisson.rvs(lam_true, size=n)
sample_mean = np.mean(data)

lo_p, hi_p, boots = bootstrap_percentile_ci(data, np.mean, n_boot=10_000)
lo_b, hi_b, _     = bootstrap_basic_ci(data, np.mean, n_boot=10_000)
lo_bca, hi_bca, _ = bootstrap_bca_ci(data, np.mean, n_boot=10_000)
```

Typical output:

| Method | Lower | Upper |
|---|---|---|
| Percentile | 3.213 | 3.875 |
| Basic | 3.200 | 3.862 |
| BCa | 3.225 | 3.900 |

All three intervals are similar here because $n = 80$ is moderately large and the sample mean is well-behaved. For smaller samples or highly skewed statistics, BCa often provides better coverage.

## Interpretation

- **Percentile** is the simplest; it works well when the bootstrap distribution is approximately symmetric and unbiased.
- **Basic** corrects for bias in location by reflecting quantiles around $\hat\theta$, but does not account for skewness.
- **BCa** provides the most reliable coverage among the three because it corrects for both bias and acceleration (skewness). The cost is additional computation for the jackknife.

When the bootstrap distribution is symmetric and centered on $\hat\theta$, all three methods give nearly identical results. Their differences become apparent with small samples, skewed statistics (e.g., the median, variance), or heavy-tailed data.

## Exercises

**Exercise 1.** Generate a sample of size $n = 30$ from an exponential distribution with rate $\lambda = 1$ and compute all three bootstrap 95% CIs for the sample mean. Which interval is widest? Why?

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(0)
    data = np.random.exponential(scale=1.0, size=30)

    lo_p, hi_p, _ = bootstrap_percentile_ci(data, np.mean)
    lo_b, hi_b, _ = bootstrap_basic_ci(data, np.mean)
    lo_bca, hi_bca, _ = bootstrap_bca_ci(data, np.mean)

    print(f"Percentile: [{lo_p:.4f}, {hi_p:.4f}], width = {hi_p - lo_p:.4f}")
    print(f"Basic:      [{lo_b:.4f}, {hi_b:.4f}], width = {hi_b - lo_b:.4f}")
    print(f"BCa:        [{lo_bca:.4f}, {hi_bca:.4f}], width = {hi_bca - lo_bca:.4f}")
    ```

    The BCa interval is typically the widest because it shifts the percentile boundaries to correct for the positive skew of the exponential distribution. The correction pushes the upper boundary further right, widening the interval to achieve more accurate coverage. $\square$

---

**Exercise 2.** Prove that when the bootstrap distribution is exactly symmetric about $\hat\theta$ and unbiased (i.e., $z_0 = 0$ and $a = 0$), the BCa interval reduces to the percentile interval.

??? success "Solution to Exercise 2"

    When $z_0 = 0$ and $a = 0$, the adjusted percentile levels become:

    $$
    \alpha_1 = \mathcal{N}\!\left(0 + \frac{0 + z_{\alpha/2}}{1 - 0}\right) = \mathcal{N}(z_{\alpha/2}) = \frac{\alpha}{2}
    $$

    $$
    \alpha_2 = \mathcal{N}\!\left(0 + \frac{0 + z_{1-\alpha/2}}{1 - 0}\right) = \mathcal{N}(z_{1-\alpha/2}) = 1 - \frac{\alpha}{2}
    $$

    These are exactly the quantile levels used by the percentile method. Therefore:

    $$
    \text{CI}_{\text{BCa}} = \bigl[\hat\theta^*_{\alpha/2},\;\hat\theta^*_{1 - \alpha/2}\bigr] = \text{CI}_{\text{pct}}
    $$

    $\square$

---

**Exercise 3.** Show that the basic bootstrap interval can be written as $\hat\theta \pm (\hat\theta - \hat\theta^*_q)$ for appropriate quantile $q$. Explain geometrically why this is called the "reflection" method.

??? success "Solution to Exercise 3"

    The basic interval is:

    $$
    \text{CI}_{\text{basic}} = \bigl[2\hat\theta - \hat\theta^*_{1-\alpha/2},\; 2\hat\theta - \hat\theta^*_{\alpha/2}\bigr]
    $$

    The lower endpoint can be rewritten as:

    $$
    2\hat\theta - \hat\theta^*_{1-\alpha/2} = \hat\theta - (\hat\theta^*_{1-\alpha/2} - \hat\theta)
    $$

    and the upper endpoint as:

    $$
    2\hat\theta - \hat\theta^*_{\alpha/2} = \hat\theta + (\hat\theta - \hat\theta^*_{\alpha/2})
    $$

    Geometrically, the bootstrap distribution quantifies how $\hat\theta^*$ varies around $\hat\theta$. The basic method assumes that $\hat\theta$ varies around $\theta$ in the same way, so we *reflect* the bootstrap quantiles through $\hat\theta$ to obtain the confidence limits for $\theta$. The name "reflection" comes from this mirror-image transformation about the point estimate. $\square$

---

**Exercise 4.** Run a coverage simulation: draw 500 samples of size $n = 20$ from a $\chi^2(3)$ distribution. For each sample, compute the percentile and BCa 95% CIs for the mean. Report the empirical coverage (fraction of intervals containing the true mean $\mu = 3$). Which method is closer to the nominal 95%?

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(42)
    true_mean = 3.0
    n_sim = 500
    cover_pct = 0
    cover_bca = 0

    for _ in range(n_sim):
        data = np.random.chisquare(df=3, size=20)
        lo_p, hi_p, _ = bootstrap_percentile_ci(data, np.mean, n_boot=2000)
        lo_bca, hi_bca, _ = bootstrap_bca_ci(data, np.mean, n_boot=2000)
        if lo_p <= true_mean <= hi_p:
            cover_pct += 1
        if lo_bca <= true_mean <= hi_bca:
            cover_bca += 1

    print(f"Percentile coverage: {cover_pct / n_sim:.3f}")
    print(f"BCa coverage:        {cover_bca / n_sim:.3f}")
    ```

    Typical results show the percentile method achieving roughly 91--93% coverage while BCa achieves 93--95%, closer to the nominal 95%. The $\chi^2(3)$ distribution is right-skewed, and the BCa correction accounts for this skewness by adjusting the quantile boundaries. $\square$

---

**Exercise 5.** The jackknife acceleration factor $a$ involves the third moment of the jackknife values. Explain intuitively why a statistic with a positively skewed sampling distribution would have $a > 0$, and describe how this shifts the BCa interval relative to the percentile interval.

??? success "Solution to Exercise 5"

    The acceleration factor is:

    $$
    a = \frac{\sum_{i=1}^{n}(\bar\theta_{(\cdot)} - \hat\theta_{(i)})^3}{6\left[\sum_{i=1}^{n}(\bar\theta_{(\cdot)} - \hat\theta_{(i)})^2\right]^{3/2}}
    $$

    The numerator is the (unnormalized) third central moment of the jackknife values. When the sampling distribution of $\hat\theta$ is positively skewed, removing observations that lower $\hat\theta$ produces jackknife values that cluster below the mean, while removing observations that raise $\hat\theta$ produces a few jackknife values well above the mean. This asymmetry gives a positive third moment, hence $a > 0$.

    When $a > 0$, the adjusted percentile levels shift upward: both $\alpha_1$ and $\alpha_2$ increase. This pushes the BCa interval to the right compared to the percentile interval, widening the upper tail and narrowing the lower tail. The net effect is that the BCa interval captures more of the right tail of the sampling distribution, improving coverage for positively skewed statistics. $\square$
