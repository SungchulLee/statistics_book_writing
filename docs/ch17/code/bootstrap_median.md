# Bootstrap Median

## Overview

The median is a robust measure of central tendency, but unlike the mean, it has no simple closed-form expression for its standard error. The bootstrap provides a direct, assumption-free estimate of the SE and confidence intervals for the median. This page demonstrates bootstrap estimation of the median on synthetic income data, compares it with the mean, and illustrates the median's robustness to outliers.

## Why Bootstrap the Median

For a sample of size $n$ from a distribution with density $f$ and median $m$, the asymptotic standard error of the sample median is:

$$
\text{SE}(\tilde x) \approx \frac{1}{2f(m)\sqrt{n}}
$$

This formula requires knowledge of $f(m)$, the density at the population median, which is typically unknown. The bootstrap avoids this entirely by estimating the SE empirically.

## Bootstrap Procedure

Given data $x_1, \ldots, x_n$:

1. Draw $B$ bootstrap samples, each of size $n$ with replacement.
2. Compute the median of each bootstrap sample: $\tilde x^{*(1)}, \ldots, \tilde x^{*(B)}$.
3. The bootstrap standard error is the standard deviation of the bootstrap medians:

$$
\widehat{\text{SE}}_{\text{boot}}(\tilde x) = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\bigl(\tilde x^{*(b)} - \overline{\tilde x^*}\bigr)^2}
$$

4. The bootstrap bias is:

$$
\widehat{\text{bias}} = \overline{\tilde x^*} - \tilde x
$$

```python
def bootstrap_median(data, n_bootstrap=1000):
    """Compute bootstrap distribution of the median."""
    bootstrap_medians = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(data)
        bootstrap_medians.append(np.median(bootstrap_sample))
    return np.array(bootstrap_medians)
```

## Comparing Median and Mean

For symmetric distributions, the mean and median coincide. For skewed distributions, they diverge. With exponential-like income data:

$$
\text{Mean} = c + \frac{1}{\lambda}, \qquad \text{Median} = c + \frac{\ln 2}{\lambda}
$$

Since $\ln 2 \approx 0.693 < 1$, the median is always less than the mean for this family. The bootstrap captures this difference and reveals that the SE of the median is typically larger than the SE of the mean (the median trades precision for robustness).

```python
def bootstrap_mean_for_comparison(data, n_bootstrap=1000):
    """Compute bootstrap distribution of the mean for comparison."""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(data)
        bootstrap_means.append(np.mean(bootstrap_sample))
    return np.array(bootstrap_means)
```

## Robustness to Outliers

Adding a single extreme observation (e.g., \$1,000,000 income) dramatically changes the mean but barely affects the median. This is because the median depends only on the order statistics near the center of the data, not on extreme values.

Formally, the **influence function** of the median at any point $x$ is bounded:

$$
\text{IF}(x; \tilde F, F) = \frac{\text{sign}(x - m)}{2f(m)}
$$

while the influence function of the mean is $\text{IF}(x; \bar F, F) = x - \mu$, which is unbounded. This means a single outlier can shift the mean arbitrarily far, but can move the median by at most $1/(2f(m))$.

```python
def robustness_comparison(data):
    """Demonstrate robustness of median vs mean to outliers."""
    original_mean = np.mean(data)
    original_median = np.median(data)

    data_with_outlier = np.append(data, 1_000_000)

    mean_with_outlier = np.mean(data_with_outlier)
    median_with_outlier = np.median(data_with_outlier)

    print(f"Mean change:   {(mean_with_outlier - original_mean) / original_mean * 100:.2f}%")
    print(f"Median change: {(median_with_outlier - original_median) / original_median * 100:.2f}%")
```

## Bootstrap Confidence Intervals for the Median

The percentile method provides straightforward CIs:

$$
\text{CI}_{1-\alpha} = \bigl[\tilde x^*_{\alpha/2},\;\tilde x^*_{1-\alpha/2}\bigr]
$$

```python
def confidence_intervals(bootstrap_dist, confidence_levels=[90, 95, 99]):
    """Compute bootstrap CIs using the percentile method."""
    for cl in confidence_levels:
        alpha = (100 - cl) / 2
        lower = np.percentile(bootstrap_dist, alpha)
        upper = np.percentile(bootstrap_dist, 100 - alpha)
        width = upper - lower
        print(f"{cl}% CI: [{lower:,.0f}, {upper:,.0f}]  Width: {width:,.0f}")
```

Higher confidence levels produce wider intervals, reflecting the confidence-precision trade-off.

## Interpretation

- The bootstrap SE of the median provides uncertainty quantification where no formula exists.
- For skewed distributions (income, medical costs, claim sizes), the median is a more representative summary than the mean.
- The bootstrap distribution of the median may be less smooth than that of the mean because the median is a discontinuous function of the order statistics. Larger values of $B$ help.
- In finance and insurance, the median is often preferred because extreme values (large claims, market crashes) can distort the mean.

## Exercises

**Exercise 1.** Generate a sample of size $n = 50$ from a standard normal distribution. Compute the bootstrap SE of the median and compare it with the theoretical value $\sqrt{\pi/(2n)} \approx 1/(2f(0)\sqrt{n})$ where $f(0) = 1/\sqrt{2\pi}$.

??? success "Solution to Exercise 1"

    For the standard normal, $f(0) = 1/\sqrt{2\pi} \approx 0.3989$. The asymptotic SE of the median is:

    $$
    \text{SE}(\tilde x) = \frac{1}{2 \times 0.3989 \times \sqrt{50}} = \frac{1}{5.641} \approx 0.1773
    $$

    Equivalently, $\text{SE} = \sqrt{\pi/(2n)} = \sqrt{\pi/100} \approx 0.1773$.

    ```python
    import numpy as np
    from sklearn.utils import resample

    np.random.seed(42)
    data = np.random.normal(0, 1, 50)

    boot_medians = bootstrap_median(data, n_bootstrap=5000)
    se_boot = boot_medians.std(ddof=1)
    se_theory = np.sqrt(np.pi / (2 * 50))

    print(f"Bootstrap SE:    {se_boot:.4f}")
    print(f"Theoretical SE:  {se_theory:.4f}")
    ```

    The bootstrap SE should be close to 0.177. Any discrepancy is due to Monte Carlo variability and the finite sample size. $\square$

---

**Exercise 2.** The **asymptotic relative efficiency (ARE)** of the median to the mean for a normal distribution is $\pi/2 \approx 1.571$. This means the variance of the median is about 57% larger. Verify this empirically by comparing the bootstrap SEs of the mean and median from the same normal sample.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from sklearn.utils import resample

    np.random.seed(42)
    data = np.random.normal(0, 1, 200)

    boot_means = np.array([np.mean(resample(data)) for _ in range(5000)])
    boot_medians = np.array([np.median(resample(data)) for _ in range(5000)])

    se_mean = boot_means.std(ddof=1)
    se_median = boot_medians.std(ddof=1)
    ratio = (se_median / se_mean) ** 2

    print(f"SE(mean):   {se_mean:.4f}")
    print(f"SE(median): {se_median:.4f}")
    print(f"Var ratio:  {ratio:.3f} (theory: {np.pi/2:.3f})")
    ```

    The variance ratio $\text{Var}(\tilde x)/\text{Var}(\bar x)$ should be approximately $\pi/2 \approx 1.571$, meaning the median is less efficient than the mean for normal data. However, this efficiency loss is the price paid for robustness. For contaminated data (e.g., a normal mixture with outliers), the ARE reverses and the median becomes more efficient. $\square$

---

**Exercise 3.** Prove that the breakdown point of the median is 50%, meaning that up to half the observations can be replaced by arbitrary values without the median diverging to infinity. Show that the breakdown point of the mean is $0\%$ (a single outlier can make it arbitrarily large).

??? success "Solution to Exercise 3"

    **Median.** Consider a sample $x_1 \le x_2 \le \cdots \le x_n$. The median is $x_{(k)}$ where $k = \lceil n/2 \rceil$. Now replace $m$ of the $n$ observations with arbitrarily large values. As long as $m < n/2$, at least $\lceil n/2 \rceil$ of the original values remain, so the $\lceil n/2 \rceil$-th order statistic is still bounded by the original data range. The median cannot diverge.

    If $m = \lceil n/2 \rceil$ observations are replaced by $M \to \infty$, the median becomes $M \to \infty$. Therefore the breakdown point is:

    $$
    \varepsilon^* = \frac{\lceil n/2 \rceil}{n} \to \frac{1}{2} \text{ as } n \to \infty
    $$

    **Mean.** Replace a single observation $x_1$ by $M$. The mean becomes:

    $$
    \bar x_M = \frac{M + \sum_{i=2}^{n} x_i}{n}
    $$

    As $M \to \infty$, $\bar x_M \to \infty$. Therefore the breakdown point is $1/n \to 0$ as $n \to \infty$. $\square$

---

**Exercise 4.** The bootstrap distribution of the median can exhibit a "lumpy" appearance with repeated values. Explain why this happens (hint: consider ties in the bootstrap sample) and describe how increasing $B$ or $n$ affects the smoothness.

??? success "Solution to Exercise 4"

    The median of a bootstrap sample depends on the order statistics near the center. Since bootstrap sampling is with replacement, many resamples share the same few central observations. The median can only take values that appear in the original sample (for odd $n$) or averages of adjacent order statistics (for even $n$). This creates **discrete jumps** in the bootstrap distribution, producing the lumpy histogram.

    **Increasing $B$** (number of bootstrap resamples) does not help with lumpiness. It merely provides more draws from the same discrete distribution, refining the histogram heights but not creating new median values.

    **Increasing $n$** (sample size) helps substantially. With more distinct data values, the set of possible medians becomes denser. In the limit, the bootstrap distribution of the median converges to a continuous distribution.

    For small $n$, using a smoothed bootstrap (adding small noise to each resample) or the BCa method can mitigate discreteness. $\square$

---

**Exercise 5.** Compute the bootstrap 95% CI for both the mean and median of a sample of size $n = 30$ from a log-normal distribution with parameters $\mu = 10.5$ and $\sigma = 0.8$. Then add five outliers at $10^7$ and recompute. Discuss how the two statistics and their CIs are affected.

??? success "Solution to Exercise 5"

    ```python
    import numpy as np
    from sklearn.utils import resample

    np.random.seed(42)
    data = np.random.lognormal(mean=10.5, sigma=0.8, size=30)

    boot_means = np.array([np.mean(resample(data)) for _ in range(5000)])
    boot_medians = np.array([np.median(resample(data)) for _ in range(5000)])

    ci_mean = np.percentile(boot_means, [2.5, 97.5])
    ci_median = np.percentile(boot_medians, [2.5, 97.5])

    print("Before outliers:")
    print(f"  Mean = {data.mean():,.0f}, 95% CI: [{ci_mean[0]:,.0f}, {ci_mean[1]:,.0f}]")
    print(f"  Median = {np.median(data):,.0f}, 95% CI: [{ci_median[0]:,.0f}, {ci_median[1]:,.0f}]")

    # Add outliers
    data_out = np.append(data, [1e7] * 5)

    boot_means2 = np.array([np.mean(resample(data_out)) for _ in range(5000)])
    boot_medians2 = np.array([np.median(resample(data_out)) for _ in range(5000)])

    ci_mean2 = np.percentile(boot_means2, [2.5, 97.5])
    ci_median2 = np.percentile(boot_medians2, [2.5, 97.5])

    print("\nAfter adding 5 outliers at 10^7:")
    print(f"  Mean = {data_out.mean():,.0f}, 95% CI: [{ci_mean2[0]:,.0f}, {ci_mean2[1]:,.0f}]")
    print(f"  Median = {np.median(data_out):,.0f}, 95% CI: [{ci_median2[0]:,.0f}, {ci_median2[1]:,.0f}]")
    ```

    The mean and its CI are dramatically affected by the outliers: the mean shifts from roughly \$50,000 to over \$1,000,000, and the CI becomes enormously wide. The median and its CI change only slightly, as the five outliers shift at most a few order statistics. This demonstrates the practical value of the median for data with potential contamination. $\square$
