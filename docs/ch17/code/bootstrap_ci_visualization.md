# Bootstrap Confidence Interval Visualization

## Overview

This page demonstrates the bootstrap method for constructing confidence intervals at different confidence levels, using synthetic income data as a motivating example. The key insight is the trade-off between confidence and precision: higher confidence levels produce wider intervals. A coverage simulation verifies the long-run behavior of bootstrap CIs.

## Setting Up the Problem

Consider a right-skewed population of incomes generated from a shifted exponential distribution. We draw a small sample ($n = 20$) and want to estimate the population mean with a confidence interval.

For a population with density $f(x) = \lambda e^{-\lambda(x - c)}$ for $x \ge c$, the true mean is:

$$
\mu = c + \frac{1}{\lambda}
$$

With $c = 20{,}000$ and $\lambda^{-1} = 50{,}000$ (scale parameter), the true mean is approximately \$70,000.

```python
def simulate_income_data(n=5000, seed=3):
    """Create synthetic income data with right-skewed distribution."""
    np.random.seed(seed)
    income = np.random.exponential(scale=50000, size=n) + 20000
    return income

population = simulate_income_data()
sample = resample(population, n_samples=20, replace=False)
```

## Bootstrap Sampling Distribution

From the sample of size $n$, we generate $B$ bootstrap resamples, each drawn **with replacement**. For each resample, we compute the mean:

$$
\bar x^{*(b)} = \frac{1}{n}\sum_{i=1}^{n} x_{i}^{*(b)}, \qquad b = 1, \ldots, B
$$

The collection $\{\bar x^{*(1)}, \ldots, \bar x^{*(B)}\}$ approximates the sampling distribution of $\bar x$.

```python
def bootstrap_sampling_distribution(sample, n_bootstrap=500):
    """Generate bootstrap sampling distribution of the mean."""
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(sample)  # with replacement
        bootstrap_means.append(bootstrap_sample.mean())
    return np.array(bootstrap_means)
```

## Computing Confidence Intervals at Multiple Levels

The percentile method reads CI bounds directly from the bootstrap distribution. For a $100(1 - \alpha)\%$ CI:

$$
\text{CI} = \bigl[\hat\theta^*_{\alpha/2},\;\hat\theta^*_{1-\alpha/2}\bigr]
$$

At three standard levels:

| Level | Lower percentile | Upper percentile |
|---|---|---|
| 90% | 5th | 95th |
| 95% | 2.5th | 97.5th |
| 99% | 0.5th | 99.5th |

```python
def compute_confidence_intervals(bootstrap_dist):
    """Compute confidence intervals at multiple levels."""
    ci_90 = np.percentile(bootstrap_dist, [5, 95])
    ci_95 = np.percentile(bootstrap_dist, [2.5, 97.5])
    ci_99 = np.percentile(bootstrap_dist, [0.5, 99.5])
    return {'90%': ci_90, '95%': ci_95, '99%': ci_99}
```

## The Confidence-Precision Trade-Off

As the confidence level increases from 90% to 95% to 99%, the interval width grows. This reflects a fundamental trade-off:

- **Higher confidence** means the interval is more likely to contain $\mu$, but it becomes wider and less informative.
- **Lower confidence** gives a narrower, more precise interval, but with greater risk of missing $\mu$.

Formally, for a symmetric bootstrap distribution with standard deviation $\text{SE}$:

$$
\text{Width} \approx 2\,z_{1-\alpha/2}\cdot\text{SE}
$$

Since $z_{0.95} = 1.645 < z_{0.975} = 1.960 < z_{0.995} = 2.576$, the 99% CI is roughly $2.576/1.645 \approx 1.57$ times as wide as the 90% CI.

## Coverage Simulation

A single CI either contains $\mu$ or it does not -- there is no probability attached to a realized interval. The stated confidence level describes long-run behavior. A coverage simulation verifies this:

1. Repeat the entire procedure (draw sample, bootstrap, build CI) $N$ times.
2. Record the fraction of intervals that contain the true mean.

$$
\widehat{\text{coverage}} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\!\bigl(\mu \in \text{CI}_i\bigr)
$$

```python
def simulate_coverage(population, true_mean, n_samples=20,
                      n_bootstrap=500, n_simulations=100):
    """Simulate coverage of bootstrap CIs."""
    coverage_90, coverage_95 = [], []
    for _ in range(n_simulations):
        sample = np.random.choice(population, size=n_samples, replace=False)
        boot_means = np.array([
            np.mean(np.random.choice(sample, size=len(sample), replace=True))
            for _ in range(n_bootstrap)
        ])
        ci_90 = np.percentile(boot_means, [5, 95])
        ci_95 = np.percentile(boot_means, [2.5, 97.5])
        coverage_90.append(ci_90[0] <= true_mean <= ci_90[1])
        coverage_95.append(ci_95[0] <= true_mean <= ci_95[1])
    return 100 * np.mean(coverage_90), 100 * np.mean(coverage_95)
```

For the exponential income data with $n = 20$, the actual coverage of the percentile 95% CI may be slightly below 95% due to the skewness of the distribution. Larger sample sizes or BCa corrections improve coverage.

## Interpretation

- The bootstrap sampling distribution of the mean is approximately normal even when the population is skewed, provided $n$ is not too small (a manifestation of the CLT).
- The percentile CI is the simplest bootstrap interval but may under-cover for small, skewed samples.
- The coverage simulation confirms that stated confidence levels are long-run frequencies, not probabilities for individual intervals.
- For income-like data (exponential, log-normal), the mean and median can differ substantially; consider which estimand is more relevant to the research question.

## Exercises

**Exercise 1.** Using the same population, increase the sample size from $n = 20$ to $n = 100$. How does the width of the 95% bootstrap CI change? Verify that the width decreases approximately as $1/\sqrt{n}$.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from sklearn.utils import resample

    population = simulate_income_data()

    for n in [20, 50, 100, 200]:
        np.random.seed(3)
        sample = resample(population, n_samples=n, replace=False)
        boot_means = bootstrap_sampling_distribution(sample, n_bootstrap=500)
        ci = np.percentile(boot_means, [2.5, 97.5])
        width = ci[1] - ci[0]
        print(f"n = {n:>3}: 95% CI width = ${width:,.0f}")
    ```

    The width decreases approximately as $1/\sqrt{n}$. Doubling $n$ from 50 to 100 should reduce the width by a factor of $\sqrt{2} \approx 1.41$. This follows from the CLT: $\text{SE}(\bar x) = \sigma/\sqrt{n}$, so the CI width is proportional to $2z_{0.975}\sigma/\sqrt{n}$. $\square$

---

**Exercise 2.** Modify the coverage simulation to compare the percentile method with the normal bootstrap interval ($\bar x \pm z_{1-\alpha/2}\cdot\widehat{\text{SE}}_{\text{boot}}$). Which achieves better coverage for the skewed income data with $n = 20$?

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(42)
    population = simulate_income_data()
    true_mean = population.mean()
    z = stats.norm.ppf(0.975)

    cover_pct, cover_normal = 0, 0
    N = 500
    for _ in range(N):
        sample = np.random.choice(population, 20, replace=False)
        boot_means = np.array([
            np.mean(np.random.choice(sample, 20, True)) for _ in range(500)
        ])
        # Percentile
        ci_pct = np.percentile(boot_means, [2.5, 97.5])
        if ci_pct[0] <= true_mean <= ci_pct[1]:
            cover_pct += 1
        # Normal
        se = boot_means.std(ddof=1)
        ci_norm = (sample.mean() - z * se, sample.mean() + z * se)
        if ci_norm[0] <= true_mean <= ci_norm[1]:
            cover_normal += 1

    print(f"Percentile coverage: {cover_pct / N:.3f}")
    print(f"Normal coverage:     {cover_normal / N:.3f}")
    ```

    For skewed data with small $n$, the normal method often achieves slightly better coverage (around 92--94%) than the percentile method (around 89--92%), because the normal interval is symmetric and the CLT provides a reasonable approximation for the mean even at $n = 20$. However, neither achieves the full 95%. The BCa method would be a better choice for accurate coverage with skewed data. $\square$

---

**Exercise 3.** Derive the relationship between the number of bootstrap resamples $B$ and the Monte Carlo error in the CI endpoints. Specifically, show that the standard deviation of the estimated $q$-th quantile of the bootstrap distribution is approximately:

$$
\text{SD}(\hat\theta^*_q) \approx \frac{\sqrt{q(1-q)}}{f(\hat\theta^*_q)\sqrt{B}}
$$

where $f$ is the density of the bootstrap distribution at the quantile.

??? success "Solution to Exercise 3"

    Let $F_B^*$ be the empirical CDF of $B$ bootstrap replicates, and let $\hat\theta^*_q = F_B^{*-1}(q)$. The number of bootstrap values below $\hat\theta^*_q$ follows approximately a $\text{Binomial}(B, q)$ distribution.

    By the delta method applied to the quantile function $F^{*-1}$ at level $q$:

    $$
    \text{Var}(\hat\theta^*_q) \approx \frac{q(1-q)}{B\,[f^*(\theta^*_q)]^2}
    $$

    where $f^*$ is the density of the bootstrap distribution. Taking the square root:

    $$
    \text{SD}(\hat\theta^*_q) \approx \frac{\sqrt{q(1-q)}}{f^*(\theta^*_q)\sqrt{B}}
    $$

    For a 95% CI, the endpoints use $q = 0.025$ and $q = 0.975$, so $\sqrt{q(1-q)} = \sqrt{0.025 \times 0.975} \approx 0.156$. If the bootstrap distribution is approximately $N(\hat\theta, \text{SE}^2)$, then $f^*(\theta^*_{0.025}) \approx 0.058/\text{SE}$, and:

    $$
    \text{SD} \approx \frac{0.156\,\text{SE}}{0.058\sqrt{B}} = \frac{2.69\,\text{SE}}{\sqrt{B}}
    $$

    With $B = 500$, this is about $0.12\,\text{SE}$. Increasing to $B = 10{,}000$ reduces it to about $0.027\,\text{SE}$. $\square$

---

**Exercise 4.** The coverage simulation uses $N = 100$ repetitions. Compute a 95% confidence interval for the coverage probability itself. (Hint: the coverage indicator is Bernoulli, so use the normal approximation for a proportion.)

??? success "Solution to Exercise 4"

    Let $\hat p$ be the observed coverage proportion from $N = 100$ simulations. Each simulation produces a Bernoulli outcome (CI covers or not), so:

    $$
    \text{SE}(\hat p) = \sqrt{\frac{\hat p(1 - \hat p)}{N}}
    $$

    The 95% CI for the true coverage is:

    $$
    \hat p \pm 1.96\sqrt{\frac{\hat p(1 - \hat p)}{N}}
    $$

    If the observed 95% CI coverage is $\hat p = 0.93$ from $N = 100$:

    $$
    \text{SE} = \sqrt{\frac{0.93 \times 0.07}{100}} = \sqrt{0.000651} \approx 0.0255
    $$

    $$
    \text{CI for coverage} = 0.93 \pm 1.96 \times 0.0255 = [0.880, 0.980]
    $$

    This interval includes 0.95, so we cannot conclude that the bootstrap under-covers. To detect under-coverage more precisely, increase $N$ to 1000 or more. $\square$

---

**Exercise 5.** Explain why the bootstrap is particularly valuable for skewed distributions like income data. Compare the bootstrap CI for the mean with the bootstrap CI for the median on the same sample, and discuss which is more appropriate when the goal is to describe "typical" income.

??? success "Solution to Exercise 5"

    For skewed distributions, the sampling distribution of the mean is itself skewed (especially for small $n$), making symmetric parametric intervals ($\bar x \pm t \cdot s/\sqrt{n}$) inaccurate. The bootstrap captures this asymmetry automatically because it generates the actual shape of the sampling distribution from the data.

    ```python
    import numpy as np
    from sklearn.utils import resample

    np.random.seed(3)
    population = simulate_income_data()
    sample = resample(population, n_samples=20, replace=False)

    boot_means = [np.mean(resample(sample)) for _ in range(1000)]
    boot_medians = [np.median(resample(sample)) for _ in range(1000)]

    ci_mean = np.percentile(boot_means, [2.5, 97.5])
    ci_median = np.percentile(boot_medians, [2.5, 97.5])

    print(f"Sample mean:   ${np.mean(sample):,.0f}")
    print(f"Sample median: ${np.median(sample):,.0f}")
    print(f"95% CI (mean):   [${ci_mean[0]:,.0f}, ${ci_mean[1]:,.0f}]")
    print(f"95% CI (median): [${ci_median[0]:,.0f}, ${ci_median[1]:,.0f}]")
    ```

    For income data, the mean is pulled upward by high earners, so it overestimates what a "typical" person earns. The median is robust to outliers and better represents the center of the distribution. The bootstrap CI for the median is particularly valuable because there is no simple closed-form SE for the median, whereas the mean has $\text{SE} = \sigma/\sqrt{n}$.

    If the goal is policy-oriented (e.g., "what does the average citizen earn?"), the median with its bootstrap CI is more informative and more honest about "typical" income. $\square$
