# Bootstrapping as an Alternative

## Motivation

The earlier sections of this chapter provide tools for detecting non-normality -- Q-Q plots, the Shapiro-Wilk test, and measures of skewness and kurtosis. When these diagnostics reveal that the data deviate substantially from the normal distribution, the standard parametric methods (confidence intervals based on $t$-distributions, hypothesis tests relying on $\chi^2$ or $F$-distributions) may no longer control their stated error rates. Bootstrapping offers a way to perform inference without assuming a specific parametric form for the population distribution. Instead of deriving a sampling distribution from theoretical assumptions, the bootstrap **estimates** the sampling distribution by resampling from the observed data itself.

## The Bootstrap Principle

The core idea behind the bootstrap is conceptually simple. In classical statistics, we imagine drawing repeated samples from the unknown population distribution $F$ to understand the variability of a statistic $T = T(X_1, X_2, \ldots, X_n)$. Since we cannot actually draw new samples from $F$, the bootstrap substitutes the empirical distribution function $\hat{F}_n$ -- which places probability $1/n$ on each observed data point -- in place of $F$.

Sampling from $\hat{F}_n$ is equivalent to sampling with replacement from the observed data. Each bootstrap sample $X_1^*, X_2^*, \ldots, X_n^*$ has the same size as the original sample and is drawn with replacement, meaning some observations may appear multiple times while others may not appear at all.

### Algorithm

The **nonparametric bootstrap** proceeds as follows:

1. Start with the observed sample $\mathbf{x} = (x_1, x_2, \ldots, x_n)$.
2. For $b = 1, 2, \ldots, B$:
    - Draw a bootstrap sample $\mathbf{x}^{*b} = (x_1^{*b}, x_2^{*b}, \ldots, x_n^{*b})$ by sampling $n$ values from $\mathbf{x}$ with replacement.
    - Compute the bootstrap replicate $\hat{\theta}^{*b} = T(\mathbf{x}^{*b})$.
3. The collection $\{\hat{\theta}^{*1}, \hat{\theta}^{*2}, \ldots, \hat{\theta}^{*B}\}$ approximates the sampling distribution of $T$.

The number of bootstrap replicates $B$ is typically chosen to be at least 1000 for standard error estimation and at least 5000--10000 for confidence interval construction.

## Bootstrap Confidence Intervals

### Percentile Method

The simplest bootstrap confidence interval takes the $\alpha/2$ and $1 - \alpha/2$ quantiles of the bootstrap distribution directly:

$$
\text{CI}_{1-\alpha} = \left[\hat{\theta}^*_{(\alpha/2)},\; \hat{\theta}^*_{(1-\alpha/2)}\right]
$$

For a 95% confidence interval, this gives the 2.5th and 97.5th percentiles of the $B$ bootstrap replicates.

!!! warning "Limitations of the percentile method"

    The percentile method is intuitive but can have poor coverage when the bootstrap distribution is skewed or when the statistic is biased. It is not transformation-invariant, meaning that a monotone transformation of the statistic can change the coverage probability.

### BCa (Bias-Corrected and Accelerated) Method

The **BCa method** adjusts for both bias and skewness in the bootstrap distribution. It modifies the percentile endpoints using two correction factors:

- **Bias correction** $\hat{z}_0$: measures how far the median of the bootstrap distribution is from the original estimate $\hat{\theta}$
- **Acceleration** $\hat{a}$: accounts for the rate at which the standard error of $\hat{\theta}$ changes with the true parameter value

The adjusted percentiles are:

$$
\alpha_1 = \Phi\!\left(\hat{z}_0 + \frac{\hat{z}_0 + z_{\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{\alpha/2})}\right), \quad \alpha_2 = \Phi\!\left(\hat{z}_0 + \frac{\hat{z}_0 + z_{1-\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{1-\alpha/2})}\right)
$$

where $\Phi$ is the standard normal CDF and $z_{\alpha/2}$ is the standard normal quantile. When both $\hat{z}_0 = 0$ and $\hat{a} = 0$, the BCa method reduces to the percentile method. In practice, BCa intervals generally provide better coverage than the simple percentile method.

## Worked Example

Consider the following 12 observations of annual returns (%) from a small-cap fund:

$$
\mathbf{x} = \{-8.2,\; 3.1,\; 15.7,\; 2.4,\; -1.5,\; 22.3,\; 6.8,\; -3.4,\; 11.2,\; 1.9,\; 18.6,\; 7.5\}
$$

The sample mean is $\bar{x} = 6.37\%$. A Q-Q plot reveals a right-skewed distribution, making the standard $t$-interval questionable. We apply the bootstrap to construct a 95% confidence interval for the population mean.

**Step 1.** Draw $B = 5000$ bootstrap samples of size 12 with replacement.

**Step 2.** Compute the mean of each bootstrap sample, producing the bootstrap distribution $\{\hat{\theta}^{*1}, \ldots, \hat{\theta}^{*5000}\}$.

**Step 3.** Sort the bootstrap means and extract the 2.5th and 97.5th percentiles.

Suppose the bootstrap procedure yields the sorted percentiles $\hat{\theta}^*_{(0.025)} = 1.85$ and $\hat{\theta}^*_{(0.975)} = 11.02$. The 95% percentile bootstrap confidence interval for the population mean return is $[1.85\%, \; 11.02\%]$.

For comparison, the standard $t$-interval is $\bar{x} \pm t_{0.025, 11} \cdot s / \sqrt{12} = 6.37 \pm 2.201 \times 8.92 / 3.464 = 6.37 \pm 5.66$, giving $[0.71\%, \; 12.03\%]$. The bootstrap interval is narrower and slightly shifted, reflecting the skewness in the data that the symmetric $t$-interval cannot capture.

## When the Bootstrap Works

The bootstrap is a powerful tool, but it is not universally applicable. Its validity depends on several conditions:

- **Sample representativeness.** The bootstrap resamples from the observed data, so the original sample must be representative of the population. With very small samples ($n < 15$), the empirical distribution may be a poor approximation of $F$, leading to unreliable bootstrap intervals.

- **Smoothness of the statistic.** The bootstrap works well for statistics that are smooth functions of the data (means, regression coefficients, correlation coefficients). For non-smooth statistics like the sample maximum, the standard nonparametric bootstrap can fail.

- **Independence.** The standard bootstrap assumes that observations are independent. For time series or clustered data, the block bootstrap or cluster bootstrap must be used instead.

!!! tip "Bootstrap vs. transformations vs. nonparametric tests"

    When normality fails, three alternatives are available: (1) transform the data to achieve approximate normality, then apply standard methods; (2) use nonparametric tests that make no distributional assumptions; (3) use the bootstrap. Transformations preserve parametric interpretability but require finding an appropriate transformation. Nonparametric tests are robust but often limited to testing hypotheses (not constructing confidence intervals for arbitrary parameters). The bootstrap is the most flexible option, applicable to virtually any statistic, but requires moderate sample sizes and more computation.

## Python Implementation

```python
"""
Bootstrap confidence interval for a sample mean.

Demonstrates the nonparametric bootstrap procedure on skewed data
and compares the result with the standard t-interval.
"""

import numpy as np

# ===================================================================
# Data
# ===================================================================
data = np.array([-8.2, 3.1, 15.7, 2.4, -1.5, 22.3,
                  6.8, -3.4, 11.2, 1.9, 18.6, 7.5])

# ===================================================================
# Bootstrap procedure
# ===================================================================
rng = np.random.default_rng(42)
B = 5000
n = len(data)

bootstrap_means = np.array([
    np.mean(rng.choice(data, size=n, replace=True))
    for _ in range(B)
])

# ===================================================================
# Percentile confidence interval
# ===================================================================
ci_lower, ci_upper = np.percentile(bootstrap_means, [2.5, 97.5])

if __name__ == "__main__":
    print(f"Sample mean: {np.mean(data):.2f}%")
    print(f"Bootstrap SE: {np.std(bootstrap_means, ddof=0):.2f}%")
    print(f"95% Percentile CI: [{ci_lower:.2f}%, {ci_upper:.2f}%]")
```

The bootstrap distribution of the sample mean reflects the skewness in the original data. Because the resampling procedure makes no assumption about the population distribution, the resulting confidence interval adapts to the actual shape of the sampling distribution rather than forcing symmetry.
