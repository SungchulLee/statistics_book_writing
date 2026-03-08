# Bootstrap Methods

## Motivation

Many statistical procedures require knowing the sampling distribution of an estimator $\hat{\theta}$. In simple cases (e.g., the sample mean from a normal population), the sampling distribution has a known closed form. But for complex statistics — medians, correlation coefficients, regression coefficients with heteroscedasticity, ratio estimators — the exact sampling distribution may be unknown or intractable.

The **bootstrap**, introduced by Bradley Efron (1979), solves this problem by using the data itself to approximate the sampling distribution.

## The Bootstrap Principle

The key insight is a substitution: we replace the unknown population distribution $F$ with the **empirical distribution function** $\hat{F}_n$, which places mass $1/n$ on each observed data point.

**Population world:** $X_1, \ldots, X_n \overset{\text{iid}}{\sim} F$ → compute $\hat{\theta} = g(X_1, \ldots, X_n)$

**Bootstrap world:** $X_1^*, \ldots, X_n^* \overset{\text{iid}}{\sim} \hat{F}_n$ → compute $\hat{\theta}^* = g(X_1^*, \ldots, X_n^*)$

In practice, sampling from $\hat{F}_n$ means **sampling with replacement** from the original data $\{x_1, \ldots, x_n\}$.

## The Nonparametric Bootstrap Algorithm

1. **Observe** the original sample $x_1, x_2, \ldots, x_n$
2. **Repeat** for $b = 1, 2, \ldots, B$:
    - Draw a **bootstrap sample** $x_1^*, x_2^*, \ldots, x_n^*$ by sampling **with replacement** from $\{x_1, \ldots, x_n\}$
    - Compute the **bootstrap replicate** $\hat{\theta}^{*(b)} = g(x_1^*, \ldots, x_n^*)$
3. **Use** the collection $\{\hat{\theta}^{*(1)}, \ldots, \hat{\theta}^{*(B)}\}$ to approximate the sampling distribution of $\hat{\theta}$

!!! note "Bootstrap Sample Size"
    Each bootstrap sample has the same size $n$ as the original data. Some observations may appear multiple times; others may not appear at all. On average, each bootstrap sample contains about $1 - (1 - 1/n)^n \approx 1 - e^{-1} \approx 63.2\%$ of the unique original observations.

## Bootstrap Standard Error

The simplest bootstrap application is estimating the standard error of $\hat{\theta}$:

$$
\widehat{\text{SE}}_{\text{boot}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^B \left(\hat{\theta}^{*(b)} - \bar{\hat{\theta}}^*\right)^2}
$$

where $\bar{\hat{\theta}}^* = \frac{1}{B}\sum_{b=1}^B \hat{\theta}^{*(b)}$.

This works for *any* statistic, not just the mean. Typical choices: $B = 1{,}000$ to $10{,}000$.

## Bootstrap Confidence Intervals

### Method 1: Normal Interval (Bootstrap SE)

If $\hat{\theta}$ is approximately normal:

$$
\hat{\theta} \pm z_{\alpha/2} \cdot \widehat{\text{SE}}_{\text{boot}}
$$

This is the simplest but assumes normality of the sampling distribution.

### Method 2: Percentile Interval

Use the quantiles of the bootstrap distribution directly:

$$
\left[\hat{\theta}^*_{(\alpha/2)}, \quad \hat{\theta}^*_{(1-\alpha/2)}\right]
$$

where $\hat{\theta}^*_{(q)}$ is the $q$-th quantile of the bootstrap replicates.

**Advantages:** Respects the shape of the sampling distribution (e.g., skewness), automatically handles transformations, and the interval stays within natural bounds.

**Disadvantage:** Can have poor coverage when the bootstrap distribution is biased.

### Method 3: Basic (Pivotal) Bootstrap

Based on the pivot $\hat{\theta}^* - \hat{\theta}$:

$$
\left[2\hat{\theta} - \hat{\theta}^*_{(1-\alpha/2)}, \quad 2\hat{\theta} - \hat{\theta}^*_{(\alpha/2)}\right]
$$

Note the reversal of quantiles. This corrects for bias in the bootstrap distribution.

### Method 4: Bias-Corrected and Accelerated (BCa)

The **BCa interval** adjusts for both bias and skewness:

$$
\left[\hat{\theta}^*_{(\alpha_1)}, \quad \hat{\theta}^*_{(\alpha_2)}\right]
$$

where $\alpha_1$ and $\alpha_2$ are modified percentiles:

$$
\alpha_1 = \Phi\left(\hat{z}_0 + \frac{\hat{z}_0 + z_{\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{\alpha/2})}\right)
$$

$$
\alpha_2 = \Phi\left(\hat{z}_0 + \frac{\hat{z}_0 + z_{1-\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{1-\alpha/2})}\right)
$$

Here $\hat{z}_0$ is the **bias correction** (proportion of bootstrap replicates below $\hat{\theta}$, converted to z-score) and $\hat{a}$ is the **acceleration** (estimated via jackknife). BCa has better theoretical coverage properties than percentile or basic intervals.

### Comparison of Bootstrap CI Methods

| Method | Bias Correction | Skewness Adjustment | Theoretical Order |
|---|---|---|---|
| Normal | ✗ | ✗ | First-order |
| Percentile | Partial | Partial | First-order |
| Basic (Pivotal) | ✓ | ✗ | First-order |
| BCa | ✓ | ✓ | Second-order |

## Bootstrap Hypothesis Testing

### One-Sample Test

To test $H_0: \theta = \theta_0$:

1. Compute the observed test statistic $t_{\text{obs}} = \hat{\theta} - \theta_0$
2. Generate bootstrap replicates $\hat{\theta}^{*(1)}, \ldots, \hat{\theta}^{*(B)}$
3. Center the bootstrap distribution: $t^{*(b)} = \hat{\theta}^{*(b)} - \hat{\theta}$
4. The bootstrap p-value is:

$$
p = \frac{1}{B}\sum_{b=1}^B \mathbf{1}\left(|t^{*(b)}| \geq |t_{\text{obs}}|\right)
$$

### Two-Sample Test

To test $H_0: \theta_X = \theta_Y$ (e.g., equal means):

1. Compute $t_{\text{obs}} = \hat{\theta}_X - \hat{\theta}_Y$
2. **Pool** the samples under $H_0$: combine $x_1, \ldots, x_m$ and $y_1, \ldots, y_n$
3. For each bootstrap replicate, draw $m$ observations for "group X" and $n$ for "group Y" from the pooled data
4. Compute $t^{*(b)} = \hat{\theta}^*_X - \hat{\theta}^*_Y$
5. p-value: proportion of $|t^{*(b)}| \geq |t_{\text{obs}}|$

## Parametric Bootstrap

Instead of resampling from $\hat{F}_n$, the **parametric bootstrap** assumes a parametric model $F_{\hat{\theta}}$ and generates bootstrap samples from the *fitted* distribution.

**Algorithm:**

1. Fit the parametric model: estimate $\hat{\theta}$ from the data
2. Generate bootstrap samples from $F_{\hat{\theta}}$ (not from the data directly)
3. Proceed as in the nonparametric bootstrap

**When to use:** When you have a specific distributional model and want to exploit it for better efficiency. The parametric bootstrap produces tighter confidence intervals when the model is correct, but is invalid if the model is misspecified.

## The Jackknife

The **jackknife** (Quenouille, 1949; Tukey, 1958) predates the bootstrap and is based on *leave-one-out* resampling.

**Jackknife replicates:**

$$
\hat{\theta}_{(-i)} = g(x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n)
$$

**Jackknife estimate of bias:**

$$
\widehat{\text{Bias}}_{\text{jack}} = (n-1)\left(\bar{\hat{\theta}}_{(\cdot)} - \hat{\theta}\right)
$$

where $\bar{\hat{\theta}}_{(\cdot)} = \frac{1}{n}\sum_{i=1}^n \hat{\theta}_{(-i)}$.

**Jackknife standard error:**

$$
\widehat{\text{SE}}_{\text{jack}} = \sqrt{\frac{n-1}{n}\sum_{i=1}^n \left(\hat{\theta}_{(-i)} - \bar{\hat{\theta}}_{(\cdot)}\right)^2}
$$

The jackknife is deterministic (no random resampling), which is sometimes advantageous. However, it fails for non-smooth statistics like the median.

## When the Bootstrap Fails

The bootstrap is not universally valid. It can fail when:

1. **Extreme order statistics.** Estimating $\theta$ in $\text{Uniform}(0, \theta)$ by $\hat{\theta} = X_{(n)}$ — the bootstrap distribution of $X^*_{(n)}$ does not mimic the true distribution of $X_{(n)}$.

2. **Heavy-tailed distributions.** If the population has infinite variance, $\bar{X}$ does not have a normal distribution, and the bootstrap may give misleading intervals.

3. **Dependent data.** Standard bootstrap assumes iid data. For time series, use the **block bootstrap** (see below).

4. **Small sample sizes.** With very small $n$ (say $n < 10$), $\hat{F}_n$ is a poor approximation to $F$.

## Block Bootstrap for Dependent Data

For time series or spatially dependent data, the standard iid bootstrap destroys the dependence structure. The **block bootstrap** (Kunsch, 1989; Liu and Singh, 1992) preserves it:

1. Divide the data into overlapping blocks of length $\ell$
2. Resample *blocks* with replacement
3. Concatenate to form a bootstrap sample

The **moving block bootstrap** uses all $n - \ell + 1$ overlapping blocks. The **circular block bootstrap** wraps the data around to ensure all observations appear in the same number of blocks.

Choosing the block length $\ell$ is crucial: too small destroys dependence, too large reduces the number of effective resamples. Common choices: $\ell \approx n^{1/3}$.
