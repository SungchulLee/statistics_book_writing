# Bias-Corrected and Accelerated Bootstrap

## Motivation

The percentile method reads confidence interval endpoints directly from the bootstrap distribution. While simple, it can suffer from poor coverage when the bootstrap distribution is biased or when the standard error of $\hat{\theta}$ depends on $\theta$ (i.e., the sampling distribution is skewed). The **bias-corrected and accelerated (BCa) method**, introduced by Efron (1987), adjusts the percentile interval to correct for both of these problems.

The BCa interval is widely regarded as the best general-purpose bootstrap confidence interval. It achieves **second-order accuracy** (coverage error of $O(n^{-1})$ instead of $O(n^{-1/2})$) while retaining the transformation invariance of the percentile method.

## The BCa Interval

The BCa interval has the same form as the percentile interval but uses **adjusted quantiles**:

$$
\left[\hat{\theta}^*_{(\alpha_1)}, \quad \hat{\theta}^*_{(\alpha_2)}\right]
$$

where $\alpha_1$ and $\alpha_2$ replace the simple $\alpha/2$ and $1 - \alpha/2$ of the percentile method. These adjusted levels are:

$$
\alpha_1 = \mathcal{N}\!\left(\hat{z}_0 + \frac{\hat{z}_0 + z_{\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{\alpha/2})}\right)
$$

$$
\alpha_2 = \mathcal{N}\!\left(\hat{z}_0 + \frac{\hat{z}_0 + z_{1-\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{1-\alpha/2})}\right)
$$

Here $\mathcal{N}$ is the standard normal CDF, $z_q = \mathcal{N}^{-1}(q)$ is the $q$-th standard normal quantile, and the two correction factors $\hat{z}_0$ and $\hat{a}$ are defined below.

## The Bias Correction Factor

The **bias correction** $\hat{z}_0$ measures how far the center of the bootstrap distribution is from the observed estimate $\hat{\theta}$. It is defined as:

$$
\hat{z}_0 = \mathcal{N}^{-1}\!\left(\frac{\#\{\hat{\theta}^{*(b)} < \hat{\theta}\}}{B}\right)
$$

This is the proportion of bootstrap replicates that fall below the observed statistic, converted to a z-score. If the bootstrap distribution is centered exactly at $\hat{\theta}$, then half the replicates fall below, and $\hat{z}_0 = \mathcal{N}^{-1}(0.5) = 0$. In that case, the bias correction has no effect.

When $\hat{z}_0 \neq 0$, the bootstrap distribution is biased relative to $\hat{\theta}$. The BCa interval shifts the quantile cutoffs to compensate.

!!! note "Interpreting the Bias Correction"
    A positive $\hat{z}_0$ means that more than half the bootstrap replicates exceed $\hat{\theta}$, indicating the bootstrap distribution is shifted upward relative to the observed estimate. A negative $\hat{z}_0$ indicates a downward shift.

## The Acceleration Factor

The **acceleration** $\hat{a}$ measures how the standard error of $\hat{\theta}$ changes as $\theta$ varies. When the standard error is constant (independent of $\theta$), the acceleration is zero and the BCa interval reduces to the bias-corrected (BC) interval. When the standard error depends on $\theta$, the acceleration adjusts the quantile cutoffs to account for skewness.

The acceleration is typically estimated using the **jackknife**:

$$
\hat{a} = \frac{\sum_{i=1}^{n}\left(\bar{\hat{\theta}}_{(\cdot)} - \hat{\theta}_{(-i)}\right)^3}{6\left[\sum_{i=1}^{n}\left(\bar{\hat{\theta}}_{(\cdot)} - \hat{\theta}_{(-i)}\right)^2\right]^{3/2}}
$$

where $\hat{\theta}_{(-i)} = g(x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n)$ is the statistic computed with the $i$-th observation removed, and $\bar{\hat{\theta}}_{(\cdot)} = \frac{1}{n}\sum_{i=1}^{n}\hat{\theta}_{(-i)}$ is the average of the jackknife values.

The numerator captures the skewness of the jackknife distribution, and the denominator normalizes it. This formula is a consistent estimator of the acceleration constant in the underlying transformation model.

## Algorithm

1. Compute the observed statistic $\hat{\theta}$ from the original sample
2. Generate $B$ bootstrap replicates $\hat{\theta}^{*(1)}, \ldots, \hat{\theta}^{*(B)}$
3. Compute the **bias correction** $\hat{z}_0$:
    - Count the proportion of replicates below $\hat{\theta}$
    - Convert to a z-score via $\hat{z}_0 = \mathcal{N}^{-1}(\text{proportion})$
4. Compute the **acceleration** $\hat{a}$:
    - For each $i = 1, \ldots, n$: compute $\hat{\theta}_{(-i)}$ (leave-one-out)
    - Apply the skewness formula above
5. Compute adjusted quantile levels $\alpha_1$ and $\alpha_2$
6. The BCa interval is $[\hat{\theta}^*_{(\alpha_1)}, \hat{\theta}^*_{(\alpha_2)}]$

!!! warning "Computational Cost"
    The jackknife step requires computing $\hat{\theta}$ a total of $n$ additional times (once for each leave-one-out sample). For expensive statistics or large $n$, this can be a substantial overhead beyond the $B$ bootstrap replicates.

## Special Cases

When $\hat{z}_0 = 0$ and $\hat{a} = 0$, the adjusted quantiles reduce to $\alpha_1 = \alpha/2$ and $\alpha_2 = 1 - \alpha/2$, recovering the ordinary percentile interval.

When $\hat{a} = 0$ but $\hat{z}_0 \neq 0$, the method is called the **bias-corrected (BC) interval**. It corrects for bias but not for skewness.

When both corrections are active, the BCa interval can produce substantially different endpoints from the percentile interval, especially for statistics with skewed sampling distributions (variance, odds ratios, correlation coefficients near $\pm 1$).

## Theoretical Properties

The BCa interval achieves **second-order accuracy**: the coverage probability satisfies:

$$
P(\theta \in \text{BCa interval}) = 1 - \alpha + O(n^{-1})
$$

compared to $1 - \alpha + O(n^{-1/2})$ for the percentile interval. This means the coverage error shrinks faster as the sample size grows.

The BCa interval is also **transformation invariant**: for any monotone increasing function $m$, the BCa interval for $\phi = m(\theta)$ is exactly $[m(L), m(U)]$ where $[L, U]$ is the BCa interval for $\theta$.

!!! tip "Why Second-Order Accuracy Matters"
    For a 95% interval with $n = 20$, first-order accuracy might give actual coverage of 90%, while second-order accuracy typically gives 93-95%. The improvement is most noticeable for moderate sample sizes and skewed statistics.

## Example: BCa Interval for the Variance

Consider a sample of $n = 15$ observations from a right-skewed distribution. The sample variance is $s^2 = 8.4$.

1. Generate $B = 10{,}000$ bootstrap replicates of $s^2$
2. Suppose 62% of the replicates fall below $s^2 = 8.4$, so $\hat{z}_0 = \mathcal{N}^{-1}(0.62) = 0.305$
3. Compute the $n = 15$ jackknife values $s^2_{(-1)}, \ldots, s^2_{(-15)}$ and find $\hat{a} = 0.042$
4. The adjusted quantiles for a 95% interval are:

$$
\alpha_1 = \mathcal{N}\!\left(0.305 + \frac{0.305 + (-1.96)}{1 - 0.042(0.305 + (-1.96))}\right) \approx \mathcal{N}(-1.44) \approx 0.075
$$

$$
\alpha_2 = \mathcal{N}\!\left(0.305 + \frac{0.305 + 1.96}{1 - 0.042(0.305 + 1.96)}\right) \approx \mathcal{N}(2.65) \approx 0.996
$$

The BCa interval uses the 7.5th and 99.6th percentiles of the bootstrap distribution instead of the standard 2.5th and 97.5th. This shift upward reflects both the positive bias correction and the positive acceleration, producing a wider and higher interval appropriate for the right-skewed distribution of $s^2$.

## Summary

The BCa method improves upon the percentile interval by adjusting the quantile cutoffs using two correction factors: the bias correction $\hat{z}_0$ (measuring median bias in the bootstrap distribution) and the acceleration $\hat{a}$ (measuring how the standard error varies with the parameter, estimated via jackknife). These corrections yield second-order accurate, transformation-invariant confidence intervals. The BCa interval is the recommended default when computational cost permits the additional jackknife calculations.

## Exercises

**Exercise 1.**
Explain the difference between the **percentile** and **BCa** bootstrap confidence intervals. When does the BCa interval substantially differ from the percentile interval?

## Computation
