# Non-Parametric Bootstrap

## Motivation

The nonparametric bootstrap is the most widely used form of the bootstrap. It makes no assumptions about the underlying population distribution $F$ and works for virtually any statistic. Given an observed sample, the procedure generates an approximate sampling distribution through repeated resampling with replacement.

This section details the algorithm, derives the key quantities it produces (standard errors, bias estimates, bootstrap distributions), and clarifies when and why the method works.

## The Algorithm

Given an observed iid sample $x_1, x_2, \ldots, x_n$ and a statistic $\hat{\theta} = g(x_1, \ldots, x_n)$:

1. **Set** the number of bootstrap replicates $B$ (typically $B = 1{,}000$ to $10{,}000$)
2. **For** $b = 1, 2, \ldots, B$:
    - Draw a bootstrap sample $x_1^*, x_2^*, \ldots, x_n^*$ by sampling $n$ observations **with replacement** from $\{x_1, \ldots, x_n\}$
    - Compute the bootstrap replicate $\hat{\theta}^{*(b)} = g(x_1^*, \ldots, x_n^*)$
3. **Collect** the bootstrap distribution $\{\hat{\theta}^{*(1)}, \hat{\theta}^{*(2)}, \ldots, \hat{\theta}^{*(B)}\}$

The collection of $B$ bootstrap replicates forms the **bootstrap distribution** of $\hat{\theta}^*$, which serves as an approximation to the true sampling distribution of $\hat{\theta}$.

!!! note "Same Sample Size"
    Each bootstrap sample has the same size $n$ as the original data. Drawing fewer or more observations would change the sampling variability and invalidate the approximation.

## Bootstrap Standard Error

The bootstrap estimate of the standard error of $\hat{\theta}$ is the sample standard deviation of the bootstrap replicates:

$$
\widehat{\text{SE}}_{\text{boot}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\left(\hat{\theta}^{*(b)} - \bar{\hat{\theta}}^*\right)^2}
$$

where $\bar{\hat{\theta}}^* = \frac{1}{B}\sum_{b=1}^{B}\hat{\theta}^{*(b)}$ is the mean of the bootstrap distribution.

This formula applies to any statistic: means, medians, correlation coefficients, regression parameters, or any function of the data.

## Bootstrap Bias Estimation

The bootstrap also provides an estimate of the bias of $\hat{\theta}$ as an estimator of $\theta$:

$$
\widehat{\text{Bias}}_{\text{boot}} = \bar{\hat{\theta}}^* - \hat{\theta}
$$

The logic is as follows. In the real world, the bias is $E_F[\hat{\theta}] - \theta$. In the bootstrap world, $\hat{\theta}$ plays the role of the true parameter and $\hat{\theta}^*$ plays the role of the estimator, so the bootstrap bias is $E^*[\hat{\theta}^*] - \hat{\theta}$, which we estimate by $\bar{\hat{\theta}}^* - \hat{\theta}$.

A **bias-corrected** estimator can be constructed as:

$$
\hat{\theta}_{\text{corrected}} = \hat{\theta} - \widehat{\text{Bias}}_{\text{boot}} = 2\hat{\theta} - \bar{\hat{\theta}}^*
$$

!!! warning "Bias Correction Can Increase Variance"
    While bias correction reduces systematic error, it can substantially increase the variance of the estimator. The bias-corrected estimator has lower bias but may have higher mean squared error. In practice, bias correction is most useful when the bias is large relative to the standard error.

## The Bootstrap Distribution

The histogram of $\{\hat{\theta}^{*(1)}, \ldots, \hat{\theta}^{*(B)}\}$ provides a visual approximation of the sampling distribution of $\hat{\theta}$. Key features that can be read from this distribution include:

- **Center**: $\bar{\hat{\theta}}^*$ approximates $E[\hat{\theta}]$
- **Spread**: $\widehat{\text{SE}}_{\text{boot}}$ approximates $\text{SE}(\hat{\theta})$
- **Shape**: skewness, multimodality, or heavy tails of the sampling distribution
- **Quantiles**: used directly for confidence interval construction

The shape information is particularly valuable because it reveals when normal-theory approximations would be inappropriate.

## Properties of Bootstrap Samples

Since each bootstrap sample draws $n$ observations with replacement from $n$ data points, some observations are selected multiple times and others are omitted entirely.

The number of times observation $x_i$ appears in a single bootstrap sample follows a $\text{Binomial}(n, 1/n)$ distribution, which for large $n$ is approximately $\text{Poisson}(1)$. The probability that $x_i$ does not appear at all is:

$$
P(x_i \notin \text{bootstrap sample}) = \left(1 - \frac{1}{n}\right)^n \to e^{-1} \approx 0.368
$$

On average, each bootstrap sample contains approximately $63.2\%$ of the unique original observations.

!!! tip "Out-of-Bag Observations"
    The observations not included in a given bootstrap sample are called **out-of-bag** (OOB) observations. These provide a natural held-out set for estimating prediction error, a technique exploited extensively in random forests and bagging.

## Theoretical Justification

The nonparametric bootstrap is consistent for a broad class of statistics. If $\hat{\theta}_n = T(\hat{F}_n)$ is a smooth functional of the empirical distribution, then under regularity conditions:

$$
\sup_t \left|P^*\!\left(\sqrt{n}(\hat{\theta}^* - \hat{\theta}) \le t\right) - P\!\left(\sqrt{n}(\hat{\theta} - \theta) \le t\right)\right| \xrightarrow{P} 0
$$

where $P^*$ denotes probability under bootstrap resampling. This result follows from the Glivenko-Cantelli theorem (ensuring $\hat{F}_n \to F$ uniformly) combined with the functional delta method (ensuring smooth statistics inherit this convergence).

For the sample mean specifically, the bootstrap achieves the same $O(n^{-1/2})$ rate of approximation as the central limit theorem. For more refined methods such as the bootstrap-$t$, second-order accuracy of $O(n^{-1})$ is attainable.

## Example: Bootstrap Standard Error of the Correlation Coefficient

Suppose we observe $n = 30$ paired observations $(x_i, y_i)$ and compute the sample correlation $r = 0.62$. There is no simple exact formula for $\text{SE}(r)$ that works for non-normal data.

**Bootstrap procedure:**

1. For $b = 1, \ldots, 5000$: resample 30 pairs $(x_i, y_i)$ with replacement, compute $r^{*(b)}$
2. Compute $\widehat{\text{SE}}_{\text{boot}} = \text{sd}(r^{*(1)}, \ldots, r^{*(5000)})$
3. Compute $\widehat{\text{Bias}}_{\text{boot}} = \bar{r}^* - r$

!!! example "Interpreting the Bootstrap Distribution"
    If the histogram of $r^{*(1)}, \ldots, r^{*(5000)}$ is left-skewed (common for $r$ values near 1), then normal-based confidence intervals would be inappropriate. The bootstrap distribution directly reveals this skewness, guiding the choice of confidence interval method (e.g., BCa over the percentile method).

## Choosing the Number of Replicates

The number of bootstrap replicates $B$ controls the **Monte Carlo error** in the bootstrap approximation. Larger $B$ reduces the noise in the bootstrap estimate but increases computation time:

- **Standard errors**: $B = 1{,}000$ is usually sufficient
- **Confidence intervals**: $B = 5{,}000$ to $10{,}000$ for stable quantile estimates
- **Hypothesis testing**: $B = 10{,}000$ or more for precise $p$-values

The Monte Carlo standard error of the bootstrap standard error is approximately $\widehat{\text{SE}}_{\text{boot}} / \sqrt{2B}$, so doubling $B$ reduces the Monte Carlo error by a factor of $\sqrt{2}$.

## Summary

The nonparametric bootstrap replaces the unknown population distribution with the empirical distribution and uses repeated resampling to approximate the sampling distribution of any statistic. It requires no distributional assumptions and provides estimates of standard errors, bias, and distributional shape that would be analytically intractable for most statistics. The method is consistent under regularity conditions and serves as the foundation for the bootstrap confidence intervals and hypothesis tests developed in subsequent sections.
