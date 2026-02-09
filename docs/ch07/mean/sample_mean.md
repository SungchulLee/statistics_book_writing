# Sample Mean as Estimator

## Introduction

The **sample mean** $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ is the most fundamental estimator in all of statistics. It serves as the natural estimator for the population mean $\mu = E[X]$ and plays a central role in estimation theory, hypothesis testing, and nearly every branch of applied statistics and finance. Understanding its properties — why it works, when it works optimally, and when it fails — is essential for statistical practice.

## Definition

Given a random sample $X_1, X_2, \ldots, X_n$ from a population with mean $\mu$ and variance $\sigma^2$, the **sample mean** is:

$$\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$$

The sample mean is a **statistic** (a function of the data) and hence a random variable. Its value changes with each sample drawn.

## Properties

### Expectation

$$E[\bar{X}] = E\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \frac{1}{n} \cdot n\mu = \mu$$

The sample mean is an **unbiased estimator** of $\mu$ regardless of the population distribution, sample size, or whether observations are identically distributed (as long as they share the same mean).

### Variance

For iid observations:

$$\text{Var}(\bar{X}) = \text{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{\sigma^2}{n}$$

**Key implications:**
- Variance decreases linearly in $n$
- Standard error: $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$
- To halve the standard error, quadruple the sample size

### Mean Squared Error

Since $\bar{X}$ is unbiased:

$$\text{MSE}(\bar{X}) = \text{Var}(\bar{X}) = \frac{\sigma^2}{n}$$

### Distribution of the Sample Mean

**For normal populations:** If $X_i \sim N(\mu, \sigma^2)$, then:

$$\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{exactly, for all } n$$

**Central Limit Theorem (any population):** For large $n$:

$$\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1)$$

or equivalently $\bar{X} \approx N(\mu, \sigma^2/n)$ for large $n$.

## Optimality Properties

### MVUE for Normal Populations

For $X_i \sim N(\mu, \sigma^2)$ with known $\sigma^2$, the sample mean $\bar{X}$ is the **Minimum Variance Unbiased Estimator (MVUE)** of $\mu$. It achieves the Cramér-Rao Lower Bound:

$$\text{Var}(\bar{X}) = \frac{\sigma^2}{n} = \frac{1}{nI_1(\mu)}$$

where $I_1(\mu) = 1/\sigma^2$ is the Fisher information per observation.

### Sufficiency

For the normal distribution, $\bar{X}$ is a **sufficient statistic** for $\mu$ (when $\sigma^2$ is known). By the Rao-Blackwell theorem, any unbiased estimator can be improved (in terms of MSE) by conditioning on a sufficient statistic. Since $\bar{X}$ is already based on the sufficient statistic, it cannot be improved.

### Gauss-Markov Theorem

In the linear regression context $Y = X\beta + \epsilon$, the OLS estimator (which includes $\bar{X}$ as a special case when $X = \mathbf{1}$) is the **Best Linear Unbiased Estimator (BLUE)**: it has the smallest variance among all linear unbiased estimators, regardless of the error distribution.

### Efficiency Relative to Other Estimators

For normal populations, compare the sample mean with alternatives:

| Estimator | Variance (Normal) | Relative Efficiency |
|-----------|-------------------|-------------------|
| Sample Mean $\bar{X}$ | $\sigma^2/n$ | 1.000 (reference) |
| Sample Median | $\frac{\pi}{2} \cdot \frac{\sigma^2}{n}$ | $\frac{2}{\pi} \approx 0.637$ |
| Midrange | $\frac{\sigma^2}{2\log n}$ (approx.) | Inconsistent |
| 10% Trimmed Mean | $\approx 1.05 \cdot \frac{\sigma^2}{n}$ | $\approx 0.952$ |

For normal data, the sample mean is strictly the best. For heavy-tailed data, alternatives like the trimmed mean or median can be better.

## Sample Mean for Non-iid Data

### Correlated Observations

If observations are correlated (common in time series), the variance formula changes:

$$\text{Var}(\bar{X}) = \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n \text{Cov}(X_i, X_j)$$

For stationary data with autocorrelation $\rho_k = \text{Corr}(X_t, X_{t+k})$:

$$\text{Var}(\bar{X}) = \frac{\sigma^2}{n}\left(1 + 2\sum_{k=1}^{n-1}\left(1 - \frac{k}{n}\right)\rho_k\right)$$

With positive autocorrelation, $\text{Var}(\bar{X}) > \sigma^2/n$ — the effective sample size is smaller than $n$.

### Weighted Mean

When observations have unequal reliability, use the **weighted mean**:

$$\bar{X}_w = \frac{\sum_{i=1}^n w_i X_i}{\sum_{i=1}^n w_i}$$

If $\text{Var}(X_i) = \sigma_i^2$ and observations are independent, the optimal weights are $w_i = 1/\sigma_i^2$ (inverse variance weighting), yielding:

$$\text{Var}(\bar{X}_w) = \frac{1}{\sum_{i=1}^n 1/\sigma_i^2}$$

## When the Sample Mean Fails

The sample mean is not always the best estimator:

1. **Heavy-tailed distributions** (e.g., Cauchy, Student-t with few df): The sample mean may have infinite variance or not even exist. The median is more robust.

2. **Contaminated data** (outliers): A single extreme observation can dramatically shift $\bar{X}$. Robust alternatives (trimmed mean, Huber estimator) are preferable.

3. **Skewed distributions**: For highly skewed data, the mean may not be the most useful summary. The median or a transformed mean may be better.

4. **Small samples from non-normal populations**: The CLT approximation may be poor, leading to unreliable confidence intervals based on $\bar{X}$.

## Connections to Finance

The sample mean is ubiquitous in finance, but its limitations are especially important:

- **Expected return estimation**: The sample mean of historical returns is the standard estimator for expected returns, but it is notoriously imprecise. With typical annual return volatility of 20% and 50 years of data, $\text{SE}(\bar{X}) \approx 20\%/\sqrt{50} \approx 2.8\%$, which is large relative to typical risk premia.

- **Sharpe ratio**: $\hat{SR} = \bar{X}/S$ inherits the imprecision of $\bar{X}$. The estimation error in the numerator dominates.

- **Time series dependence**: Financial returns exhibit volatility clustering and other forms of dependence, making the standard $\sigma/\sqrt{n}$ formula an underestimate of uncertainty.

- **Portfolio optimization**: The sensitivity of mean-variance optimization to estimated means is well known (estimation error dominates, leading to extreme weights). Shrinkage estimators and Bayesian approaches address this.

## Summary

The sample mean is the simplest and most natural estimator of a population mean. For normal populations, it is optimal by every criterion: unbiased, minimum variance, sufficient, and efficient. For large samples from any distribution with finite variance, it is consistent and approximately normal (by the CLT). However, its sensitivity to outliers and heavy tails, and the slowness of its convergence ($1/\sqrt{n}$), make it important to understand alternatives and to use it wisely, especially in financial applications where estimation precision is at a premium.

## Key Formulas

| Quantity | Formula |
|----------|---------|
| Sample mean | $\bar{X} = \frac{1}{n}\sum X_i$ |
| $E[\bar{X}]$ | $\mu$ (unbiased) |
| $\text{Var}(\bar{X})$ | $\sigma^2/n$ |
| Standard error | $\sigma/\sqrt{n}$ |
| For normal: exact distribution | $\bar{X} \sim N(\mu, \sigma^2/n)$ |
| CLT | $\sqrt{n}(\bar{X} - \mu)/\sigma \to N(0,1)$ |
| Cramér-Rao bound | $\sigma^2/n$ (achieved) |
