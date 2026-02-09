# Bessel's Correction

## Introduction

**Bessel's correction** refers to the use of $n-1$ instead of $n$ in the denominator of the sample variance formula, yielding an unbiased estimator of the population variance. Named after Friedrich Bessel, this correction accounts for the fact that estimating the mean from the same data "uses up" one degree of freedom, causing the naive estimator (dividing by $n$) to systematically underestimate the true variance.

## Definition

The **Bessel-corrected sample variance** is:

$$S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$$

This is the standard "sample variance" used in most statistical software (e.g., `numpy.var(ddof=1)`, R's `var()`).

## Unbiasedness Proof

### Main Result

$$E[S^2] = \sigma^2$$

### Proof

Starting from the key identity:

$$\sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n (X_i - \mu)^2 - n(\bar{X} - \mu)^2$$

Taking expectations:

$$E\left[\sum_{i=1}^n (X_i - \bar{X})^2\right] = n\sigma^2 - n \cdot \frac{\sigma^2}{n} = (n-1)\sigma^2$$

Therefore:

$$E\left[\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2\right] = \frac{(n-1)\sigma^2}{n-1} = \sigma^2$$

### Why $n-1$? The Degrees of Freedom Argument

The $n$ deviations $d_i = X_i - \bar{X}$ are subject to the constraint:

$$\sum_{i=1}^n d_i = \sum_{i=1}^n (X_i - \bar{X}) = 0$$

This means only $n-1$ of the deviations are free to vary independently. We say there are $n-1$ **degrees of freedom**. Dividing by the degrees of freedom ($n-1$) instead of the number of observations ($n$) corrects the bias.

**General principle:** When estimating a variance using $k$ estimated parameters, divide by $n - k$:
- Mean unknown, variance of $X$: divide by $n - 1$
- Regression with $p$ coefficients: residual variance uses $n - p$

## Distribution of $S^2$

### For Normal Populations

If $X_i \sim N(\mu, \sigma^2)$, then:

$$\frac{(n-1)S^2}{\sigma^2} = \frac{\sum_{i=1}^n (X_i - \bar{X})^2}{\sigma^2} \sim \chi^2_{n-1}$$

This exact distributional result gives us:

$$E[S^2] = \sigma^2 \cdot \frac{n-1}{n-1} = \sigma^2 \quad \text{(confirming unbiasedness)}$$

$$\text{Var}(S^2) = \frac{2\sigma^4}{n-1}$$

### Independence of $\bar{X}$ and $S^2$

For normal populations, $\bar{X}$ and $S^2$ are **independent**. This is a remarkable property unique to the normal distribution (by Cochran's theorem) and is crucial for the derivation of the $t$-distribution used in hypothesis testing.

## Properties

### Variance and MSE

For normal populations:

$$\text{Var}(S^2) = \frac{2\sigma^4}{n-1}$$

$$\text{MSE}(S^2) = \text{Var}(S^2) = \frac{2\sigma^4}{n-1} \quad \text{(since bias = 0)}$$

### Consistency

$S^2$ is consistent for $\sigma^2$:

$$S^2 \xrightarrow{p} \sigma^2 \quad \text{as } n \to \infty$$

### MSE Comparison with Alternatives

For normal populations:

$$\text{MSE}(S^2) = \frac{2}{n-1}\sigma^4 > \frac{2n-1}{n^2}\sigma^4 = \text{MSE}(\tilde{S}^2)$$

The unbiased estimator has **higher MSE** than the biased naive estimator. This is because unbiasedness comes at the cost of increased variance, and the variance increase outweighs the bias reduction (in MSE terms).

## Practical Significance

### When Does It Matter?

The difference between dividing by $n$ and $n-1$:

| $n$ | $(n-1)/n$ | Relative error |
|-----|-----------|---------------|
| 3 | 0.667 | 33.3% |
| 5 | 0.800 | 20.0% |
| 10 | 0.900 | 10.0% |
| 30 | 0.967 | 3.3% |
| 100 | 0.990 | 1.0% |
| 1000 | 0.999 | 0.1% |

For $n > 30$, the practical difference is small. For $n < 10$, the correction is substantial.

### Standard Deviation Bias

While $S^2$ is unbiased for $\sigma^2$, the sample standard deviation $S = \sqrt{S^2}$ is **not** unbiased for $\sigma$. By Jensen's inequality (since $\sqrt{\cdot}$ is concave):

$$E[S] = E[\sqrt{S^2}] < \sqrt{E[S^2]} = \sigma$$

For normal populations:

$$E[S] = \sigma \cdot \sqrt{\frac{2}{n-1}} \cdot \frac{\Gamma(n/2)}{\Gamma((n-1)/2)}$$

The correction factor $c_4 = \sqrt{2/(n-1)} \cdot \Gamma(n/2)/\Gamma((n-1)/2)$ can be used to obtain an unbiased estimator of $\sigma$: $\hat{\sigma} = S/c_4$.

## Software Implementation

Different software has different defaults:

| Software | `var()` default | Divisor |
|----------|----------------|---------|
| Python `numpy.var()` | $n$ (population) | `ddof=0` |
| Python `numpy.var(ddof=1)` | $n-1$ (sample) | `ddof=1` |
| R `var()` | $n-1$ (sample) | — |
| Excel `VAR.S()` | $n-1$ (sample) | — |
| Excel `VAR.P()` | $n$ (population) | — |
| pandas `.var()` | $n-1$ (sample) | `ddof=1` |

**Common pitfall:** Using `numpy.var()` without `ddof=1` gives the biased (naive) estimator. Always specify `ddof=1` when you want the unbiased sample variance.

## Generalization: Degrees of Freedom in Regression

In linear regression $Y = X\beta + \epsilon$, the residual variance estimator is:

$$\hat{\sigma}^2 = \frac{1}{n-p}\sum_{i=1}^n (Y_i - \hat{Y}_i)^2 = \frac{\text{RSS}}{n-p}$$

where $p$ is the number of estimated coefficients. This generalizes Bessel's correction: we lose one degree of freedom for each estimated parameter.

## Connections to Finance

- **Realized volatility**: When computing daily realized volatility from intraday returns, the choice of $n$ vs $n-1$ is often irrelevant (many observations). But for monthly volatility from daily data (~21 observations), the correction matters.

- **Tracking error**: Computing tracking error of a portfolio vs. benchmark uses $S = \sqrt{\frac{1}{n-1}\sum(r_p - r_b)^2}$ with Bessel's correction.

- **Risk budgeting**: Variance decomposition in portfolio risk uses the unbiased covariance matrix, which divides by $n-1$.

## Summary

Bessel's correction ($n-1$ in the denominator) produces an unbiased estimator of the population variance. The correction compensates for the "lost" degree of freedom from estimating the mean. While the unbiased estimator has higher MSE than the naive one, unbiasedness is often preferred for its theoretical properties and is the standard in most statistical software. For large samples, the choice between $n$ and $n-1$ is inconsequential.

## Key Formulas

| Quantity | Formula |
|----------|---------|
| Bessel-corrected variance | $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$ |
| Unbiasedness | $E[S^2] = \sigma^2$ |
| Distribution (Normal) | $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$ |
| Variance (Normal) | $\text{Var}(S^2) = 2\sigma^4/(n-1)$ |
| $S$ is biased for $\sigma$ | $E[S] < \sigma$ (Jensen's inequality) |
| General regression | $\hat{\sigma}^2 = \text{RSS}/(n-p)$ |
