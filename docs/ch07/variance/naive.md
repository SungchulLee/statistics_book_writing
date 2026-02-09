# Naive Variance Estimator

## Introduction

The **naive variance estimator** divides the sum of squared deviations by $n$ (the sample size) rather than by $n-1$. While this is the most intuitive approach — simply averaging the squared deviations from the sample mean — it turns out to be biased. Understanding *why* it is biased provides deep insight into the nature of estimation and motivates Bessel's correction.

## Definition

Given a random sample $X_1, X_2, \ldots, X_n$ with sample mean $\bar{X}$, the **naive variance estimator** is:

$$\tilde{S}^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2$$

This is also called the **population variance formula applied to the sample**, the **biased sample variance**, or the **MLE of variance** (for normal populations).

## Bias Derivation

### The Key Identity

The fundamental identity underlying the bias calculation is:

$$\sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n (X_i - \mu)^2 - n(\bar{X} - \mu)^2$$

**Proof:** Expand $(X_i - \bar{X})^2 = (X_i - \mu - (\bar{X} - \mu))^2$:

$$\sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n(X_i - \mu)^2 - 2(\bar{X} - \mu)\sum_{i=1}^n(X_i - \mu) + n(\bar{X} - \mu)^2$$

Since $\sum(X_i - \mu) = n(\bar{X} - \mu)$, the middle term is $-2n(\bar{X} - \mu)^2$:

$$= \sum_{i=1}^n(X_i - \mu)^2 - n(\bar{X} - \mu)^2$$

### Computing the Expectation

Taking expectations:

$$E\left[\sum_{i=1}^n (X_i - \bar{X})^2\right] = \sum_{i=1}^n E[(X_i - \mu)^2] - nE[(\bar{X} - \mu)^2]$$

$$= n\sigma^2 - n \cdot \frac{\sigma^2}{n} = n\sigma^2 - \sigma^2 = (n-1)\sigma^2$$

Therefore:

$$E[\tilde{S}^2] = E\left[\frac{1}{n}\sum_{i=1}^n(X_i - \bar{X})^2\right] = \frac{n-1}{n}\sigma^2$$

### The Bias

$$\text{Bias}(\tilde{S}^2) = E[\tilde{S}^2] - \sigma^2 = \frac{n-1}{n}\sigma^2 - \sigma^2 = -\frac{\sigma^2}{n}$$

The naive estimator **underestimates** the true variance by a factor of $(n-1)/n$.

## Intuition: Why the Bias Exists

The bias arises because we use $\bar{X}$ instead of $\mu$ in the sum of squares. Since $\bar{X}$ is the value that minimizes $\sum(X_i - c)^2$ over all constants $c$, we have:

$$\sum_{i=1}^n (X_i - \bar{X})^2 \leq \sum_{i=1}^n (X_i - \mu)^2$$

The sum of squared deviations from $\bar{X}$ is **always** less than or equal to the sum from the true mean $\mu$. By using $\bar{X}$, we systematically undercount the variability, leading to downward bias.

Another way to see it: computing $\bar{X}$ "uses up" one piece of information from the data. The $n$ deviations $(X_i - \bar{X})$ satisfy $\sum(X_i - \bar{X}) = 0$, so only $n-1$ of them are free to vary. There are only $n-1$ **degrees of freedom**, not $n$.

## Properties

### Variance of $\tilde{S}^2$

For normal populations:

$$\text{Var}(\tilde{S}^2) = \frac{2(n-1)}{n^2}\sigma^4$$

### MSE of $\tilde{S}^2$

$$\text{MSE}(\tilde{S}^2) = \text{Var}(\tilde{S}^2) + [\text{Bias}(\tilde{S}^2)]^2 = \frac{2(n-1)}{n^2}\sigma^4 + \frac{\sigma^4}{n^2} = \frac{2n-1}{n^2}\sigma^4$$

### Consistency

Despite being biased, $\tilde{S}^2$ is **consistent**:

$$\tilde{S}^2 = \frac{n-1}{n} S^2 \xrightarrow{p} \sigma^2$$

since $(n-1)/n \to 1$ and $S^2 \xrightarrow{p} \sigma^2$.

### Asymptotic Equivalence

For large $n$, $\tilde{S}^2$ and $S^2$ are practically identical:

$$\tilde{S}^2 = \frac{n-1}{n}S^2 \approx S^2 \quad \text{for large } n$$

The bias $-\sigma^2/n \to 0$, and the ratio $(n-1)/n \to 1$.

## Comparison: Divide by $n$ vs $n-1$ vs $n+1$

| Estimator | Divisor | Bias | MSE (Normal) | Notes |
|-----------|---------|------|--------------|-------|
| $\tilde{S}^2$ | $n$ | $-\sigma^2/n$ | $\frac{2n-1}{n^2}\sigma^4$ | MLE; biased |
| $S^2$ | $n-1$ | $0$ | $\frac{2}{n-1}\sigma^4$ | Unbiased (Bessel's) |
| $\hat{S}^2$ | $n+1$ | $-\frac{2}{n+1}\sigma^2$ | $\frac{2(n-1)}{(n+1)^2}\sigma^4 + \frac{4}{(n+1)^2}\sigma^4$ | MSE-optimal (Normal) |

**Surprising fact:** $\text{MSE}(\tilde{S}^2) < \text{MSE}(S^2)$ for all $n$. The biased estimator has lower MSE than the unbiased one! This is a textbook example of the bias-variance tradeoff.

## When $\mu$ is Known

If the true mean $\mu$ is known (rare in practice), we can use:

$$\hat{\sigma}^2_\mu = \frac{1}{n}\sum_{i=1}^n (X_i - \mu)^2$$

This estimator is unbiased: $E[\hat{\sigma}^2_\mu] = \sigma^2$, and it has lower variance than $S^2$:

$$\text{Var}(\hat{\sigma}^2_\mu) = \frac{2\sigma^4}{n} < \frac{2\sigma^4}{n-1} = \text{Var}(S^2)$$

## Connection to MLE

For normal populations, $\tilde{S}^2$ is the MLE of $\sigma^2$. The MLE is biased in finite samples but asymptotically unbiased. This is a common pattern: MLEs are often biased for finite samples but consistent.

## Connections to Finance

- **Volatility estimation**: The realized variance of daily returns uses the formula $\frac{1}{n}\sum r_i^2$ (with $\mu \approx 0$), which is the naive estimator when the mean is set to zero.
- **Risk metrics**: When computing portfolio variance for risk management with large samples ($n > 250$ daily observations), the difference between dividing by $n$ and $n-1$ is negligible.
- **Bias correction**: For small samples (e.g., monthly data over a few years), the bias can be material and Bessel's correction should be used.

## Summary

The naive variance estimator $\tilde{S}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$ is biased downward by $\sigma^2/n$ because using the sample mean instead of the true mean systematically underestimates variability. Despite this bias, it has lower MSE than the unbiased $S^2$ and is consistent. It is also the MLE for normal populations. For large samples, the bias is negligible, but for small samples, Bessel's correction (dividing by $n-1$) is standard.

## Key Formulas

| Quantity | Formula |
|----------|---------|
| Naive estimator | $\tilde{S}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$ |
| Expectation | $E[\tilde{S}^2] = \frac{n-1}{n}\sigma^2$ |
| Bias | $-\sigma^2/n$ |
| MSE (Normal) | $\frac{2n-1}{n^2}\sigma^4$ |
| Key identity | $\sum(X_i - \bar{X})^2 = \sum(X_i - \mu)^2 - n(\bar{X}-\mu)^2$ |
