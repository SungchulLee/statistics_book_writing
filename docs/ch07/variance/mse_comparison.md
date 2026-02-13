# MSE of Variance Estimators

## Overview

We compare the MSE of different variance estimators for Normal data.

## Three Estimators

| Estimator | Formula | Bias | MSE |
|---|---|---|---|
| MLE ($n$) | $\frac{1}{n}\sum(X_i-\bar{X})^2$ | $-\sigma^2/n$ | $\frac{2n-1}{n^2}\sigma^4$ |
| Bessel ($n-1$) | $\frac{1}{n-1}\sum(X_i-\bar{X})^2$ | $0$ | $\frac{2\sigma^4}{n-1}$ |
| MSE-optimal ($n+1$) | $\frac{1}{n+1}\sum(X_i-\bar{X})^2$ | $-\frac{2\sigma^2}{n+1}$ | $\frac{2\sigma^4}{n+1}$ |

## Key Insight

The MSE-optimal estimator (dividing by $n+1$) has the smallest MSE among estimators of the form $c \sum(X_i - \bar{X})^2$, demonstrating the bias-variance tradeoff.
