# Bias of Gaussian MLE for σ²

## Overview

The MLE estimator for the variance of a Normal distribution is biased.

## MLE Estimator

$$
\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

## Bias Calculation

$$
E[\hat{\sigma}^2_{MLE}] = \frac{n-1}{n}\sigma^2 \neq \sigma^2
$$

The bias is $-\sigma^2/n$, which vanishes as $n \to \infty$ (consistency).

## Bessel's Correction

Dividing by $n-1$ instead of $n$ gives the unbiased estimator $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$.

## MSE Comparison

Despite being biased, the MLE has lower MSE than $S^2$ when minimizing MSE is the goal.
