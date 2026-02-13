# MLE for Exponential Distribution

## Overview

We derive the MLE for the Exponential distribution $X \sim \text{Exp}(\lambda)$ with density $f(x; \lambda) = \lambda e^{-\lambda x}$ for $x > 0$.

## Derivation

The log-likelihood is:

$$
\ell(\lambda) = n \log \lambda - \lambda \sum_{i=1}^n x_i
$$

Setting the score to zero:

$$
\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0 \implies \hat{\lambda}_{MLE} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{X}}
$$

## Properties

- $\hat{\lambda}_{MLE} = 1/\bar{X}$ is biased but consistent
- Fisher information: $I(\lambda) = 1/\lambda^2$
- Asymptotic variance: $\text{Var}(\hat{\lambda}) \approx \lambda^2/n$
