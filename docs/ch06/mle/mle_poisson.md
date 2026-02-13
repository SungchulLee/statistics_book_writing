# MLE for Poisson Distribution

## Overview

We derive the MLE for the Poisson distribution $X \sim \text{Poisson}(\lambda)$ with PMF $P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$.

## Derivation

The log-likelihood is:

$$
\ell(\lambda) = \left(\sum_{i=1}^n x_i\right) \log \lambda - n\lambda - \sum_{i=1}^n \log(x_i!)
$$

Setting the derivative to zero:

$$
\frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n = 0 \implies \hat{\lambda}_{MLE} = \bar{X}
$$

## Properties

- $\hat{\lambda}_{MLE} = \bar{X}$ is unbiased
- Fisher information: $I(\lambda) = 1/\lambda$
- $\hat{\lambda}$ is efficient (achieves CRLB)
