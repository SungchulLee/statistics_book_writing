# Fisher Information and Standard Errors

## Overview

The **Fisher information** measures the amount of information a random variable carries about a parameter.

## Definition

$$
I(\theta) = E\left[\left(\frac{\partial}{\partial\theta} \log f(X;\theta)\right)^2\right] = -E\left[\frac{\partial^2}{\partial\theta^2} \log f(X;\theta)\right]
$$

For $n$ i.i.d. observations: $I_n(\theta) = n \cdot I(\theta)$.

## Standard Errors from Fisher Information

The asymptotic standard error of the MLE is:

$$
\text{SE}(\hat{\theta}_{MLE}) \approx \frac{1}{\sqrt{I_n(\hat{\theta})}}
$$

## Examples

| Distribution | Parameter | Fisher Information |
|---|---|---|
| Bernoulli($p$) | $p$ | $\frac{1}{p(1-p)}$ |
| Normal($\mu, \sigma^2$) | $\mu$ | $\frac{1}{\sigma^2}$ |
| Poisson($\lambda$) | $\lambda$ | $\frac{1}{\lambda}$ |
