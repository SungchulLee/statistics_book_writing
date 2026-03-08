# Efficiency and Cramér–Rao Lower Bound

## Overview

The **Cramér–Rao Lower Bound (CRLB)** provides a lower bound on the variance of any unbiased estimator.

## Cramér–Rao Inequality

For an unbiased estimator $\hat{\theta}$:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

where $I(\theta)$ is the **Fisher information**:

$$
I(\theta) = E\left[\left(\frac{\partial}{\partial\theta} \log f(X;\theta)\right)^2\right] = -E\left[\frac{\partial^2}{\partial\theta^2} \log f(X;\theta)\right]
$$

## Efficiency

An unbiased estimator is **efficient** if it achieves the CRLB, meaning $\text{Var}(\hat{\theta}) = 1/I(\theta)$.
