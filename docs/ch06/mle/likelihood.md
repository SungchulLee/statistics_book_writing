# Likelihood Function

## Overview

The **likelihood function** is the joint density of the observed data, viewed as a function of the parameters.

## Definition

Given i.i.d. observations $\mathbf{x} = (x_1, \ldots, x_n)$:

$$
L(\theta \mid \mathbf{x}) = \prod_{i=1}^n f(x_i \mid \theta)
$$

## Log-Likelihood

$$
\ell(\theta) = \log L(\theta \mid \mathbf{x}) = \sum_{i=1}^n \log f(x_i \mid \theta)
$$

## Key Properties

- The likelihood is NOT a probability distribution over $\theta$
- It measures the plausibility of parameter values given observed data
- The log-likelihood is computationally more convenient
- Maximizing $L$ is equivalent to maximizing $\ell$
