# Introduction to MLE

## Overview

Maximum Likelihood Estimation (MLE) is a method used to estimate the parameters of a statistical model by maximizing the likelihood function. The likelihood function measures how well the model, with certain parameters, explains the observed data. The MLE estimates are the parameters that make the observed data most probable.

In simpler terms, MLE finds the parameter values that make the observed data most "likely." Compared to other methods, such as the Method of Moments, MLE is often preferred because of its desirable properties, including consistency and asymptotic efficiency.

## Mathematical Formulation

The MLE is formally defined as:

$$
\hat{\theta}_{MLE} = \arg \max_{\theta} L(\theta \mid \mathbf{x}) = \arg \max_{\theta} \prod_{i=1}^n f(x_i \mid \theta)
$$

Here, $\theta$ represents the parameters we want to estimate, and $L(\theta \mid \mathbf{x})$ is the likelihood function, which is the product of the probability density function (or probability mass function for discrete data) evaluated at the observed data points $\mathbf{x} = (x_1, x_2, \ldots, x_n)$.

## Log-Likelihood

For ease of computation, we often work with the log-likelihood function:

$$
\log L(\theta \mid \mathbf{x}) = \sum_{i=1}^n \log f(x_i \mid \theta)
$$

This transformation simplifies maximization by turning the product of probabilities into a sum of log-probabilities. Since $\log$ is a monotonically increasing function, maximizing the log-likelihood is equivalent to maximizing the likelihood itself.

## Maximum Likelihood Principle

The key equivalences in the MLE framework:

$$
\text{argmax}_{\theta}\; L
\quad\Leftrightarrow\quad
\text{argmax}_{\theta}\; \ell
\quad\Leftrightarrow\quad
\text{argmin}_{\theta}\; J
$$

where $L$ is the likelihood, $\ell = \log L$ is the log-likelihood, and $J = -\ell$ is the cost function (negative log-likelihood).

## Properties of MLEs

| Property | Description |
|----------|-------------|
| **Consistency** | $\hat{\theta}_{MLE} \xrightarrow{P} \theta_0$ as $n \to \infty$ |
| **Asymptotic normality** | $\sqrt{n}(\hat{\theta}_{MLE} - \theta_0) \xrightarrow{d} N(0, I(\theta_0)^{-1})$ |
| **Asymptotic efficiency** | Achieves the Cramér–Rao lower bound asymptotically |
| **Invariance** | If $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$ |

## Summary

MLE provides a principled, general-purpose approach to parameter estimation. It connects naturally to:

- **Cost functions** in machine learning (cross-entropy loss = negative log-likelihood for classification)
- **Bayesian inference** (MLE is the MAP estimate with a flat prior)
- **Information theory** (minimizing KL divergence between the model and data)
