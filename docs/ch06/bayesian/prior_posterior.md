# Prior, Likelihood, and Posterior

## Overview

Bayesian inference combines prior beliefs with observed data through Bayes' theorem:

$$
\pi(\theta \mid \mathbf{x}) = \frac{f(\mathbf{x} \mid \theta) \pi(\theta)}{f(\mathbf{x})} \propto f(\mathbf{x} \mid \theta) \pi(\theta)
$$

## Components

- **Prior** $\pi(\theta)$: encodes beliefs about $\theta$ before seeing data
- **Likelihood** $f(\mathbf{x} \mid \theta)$: probability of the data given the parameter
- **Posterior** $\pi(\theta \mid \mathbf{x})$: updated beliefs after seeing data
- **Marginal likelihood** $f(\mathbf{x})$: normalizing constant

## Bayesian Point Estimates

| Estimate | Definition |
|---|---|
| Posterior mean | $E[\theta \mid \mathbf{x}]$ |
| Posterior median | Median of $\pi(\theta \mid \mathbf{x})$ |
| MAP | Mode of $\pi(\theta \mid \mathbf{x})$ |
