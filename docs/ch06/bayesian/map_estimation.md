# MAP Estimation

## Overview

**Maximum A Posteriori (MAP)** estimation finds the mode of the posterior distribution:

$$
\hat{\theta}_{MAP} = \arg\max_{\theta} \pi(\theta \mid \mathbf{x}) = \arg\max_{\theta} [\log f(\mathbf{x} \mid \theta) + \log \pi(\theta)]
$$

## Relationship to MLE

MAP estimation differs from MLE by adding a log-prior term. As $n \to \infty$, MAP and MLE converge because the likelihood dominates the prior.

## Relationship to Regularization

- Gaussian prior $\to$ L2 (Ridge) regularization
- Laplace prior $\to$ L1 (Lasso) regularization
