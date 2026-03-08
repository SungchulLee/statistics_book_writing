# Generalized Method of Moments (GMM)

## Overview

The **Generalized Method of Moments** extends MoM to handle overidentified models where more moment conditions than parameters are available.

## Setup

Given $r > p$ moment conditions $E[g(X, \theta)] = 0$, the GMM estimator minimizes:

$$
\hat{\theta}_{GMM} = \arg\min_{\theta} \left[\frac{1}{n} \sum_{i=1}^n g(x_i, \theta)\right]^T W \left[\frac{1}{n} \sum_{i=1}^n g(x_i, \theta)\right]
$$

where $W$ is a positive-definite weighting matrix.

## Optimal Weighting

The optimal $W$ is the inverse of the asymptotic variance of the moment conditions, yielding the most efficient GMM estimator.
