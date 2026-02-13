# Sufficiency and Minimal Sufficiency

## Overview

A statistic $T(\mathbf{X})$ is **sufficient** for $\theta$ if the conditional distribution of $\mathbf{X}$ given $T$ does not depend on $\theta$.

## Fisher–Neyman Factorization Theorem

$T(\mathbf{X})$ is sufficient for $\theta$ if and only if:

$$
f(\mathbf{x}; \theta) = g(T(\mathbf{x}), \theta) \cdot h(\mathbf{x})
$$

## Minimal Sufficiency

A sufficient statistic is **minimal sufficient** if it is a function of every other sufficient statistic. It achieves the greatest data reduction without losing information about $\theta$.

## Rao–Blackwell Theorem

If $\hat{\theta}$ is an unbiased estimator and $T$ is sufficient, then $\tilde{\theta} = E[\hat{\theta} \mid T]$ is at least as good (in terms of MSE) as $\hat{\theta}$.
