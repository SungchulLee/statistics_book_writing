# Sufficiency and Completeness

## Overview

For the Normal model, we identify sufficient and complete statistics.

## Sufficient Statistics for Normal

By the factorization theorem, $(\bar{X}, S^2)$ is jointly sufficient for $(\mu, \sigma^2)$ in the Normal model.

## Completeness

A sufficient statistic $T$ is **complete** if $E[g(T)] = 0$ for all $\theta$ implies $g(T) = 0$ a.s. Completeness ensures the UMVUE is unique.

## Lehmann–Scheffé Theorem

If $T$ is complete and sufficient, then any unbiased function of $T$ is the unique UMVUE. For the Normal model:

- $\bar{X}$ is the UMVUE of $\mu$
- $S^2$ is the UMVUE of $\sigma^2$
