# Consistency and Asymptotic Normality

## Overview

An estimator $\hat{\theta}_n$ is **consistent** for $\theta$ if $\hat{\theta}_n \xrightarrow{P} \theta$ as $n \to \infty$.

## Sufficient Conditions for Consistency

An estimator is consistent if:

$$
\text{Bias}(\hat{\theta}_n) \to 0 \quad \text{and} \quad \text{Var}(\hat{\theta}_n) \to 0 \quad \text{as } n \to \infty
$$

## Asymptotic Normality

An estimator is **asymptotically normal** if:

$$
\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{d} N(0, \sigma^2)
$$

for some $\sigma^2$. This allows construction of approximate confidence intervals and tests for large samples.
