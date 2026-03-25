# Sufficiency and Minimal Sufficiency

## Overview

In estimation theory, we routinely compress data into summary statistics such as the sample mean or sample variance. A fundamental question arises: does a given summary preserve all the information the original data contained about the unknown parameter? If so, we lose nothing by working with the summary instead of the full data set. Sufficiency formalizes this idea of lossless data reduction.

A statistic $T(\mathbf{X})$ is **sufficient** for a parameter $\theta$ if the conditional distribution of $\mathbf{X}$ given $T(\mathbf{X}) = t$ does not depend on $\theta$, for any value of $t$ in the range of $T$. Intuitively, once we know the value of a sufficient statistic, the remaining randomness in the data carries no additional information about $\theta$.

## Fisher--Neyman Factorization Theorem

Checking sufficiency directly from the definition requires computing conditional distributions, which can be algebraically demanding. The Fisher--Neyman Factorization Theorem provides a much simpler criterion: we only need to factor the joint density into two pieces.

**Theorem (Fisher--Neyman Factorization).** A statistic $T(\mathbf{X})$ is sufficient for $\theta$ if and only if the joint density (or mass function) can be written as

$$
f(\mathbf{x}; \theta) = g\bigl(T(\mathbf{x}),\, \theta\bigr) \cdot h(\mathbf{x})
$$

for all $\mathbf{x}$ in the sample space and all $\theta$ in the parameter space, where $g \geq 0$ is a function that depends on the data only through $T(\mathbf{x})$ and may depend on $\theta$, and $h \geq 0$ is a function of $\mathbf{x}$ alone that does not depend on $\theta$.

!!! example "Factorization for a Poisson sample"

    Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Poisson}(\lambda)$. The joint mass function is

    $$
    f(\mathbf{x}; \lambda) = \prod_{i=1}^n \frac{e^{-\lambda} \lambda^{x_i}}{x_i!} = e^{-n\lambda}\, \lambda^{\sum_{i=1}^n x_i} \cdot \frac{1}{\prod_{i=1}^n x_i!}
    $$

    Set $g\bigl(\sum x_i,\, \lambda\bigr) = e^{-n\lambda}\, \lambda^{\sum x_i}$ and $h(\mathbf{x}) = 1 / \prod x_i!$. The factorization theorem confirms that $T(\mathbf{X}) = \sum_{i=1}^n X_i$ is sufficient for $\lambda$.

## Minimal Sufficiency

A sufficient statistic need not be unique. In fact, the entire data vector $\mathbf{X}$ is always trivially sufficient, since we can write $f(\mathbf{x};\theta) = f(\mathbf{x};\theta) \cdot 1$ with $T(\mathbf{X}) = \mathbf{X}$. However, this extreme case provides no data reduction at all. We therefore seek the most compressed sufficient statistic — one that discards as much redundant information as possible while still retaining everything about $\theta$.

A sufficient statistic $T$ is **minimal sufficient** if, for every other sufficient statistic $T'$, there exists a function $g$ such that $T = g(T')$ almost surely. In other words, $T$ provides the maximum possible data compression while preserving all information about $\theta$.

## Rao--Blackwell Theorem

Sufficiency is not merely a theoretical concept — it has a direct impact on the quality of estimators. Because a sufficient statistic captures all the information about $\theta$, conditioning an estimator on it should not discard useful information and can in fact reduce variability. The Rao--Blackwell Theorem makes this precise.

**Theorem (Rao--Blackwell).** If $\hat{\theta}$ is an unbiased estimator of $\theta$ and $T$ is a sufficient statistic for $\theta$, then $\tilde{\theta} = E[\hat{\theta} \mid T]$ satisfies:

1. $\tilde{\theta}$ is also unbiased for $\theta$, and
2. $\operatorname{Var}(\tilde{\theta}) \leq \operatorname{Var}(\hat{\theta})$, with equality if and only if $\hat{\theta}$ is already a function of $T$ almost surely.

The Rao--Blackwell Theorem shows that we can always improve (or at worst preserve) an unbiased estimator by conditioning on a sufficient statistic. When combined with a minimal sufficient statistic, this procedure extracts the maximum possible variance reduction from the data.
