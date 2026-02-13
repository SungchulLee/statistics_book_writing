# Conjugate Priors

## Overview

A prior distribution is **conjugate** for a given likelihood if the posterior distribution belongs to the same family as the prior. This property greatly simplifies Bayesian computation.

## Definition

A family $\mathcal{F}$ of prior distributions is conjugate for a likelihood $f(x \mid \theta)$ if for every prior $\pi(\theta) \in \mathcal{F}$, the posterior $\pi(\theta \mid x) \in \mathcal{F}$.

## Common Conjugate Pairs

| Likelihood | Conjugate Prior | Posterior |
|---|---|---|
| Bernoulli/Binomial | Beta($\alpha, \beta$) | Beta($\alpha + k, \beta + n - k$) |
| Poisson | Gamma($\alpha, \beta$) | Gamma($\alpha + \sum x_i, \beta + n$) |
| Normal (known $\sigma^2$) | Normal($\mu_0, \sigma_0^2$) | Normal(weighted mean, updated variance) |
| Exponential | Gamma($\alpha, \beta$) | Gamma($\alpha + n, \beta + \sum x_i$) |
