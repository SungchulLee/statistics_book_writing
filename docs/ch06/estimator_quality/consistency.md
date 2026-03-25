# Consistency and Asymptotic Normality

## Overview

As we collect more data, we expect our estimates to improve. But does a given estimation procedure actually converge to the true parameter value as the sample size grows? Consistency formalizes this guarantee: an estimator that is consistent will eventually concentrate around the truth, no matter how it behaves for small samples.

An estimator $\hat{\theta}_n$ is **consistent** for $\theta$ if it converges in probability to $\theta$:

$$
\hat{\theta}_n \xrightarrow{P} \theta \quad \text{as } n \to \infty
$$

This means that for every $\varepsilon > 0$, $P(|\hat{\theta}_n - \theta| > \varepsilon) \to 0$ as $n \to \infty$.

!!! note "Weak versus strong consistency"

    The definition above is sometimes called **weak consistency**. A stronger notion, **strong consistency**, requires almost sure convergence: $P(\hat{\theta}_n \to \theta) = 1$. Strong consistency implies weak consistency but not vice versa. In practice, many common estimators satisfy both.

## Sufficient Conditions for Consistency

Verifying consistency directly from the definition can be difficult because it requires analyzing the full distribution of $\hat{\theta}_n$ for every $n$. A simpler approach uses the mean squared error. Recall that $\operatorname{Bias}(\hat{\theta}_n) = E[\hat{\theta}_n] - \theta$ and $\operatorname{Var}(\hat{\theta}_n) = E\bigl[(\hat{\theta}_n - E[\hat{\theta}_n])^2\bigr]$.

A **sufficient** (but not necessary) condition for consistency is that both the bias and the variance vanish as $n \to \infty$:

$$
\operatorname{Bias}(\hat{\theta}_n) \to 0 \quad \text{and} \quad \operatorname{Var}(\hat{\theta}_n) \to 0 \quad \text{as } n \to \infty
$$

This works because $\operatorname{MSE}(\hat{\theta}_n) = \operatorname{Bias}^2(\hat{\theta}_n) + \operatorname{Var}(\hat{\theta}_n)$, so both conditions together imply $\operatorname{MSE} \to 0$, which in turn implies convergence in probability.

!!! example "Sample mean is consistent for the population mean"

    Let $X_1, \ldots, X_n$ be i.i.d. with mean $\mu$ and finite variance $\sigma^2$. The sample mean $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ satisfies $\operatorname{Bias}(\bar{X}_n) = 0$ and $\operatorname{Var}(\bar{X}_n) = \sigma^2 / n \to 0$. Both conditions hold, so $\bar{X}_n$ is consistent for $\mu$.

## Asymptotic Normality

Consistency tells us that $\hat{\theta}_n$ converges to $\theta$, but it does not describe how fast the estimator concentrates or what distribution it follows for large $n$. Asymptotic normality answers both questions and provides the foundation for large-sample inference.

An estimator $\hat{\theta}_n$ is **asymptotically normal** if

$$
\sqrt{n}\,(\hat{\theta}_n - \theta) \xrightarrow{d} N(0,\, \sigma^2)
$$

for some $\sigma^2 > 0$ that depends on the estimator and the underlying distribution. The quantity $\sigma^2$ is called the **asymptotic variance**. For many standard estimators, $\sigma^2 = 1/I(\theta)$ where $I(\theta)$ is the Fisher information for a single observation.

This result justifies using normal-based confidence intervals for large samples. Specifically, an approximate $(1 - \alpha)$-level confidence interval takes the form

$$
\hat{\theta}_n \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

where $z_{\alpha/2}$ is the standard normal critical value and $\sigma / \sqrt{n}$ is the asymptotic standard error.
