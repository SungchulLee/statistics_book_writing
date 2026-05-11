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

## Exercises

**Exercise 1.**
Prove that the sample mean $\bar{X}_n$ is a consistent estimator of $\mu = E[X]$ using Chebyshev's inequality, assuming $\text{Var}(X) = \sigma^2 < \infty$.

??? success "Solution to Exercise 1"
    By Chebyshev's inequality, for any $\varepsilon > 0$:

    $$
    P(|\bar{X}_n - \mu| \geq \varepsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\varepsilon^2} = \frac{\sigma^2}{n\varepsilon^2}
    $$

    As $n \to \infty$:

    $$
    P(|\bar{X}_n - \mu| \geq \varepsilon) \leq \frac{\sigma^2}{n\varepsilon^2} \to 0
    $$

    Therefore $\bar{X}_n \xrightarrow{p} \mu$, which is the definition of consistency. $\square$

---

**Exercise 2.**
The sample median is also a consistent estimator of the population mean for symmetric distributions. Explain intuitively why it converges to $\mu$, and state one advantage of the sample median over the sample mean.

??? success "Solution to Exercise 2"
    For a symmetric distribution, the population mean and median coincide. The sample median converges to the population median by the Glivenko-Cantelli theorem (the empirical CDF converges uniformly to the true CDF, so quantiles converge). Since the population median equals $\mu$ for symmetric distributions, the sample median is consistent for $\mu$.

    **Advantage of the median:** It is robust to outliers. For heavy-tailed distributions (e.g., Cauchy), the sample mean can be highly variable and may not even be consistent (the Cauchy has no finite mean), while the sample median remains consistent and stable. Even for distributions with finite variance, the median has bounded influence function, meaning a single extreme observation cannot drastically change the estimate.

---

**Exercise 3.**
Show that if $\hat{\theta}_n$ is a consistent estimator of $\theta$ and $g$ is a continuous function, then $g(\hat{\theta}_n)$ is a consistent estimator of $g(\theta)$. State the theorem you are using.

??? success "Solution to Exercise 3"
    This is the **Continuous Mapping Theorem**: if $\hat{\theta}_n \xrightarrow{p} \theta$ and $g$ is continuous at $\theta$, then $g(\hat{\theta}_n) \xrightarrow{p} g(\theta)$.

    **Application:** Since $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2 \xrightarrow{p} \sigma^2$ (the sample variance is consistent for the population variance), the continuous function $g(x) = \sqrt{x}$ gives:

    $$
    S = \sqrt{S^2} \xrightarrow{p} \sqrt{\sigma^2} = \sigma
    $$

    So the sample standard deviation is a consistent estimator of $\sigma$. $\square$

---

**Exercise 4.**
Define asymptotic normality of an estimator $\hat{\theta}_n$. If $\hat{\theta}_n$ is the MLE and regularity conditions hold, state the asymptotic distribution of $\sqrt{n}(\hat{\theta}_n - \theta_0)$.

??? success "Solution to Exercise 4"
    An estimator $\hat{\theta}_n$ is **asymptotically normal** if:

    $$
    \sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} N(0, v^2)
    $$

    for some variance $v^2$, where $\xrightarrow{d}$ denotes convergence in distribution.

    For the MLE under standard regularity conditions (the parameter space is open, the model is identifiable, the log-likelihood is twice differentiable, etc.):

    $$
    \sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} N\!\left(0, \frac{1}{I(\theta_0)}\right)
    $$

    where $I(\theta_0) = -E\!\left[\frac{\partial^2}{\partial\theta^2}\log f(X;\theta_0)\right]$ is the Fisher information for a single observation. Equivalently, $\hat{\theta}_{\text{MLE}} \approx N(\theta_0, 1/(nI(\theta_0)))$ for large $n$. This result implies the MLE achieves the Cramer-Rao lower bound asymptotically, making it asymptotically efficient.
