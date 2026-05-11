# The Bootstrap Principle

## Motivation

Suppose we want to estimate the standard error of a statistic $\hat{\theta}$ computed from a sample $x_1, \ldots, x_n$. For the sample mean of a normal population, the answer is straightforward: $s / \sqrt{n}$. But what if $\hat{\theta}$ is a median, a trimmed mean, a ratio of two estimators, or a complex function of the data? In many practical settings, no closed-form formula for the standard error exists.

Bradley Efron introduced the **bootstrap** in 1979 to address exactly this problem. The core idea is deceptively simple: treat the observed sample as if it were the population, and use resampling to learn about the behavior of $\hat{\theta}$.

## The Plug-In Principle

The bootstrap rests on the **plug-in principle**, a general strategy in statistics. If a quantity of interest depends on the unknown population distribution $F$, we estimate it by substituting the empirical distribution function $\hat{F}_n$ in place of $F$.

The empirical distribution function places equal mass $1/n$ on each observed data point:

$$
\hat{F}_n(x) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(X_i \le x)
$$

For example, the population mean $\mu = \int x \, dF(x)$ is estimated by $\bar{x} = \int x \, d\hat{F}_n(x) = \frac{1}{n}\sum_{i=1}^n x_i$. The population variance $\sigma^2 = \int (x - \mu)^2 \, dF(x)$ is estimated by the sample analogue $\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$. The plug-in principle simply extends this idea to any functional of $F$.

!!! note "Why the Plug-In Principle Works"
    The Glivenko-Cantelli theorem guarantees that $\hat{F}_n \to F$ uniformly almost surely as $n \to \infty$. For any continuous functional $T(F)$, this convergence implies $T(\hat{F}_n) \to T(F)$, so plug-in estimates are consistent under mild regularity conditions.

## From Plug-In to Bootstrap

The plug-in principle tells us how to estimate a parameter. The bootstrap extends this idea to estimate the **sampling distribution** of an estimator.

Consider a statistic $\hat{\theta} = g(X_1, \ldots, X_n)$ computed from an iid sample $X_1, \ldots, X_n \sim F$. Its sampling distribution depends on $F$, which is unknown. The bootstrap replaces $F$ with $\hat{F}_n$ throughout:

| | Population World | Bootstrap World |
|---|---|---|
| **Distribution** | $F$ (unknown) | $\hat{F}_n$ (known) |
| **Sample** | $X_1, \ldots, X_n \overset{\text{iid}}{\sim} F$ | $X_1^*, \ldots, X_n^* \overset{\text{iid}}{\sim} \hat{F}_n$ |
| **Statistic** | $\hat{\theta} = g(X_1, \ldots, X_n)$ | $\hat{\theta}^* = g(X_1^*, \ldots, X_n^*)$ |
| **Target** | Distribution of $\hat{\theta}$ under $F$ | Distribution of $\hat{\theta}^*$ under $\hat{F}_n$ |

Since $\hat{F}_n$ is discrete with support on $\{x_1, \ldots, x_n\}$, drawing from $\hat{F}_n$ is equivalent to **sampling with replacement** from the original data.

## The Key Analogy

Efron's insight can be summarized in one sentence: the bootstrap world is to the observed sample as the real world is to the population.

In the real world, we draw a sample from the population and compute a statistic. We cannot repeat this experiment because we have only one sample. In the bootstrap world, we draw a "sample from the sample" and compute the same statistic. We can repeat this as many times as we like because the "population" (the observed data) is fully known.

The variability of $\hat{\theta}^*$ across bootstrap samples approximates the variability of $\hat{\theta}$ across hypothetical repeated samples from $F$.

## Formal Statement

Let $\hat{\theta}_n = T(\hat{F}_n)$ be a statistical functional evaluated at the empirical distribution. The bootstrap estimate of the distribution of $\hat{\theta}_n - \theta$ is:

$$
\hat{G}_n(t) = P^*\!\left(T(\hat{F}_n^*) - T(\hat{F}_n) \le t\right)
$$

where $P^*$ denotes probability with respect to bootstrap resampling (conditional on the observed data) and $\hat{F}_n^*$ is the empirical distribution of the bootstrap sample.

The bootstrap is **consistent** if $\hat{G}_n$ converges to the true distribution $G$ of $\hat{\theta}_n - \theta$ in probability. Under regularity conditions (smoothness of $T$, finite second moments), this convergence holds:

$$
\sup_t \left| \hat{G}_n(t) - G(t) \right| \xrightarrow{P} 0 \quad \text{as } n \to \infty
$$

!!! tip "Practical Implication"
    Bootstrap consistency means that for large enough $n$ and large enough $B$ (number of resamples), the bootstrap distribution of $\hat{\theta}^* - \hat{\theta}$ faithfully represents the true sampling distribution of $\hat{\theta} - \theta$. This justifies using bootstrap replicates to construct confidence intervals and perform hypothesis tests.

## Example: Standard Error of the Median

Consider a sample of $n = 20$ observations, and suppose we want the standard error of the sample median. No simple formula exists (unlike $s/\sqrt{n}$ for the mean).

**Bootstrap procedure:**

1. Compute the observed median $\hat{\theta} = \text{median}(x_1, \ldots, x_{20})$
2. For $b = 1, \ldots, B$: draw 20 observations with replacement from the data, compute the median $\hat{\theta}^{*(b)}$
3. Estimate the standard error as the standard deviation of the $B$ bootstrap medians:

$$
\widehat{\text{SE}}_{\text{boot}} = \sqrt{\frac{1}{B-1} \sum_{b=1}^{B} \left(\hat{\theta}^{*(b)} - \bar{\hat{\theta}}^*\right)^2}
$$

where $\bar{\hat{\theta}}^* = \frac{1}{B}\sum_{b=1}^{B} \hat{\theta}^{*(b)}$.

This procedure works identically regardless of whether $\hat{\theta}$ is a median, a correlation coefficient, a regression coefficient, or any other statistic.

## Properties of Bootstrap Samples

Each bootstrap sample of size $n$ is drawn with replacement, so some original observations appear multiple times while others may not appear at all. The probability that a particular observation is excluded from one bootstrap sample is:

$$
\left(1 - \frac{1}{n}\right)^n \to e^{-1} \approx 0.368 \quad \text{as } n \to \infty
$$

On average, about 63.2% of the unique original observations appear in each bootstrap sample. The remaining 36.8% are "out-of-bag" observations, which play an important role in certain applications such as random forests.

!!! warning "Bootstrap Does Not Create New Information"
    The bootstrap does not magically generate new data or reduce sampling variability. It estimates how much the statistic would vary across repeated samples from the population, using only the information contained in the single observed sample. If the sample is unrepresentative of the population, the bootstrap will faithfully reproduce that unrepresentativeness.

## Summary

The bootstrap principle converts an intractable analytical problem (finding the sampling distribution of $\hat{\theta}$) into a computational one (resampling from the observed data). By substituting $\hat{F}_n$ for $F$, any quantity that depends on the unknown population distribution can be approximated through simulation. The following sections develop specific bootstrap procedures: the nonparametric bootstrap, the parametric bootstrap, and their applications to confidence intervals and hypothesis testing.

## Exercises

**Exercise 1.**
Explain why bootstrap samples are drawn **with replacement** rather than without replacement. What would happen if we drew samples of size $n$ without replacement from the original data?
