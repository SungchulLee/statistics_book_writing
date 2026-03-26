# Efficiency and the Cramer-Rao Lower Bound

## Why a Lower Bound on Variance?

Given a parametric model, many different unbiased estimators of a parameter $\theta$ may exist. Some will have smaller variance than others, and a natural question arises: how small can the variance of an unbiased estimator possibly be? The **Cramer-Rao Lower Bound (CRLB)** answers this question by establishing a fundamental floor on the variance of any unbiased estimator. This result is central to estimation theory because it provides a benchmark against which we can measure the quality of any particular estimator.

## Fisher Information

Before stating the CRLB, we need the concept of Fisher information, which quantifies how much information a random observation carries about the unknown parameter.

Let $X$ be a random variable with probability density or mass function $f(x; \theta)$, where $\theta$ is the parameter of interest. The **score function** is the partial derivative of the log-likelihood with respect to $\theta$:

$$
s(X; \theta) = \frac{\partial}{\partial \theta} \log f(X; \theta)
$$

Under regularity conditions (the support of $f$ does not depend on $\theta$, and differentiation under the integral sign is valid), the score function has mean zero: $E[s(X; \theta)] = 0$.

The **Fisher information** for a single observation is defined as the variance of the score function:

$$
I(\theta) = E\left[\left(\frac{\partial}{\partial \theta} \log f(X;\theta)\right)^2\right]
$$

Under the same regularity conditions that permit interchanging differentiation and integration, this equals the negative expected second derivative of the log-likelihood:

$$
I(\theta) = -E\left[\frac{\partial^2}{\partial\theta^2} \log f(X;\theta)\right]
$$

The second form is often easier to compute. Intuitively, a large Fisher information means the log-likelihood is sharply curved around the true parameter value, making $\theta$ easier to estimate precisely.

For a random sample $X_1, \ldots, X_n \overset{\text{iid}}{\sim} f(x; \theta)$, the total Fisher information is $n I(\theta)$, reflecting the fact that independent observations contribute additively to information.

## The Cramer-Rao Inequality

With Fisher information in hand, we can state the fundamental result.

!!! info "Theorem (Cramer-Rao Lower Bound)"

    Let $X_1, \ldots, X_n$ be iid with density or mass function $f(x; \theta)$ satisfying the regularity conditions:

    1. The support $\{x : f(x; \theta) > 0\}$ does not depend on $\theta$.
    2. The derivatives $\frac{\partial}{\partial \theta} f(x; \theta)$ and $\frac{\partial^2}{\partial \theta^2} f(x; \theta)$ exist and are continuous.
    3. Differentiation and integration (or summation) can be interchanged.

    Then for any unbiased estimator $\hat{\theta} = \hat{\theta}(X_1, \ldots, X_n)$:

    $$
    \text{Var}(\hat{\theta}) \geq \frac{1}{n I(\theta)}
    $$

The quantity $1 / (nI(\theta))$ is the Cramer-Rao lower bound. No unbiased estimator can have variance below this threshold.

!!! example "CRLB for the Normal Mean"

    Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$ with $\sigma^2$ known. The log-likelihood for a single observation is

    $$
    \log f(x; \mu) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}
    $$

    Taking the second derivative with respect to $\mu$:

    $$
    \frac{\partial^2}{\partial \mu^2} \log f(x; \mu) = -\frac{1}{\sigma^2}
    $$

    Therefore $I(\mu) = 1/\sigma^2$, and the CRLB for $n$ observations is

    $$
    \text{Var}(\hat{\mu}) \geq \frac{1}{n \cdot (1/\sigma^2)} = \frac{\sigma^2}{n}
    $$

    Since $\text{Var}(\bar{X}) = \sigma^2 / n$, the sample mean $\bar{X}$ achieves the CRLB exactly.

## Efficiency

The CRLB naturally leads to a way of ranking unbiased estimators. An unbiased estimator that achieves the lower bound is the best possible — it extracts all the information from the data.

An unbiased estimator $\hat{\theta}$ is called **efficient** if

$$
\text{Var}(\hat{\theta}) = \frac{1}{n I(\theta)}
$$

for all $\theta$. The **efficiency** of an unbiased estimator is the ratio of the CRLB to its actual variance:

$$
e(\hat{\theta}) = \frac{1 / (nI(\theta))}{\text{Var}(\hat{\theta})} \leq 1
$$

An efficiency of 1 means the estimator is efficient; values less than 1 indicate how much variance is "wasted" relative to the theoretical optimum.

!!! example "Efficiency of the Sample Mean (Normal)"

    From the example above, $\bar{X}$ has variance $\sigma^2/n$ and the CRLB is $\sigma^2/n$, so

    $$
    e(\bar{X}) = \frac{\sigma^2/n}{\sigma^2/n} = 1
    $$

    The sample mean is an efficient estimator of the normal mean (with known variance).

!!! example "Efficiency of the Sample Median (Normal)"

    For $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$, the sample median is also an unbiased estimator of $\mu$, but its asymptotic variance is $\pi \sigma^2 / (2n)$. Therefore its asymptotic efficiency is

    $$
    e(\text{median}) = \frac{\sigma^2/n}{\pi\sigma^2/(2n)} = \frac{2}{\pi} \approx 0.637
    $$

    The sample median uses only about 64% of the information that the sample mean extracts from normally distributed data.

## When the CRLB Cannot Be Achieved

Not every parametric model admits an efficient estimator. The CRLB is achievable if and only if the score function can be written in the form

$$
\frac{\partial}{\partial \theta} \log f(x; \theta) = a(\theta)\left[T(x) - \theta\right]
$$

for some function $a(\theta)$ and statistic $T(x)$. This condition is satisfied by exponential family distributions but fails for many other models.

!!! warning "Uniform Distribution"

    Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Uniform}(0, \theta)$. The support $\{x : 0 < x < \theta\}$ depends on $\theta$, violating the first regularity condition. The CRLB does not apply, and indeed the MLE $\hat{\theta} = X_{(n)}$ (the sample maximum) has variance that decreases at rate $1/n^2$, faster than any $1/n$ rate that the CRLB would allow.
