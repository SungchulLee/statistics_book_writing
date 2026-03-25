# MLE for Exponential Distribution

## Motivation

The exponential distribution models the waiting time between successive events in a Poisson process -- for instance, the time between customer arrivals, between component failures, or between radioactive decays. Estimating the rate parameter $\lambda$ from observed waiting times is a fundamental applied problem. The exponential case yields a clean closed-form MLE that also illustrates how nonlinear transformations can introduce bias into an estimator.

## Setup

Consider a random sample $X_1, X_2, \ldots, X_n$ drawn independently from an exponential distribution with rate parameter $\lambda > 0$. Each observation has density

$$
f(x; \lambda) = \lambda e^{-\lambda x}, \quad x > 0
$$

The goal is to find the value of $\lambda$ that makes the observed data most likely.

## Derivation

Since the observations are independent, the joint density is the product of the individual densities. Taking the logarithm converts this product into a sum, yielding the log-likelihood function:

$$
\ell(\lambda) = \sum_{i=1}^n \log f(x_i; \lambda) = n \log \lambda - \lambda \sum_{i=1}^n x_i
$$

To find the value of $\lambda$ that maximizes the log-likelihood, we differentiate with respect to $\lambda$ and set the result to zero:

$$
\frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0
$$

Solving for $\lambda$ gives the maximum likelihood estimator:

$$
\hat{\lambda}_{\text{MLE}} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{X}}
$$

To confirm that this critical point is indeed a maximum, we check the second derivative:

$$
\frac{d^2\ell}{d\lambda^2} = -\frac{n}{\lambda^2} < 0 \quad \text{for all } \lambda > 0
$$

Since the second derivative is strictly negative everywhere in the parameter space, the critical point $\hat{\lambda}_{\text{MLE}} = 1/\bar{X}$ is a global maximum of the log-likelihood.

## Worked Example

Suppose we observe five waiting times (in minutes) between customer arrivals: $x_1 = 2.1$, $x_2 = 0.8$, $x_3 = 1.5$, $x_4 = 3.2$, $x_5 = 1.4$. The sample mean is

$$
\bar{x} = \frac{2.1 + 0.8 + 1.5 + 3.2 + 1.4}{5} = 1.8 \text{ minutes}
$$

The MLE of the rate parameter is therefore

$$
\hat{\lambda} = \frac{1}{\bar{x}} = \frac{1}{1.8} \approx 0.556 \text{ arrivals per minute}
$$

## Properties

With the MLE in hand, we now examine its statistical properties -- whether it hits the true parameter on average and how precisely it estimates $\lambda$ as the sample size grows.

**Bias.** The MLE $\hat{\lambda} = 1/\bar{X}$ is biased upward. Since the function $g(x) = 1/x$ is convex on $(0, \infty)$, Jensen's inequality gives

$$
E\!\left[\frac{1}{\bar{X}}\right] > \frac{1}{E[\bar{X}]} = \frac{1}{1/\lambda} = \lambda
$$

so the MLE overestimates $\lambda$ on average. However, this bias vanishes as $n \to \infty$, and the estimator is consistent.

**Fisher information.** The per-observation Fisher information is

$$
I_1(\lambda) = -E\!\left[\frac{d^2 \log f(X;\lambda)}{d\lambda^2}\right] = -E\!\left[-\frac{1}{\lambda^2}\right] = \frac{1}{\lambda^2}
$$

For a sample of size $n$, the total Fisher information is $I_n(\lambda) = n / \lambda^2$.

**Asymptotic variance.** By the general asymptotic theory of MLEs, the variance of $\hat{\lambda}$ is approximately

$$
\text{Var}(\hat{\lambda}) \approx \frac{1}{I_n(\lambda)} = \frac{\lambda^2}{n}
$$

This means the estimator becomes more precise as the sample size increases, with standard error proportional to $\lambda / \sqrt{n}$.
