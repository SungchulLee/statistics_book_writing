# MLE for Poisson Distribution

## Motivation

The Poisson distribution is the standard model for count data -- the number of emails received per hour, the number of defects per manufactured unit, or the number of accidents at an intersection per year. Estimating the rate parameter $\lambda$ from observed counts is one of the most common statistical tasks. The Poisson MLE turns out to be the sample mean, and it achieves the best possible precision among all unbiased estimators.

## Setup

Consider a random sample $X_1, X_2, \ldots, X_n$ drawn independently from a Poisson distribution with rate parameter $\lambda > 0$. Each observation takes values in $\{0, 1, 2, \ldots\}$ with probability mass function

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots
$$

The goal is to find the value of $\lambda$ that makes the observed counts most likely.

## Derivation

Since the observations are independent, the joint PMF is the product of the individual mass functions. Taking the logarithm yields the log-likelihood:

$$
\ell(\lambda) = \left(\sum_{i=1}^n x_i\right) \log \lambda - n\lambda - \sum_{i=1}^n \log(x_i!)
$$

The last term does not depend on $\lambda$, so it plays no role in the optimization. To maximize the log-likelihood, we differentiate with respect to $\lambda$ and set the result to zero:

$$
\frac{d\ell}{d\lambda} = \frac{\sum_{i=1}^n x_i}{\lambda} - n = 0
$$

Solving for $\lambda$ gives the maximum likelihood estimator:

$$
\hat{\lambda}_{\text{MLE}} = \frac{\sum_{i=1}^n x_i}{n} = \bar{X}
$$

To confirm that this critical point is a maximum, we check the second derivative:

$$
\frac{d^2\ell}{d\lambda^2} = -\frac{\sum_{i=1}^n x_i}{\lambda^2} \leq 0
$$

This is strictly negative whenever at least one observation is positive, which occurs with probability 1 for $\lambda > 0$. The critical point is therefore a global maximum of the log-likelihood.

## Worked Example

Suppose a quality inspector counts the number of defects on 6 circuit boards and observes: $x_1 = 3$, $x_2 = 1$, $x_3 = 4$, $x_4 = 0$, $x_5 = 2$, $x_6 = 2$. The sample mean is

$$
\bar{x} = \frac{3 + 1 + 4 + 0 + 2 + 2}{6} = 2.0 \text{ defects per board}
$$

The MLE of the rate parameter is therefore $\hat{\lambda} = 2.0$ defects per board.

## Properties

With the MLE established, we now examine its statistical qualities -- whether it is unbiased, how precise it is, and whether any other unbiased estimator could do better.

**Unbiasedness.** Unlike the exponential MLE, the Poisson MLE is unbiased. Since $E[\bar{X}] = E[X_1] = \lambda$, the estimator hits the true parameter on average for any sample size.

**Fisher information.** The per-observation Fisher information is

$$
I_1(\lambda) = -E\!\left[\frac{d^2 \log f(X;\lambda)}{d\lambda^2}\right] = -E\!\left[-\frac{X}{\lambda^2}\right] = \frac{E[X]}{\lambda^2} = \frac{1}{\lambda}
$$

For a sample of size $n$, the total Fisher information is $I_n(\lambda) = n / \lambda$.

**Efficiency.** The Cramer-Rao lower bound (CRLB) for the variance of any unbiased estimator of $\lambda$ is

$$
\text{Var}(\hat{\lambda}) \geq \frac{1}{I_n(\lambda)} = \frac{\lambda}{n}
$$

Since $\text{Var}(\bar{X}) = \text{Var}(X_1)/n = \lambda/n$, the MLE achieves this bound exactly. No unbiased estimator of $\lambda$ can have smaller variance, making $\bar{X}$ the uniformly minimum variance unbiased estimator (UMVUE) for the Poisson rate.
