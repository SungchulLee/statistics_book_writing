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

## Exercises

**Exercise 1.**
Derive the MLE of $\lambda$ for an i.i.d. sample $x_1, \dots, x_n$ from $\text{Exp}(\lambda)$ (where the density is $f(x;\lambda) = \lambda e^{-\lambda x}$ for $x > 0$).

??? success "Solution to Exercise 1"
    The log-likelihood is:

    $$
    \ell(\lambda) = \sum_{i=1}^n \bigl(\log\lambda - \lambda x_i\bigr) = n\log\lambda - \lambda\sum_{i=1}^n x_i
    $$

    Setting the derivative to zero:

    $$
    \frac{d\ell}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0 \implies \hat{\lambda} = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{X}}
    $$

    The second derivative is $-n/\lambda^2 < 0$, confirming this is a maximum.

---

**Exercise 2.**
A call center receives calls with exponentially distributed inter-arrival times. In a sample of 50 inter-arrival times, $\bar{x} = 4.2$ minutes. Find the MLE of the rate $\lambda$ and construct an approximate 95% confidence interval using asymptotic normality.

??? success "Solution to Exercise 2"
    The MLE is $\hat{\lambda} = 1/\bar{x} = 1/4.2 \approx 0.2381$ calls per minute.

    The asymptotic variance is $\hat{\lambda}^2/n = 0.2381^2/50 = 0.001133$, giving $\text{SE}(\hat{\lambda}) = \sqrt{0.001133} \approx 0.03367$.

    The 95% confidence interval is:

    $$
    \hat{\lambda} \pm 1.96 \times \text{SE}(\hat{\lambda}) = 0.2381 \pm 0.0660 = (0.172, 0.304)
    $$

    In terms of mean inter-arrival time $1/\lambda$: the MLE is 4.2 minutes, and by the invariance property, the interval for $1/\lambda$ is approximately $(1/0.304, 1/0.172) = (3.29, 5.81)$ minutes.

---

**Exercise 3.**
Show that the MLE $\hat{\lambda} = 1/\bar{X}$ is biased for $\lambda$ in finite samples. Compute the exact bias for $n = 2$.

??? success "Solution to Exercise 3"
    The sum $S = \sum X_i \sim \text{Gamma}(n, \lambda)$, so $\bar{X} = S/n$. The MLE is $\hat{\lambda} = n/S$.

    For the Gamma$(n, \lambda)$ distribution, $E[1/S] = \lambda/(n-1)$ (this is a known result for the inverse of a Gamma random variable when $n > 1$). Therefore:

    $$
    E[\hat{\lambda}] = E\!\left[\frac{n}{S}\right] = \frac{n\lambda}{n-1}
    $$

    The bias is:

    $$
    \text{Bias}(\hat{\lambda}) = \frac{n\lambda}{n-1} - \lambda = \frac{\lambda}{n-1}
    $$

    For $n = 2$: $\text{Bias} = \lambda/(2-1) = \lambda$. The MLE overestimates $\lambda$ by a factor of $n/(n-1)$. A bias-corrected estimator is $\tilde{\lambda} = (n-1)/\sum X_i$.

---

**Exercise 4.**
Compare the MLE $\hat{\lambda} = 1/\bar{X}$ with the Method of Moments estimator for $\lambda$. Are they the same?

??? success "Solution to Exercise 4"
    For the exponential distribution, $E[X] = 1/\lambda$. The MOM estimator sets $\bar{X} = 1/\hat{\lambda}$, giving $\hat{\lambda}_{\text{MOM}} = 1/\bar{X}$.

    This is identical to the MLE. The coincidence occurs because the exponential is a one-parameter exponential family, and for such families the score equation $\partial\ell/\partial\lambda = 0$ and the first moment equation $\bar{X} = E_\lambda[X]$ yield the same estimator.

    Both estimators share the same finite-sample bias of $\lambda/(n-1)$ and the same asymptotic variance $\lambda^2/n$.
