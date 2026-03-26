# Fisher Information and Standard Errors

## Why Quantify Information?

When estimating a parameter $\theta$ from data, we want to know: how precisely can $\theta$ be estimated? The answer depends on how much "information" the data carry about $\theta$. Fisher information formalizes this notion. It determines the sharpest possible precision for any unbiased estimator and directly provides standard errors for maximum likelihood estimates. In short, Fisher information is the bridge between a parametric model and the quality of the inferences we can draw from it.

## The Score Function

To define Fisher information, we first introduce the **score function**. Let $X$ be a random variable with probability density or mass function $f(x; \theta)$. The score function is the derivative of the log-likelihood with respect to $\theta$:

$$
s(x; \theta) = \frac{\partial}{\partial \theta} \log f(x; \theta)
$$

The score function captures how sensitive the log-likelihood is to changes in $\theta$. When the log-likelihood changes sharply as $\theta$ varies, the data are highly informative about $\theta$; when it is flat, the data carry little information.

Under regularity conditions — specifically, that the support $\{x : f(x; \theta) > 0\}$ does not depend on $\theta$ and that differentiation and integration can be interchanged — the score function has a key property:

$$
E[s(X; \theta)] = 0
$$

This means the score function is centered at zero, and its variance measures the typical magnitude of the log-likelihood's slope.

## Definition of Fisher Information

The **Fisher information** for a single observation is the variance of the score function:

$$
I(\theta) = E\left[\left(\frac{\partial}{\partial \theta} \log f(X;\theta)\right)^2\right] = \text{Var}(s(X; \theta))
$$

Under the same regularity conditions (interchange of differentiation and integration), this is equivalent to the negative expected curvature of the log-likelihood:

$$
I(\theta) = -E\left[\frac{\partial^2}{\partial\theta^2} \log f(X;\theta)\right]
$$

The second form is often easier to compute. It also provides geometric intuition: a sharply curved log-likelihood (large $I(\theta)$) means the data strongly constrain the parameter, while a flat log-likelihood (small $I(\theta)$) means the data are uninformative.

For $n$ independent and identically distributed observations $X_1, \ldots, X_n$, the total Fisher information is additive:

$$
I_n(\theta) = n \cdot I(\theta)
$$

## Worked Examples

### Bernoulli Distribution

Let $X \sim \text{Bernoulli}(p)$, so $f(x; p) = p^x (1-p)^{1-x}$ for $x \in \{0, 1\}$. The log-likelihood for a single observation is

$$
\log f(x; p) = x \log p + (1-x) \log(1-p)
$$

Taking the second derivative:

$$
\frac{\partial^2}{\partial p^2} \log f(x; p) = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}
$$

Since $E[X] = p$:

$$
I(p) = -E\left[-\frac{X}{p^2} - \frac{1-X}{(1-p)^2}\right] = \frac{1}{p} \cdot \frac{1}{1} + \frac{1}{(1-p)} = \frac{1}{p(1-p)}
$$

Notice that Fisher information is largest when $p$ is near 0 or 1 (where each observation is most informative about $p$) and smallest when $p = 1/2$.

### Normal Distribution (Mean)

Let $X \sim N(\mu, \sigma^2)$ with $\sigma^2$ known. The log-likelihood for a single observation is

$$
\log f(x; \mu) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x - \mu)^2}{2\sigma^2}
$$

The second derivative with respect to $\mu$ is

$$
\frac{\partial^2}{\partial \mu^2} \log f(x; \mu) = -\frac{1}{\sigma^2}
$$

This is a constant (does not depend on $x$), so

$$
I(\mu) = \frac{1}{\sigma^2}
$$

The Fisher information increases as the noise $\sigma^2$ decreases, which matches intuition: less noisy data carry more information about the mean.

### Poisson Distribution

Let $X \sim \text{Poisson}(\lambda)$, so $f(x; \lambda) = e^{-\lambda}\lambda^x / x!$. The log-likelihood is

$$
\log f(x; \lambda) = -\lambda + x \log \lambda - \log(x!)
$$

The second derivative is

$$
\frac{\partial^2}{\partial \lambda^2} \log f(x; \lambda) = -\frac{x}{\lambda^2}
$$

Since $E[X] = \lambda$:

$$
I(\lambda) = \frac{1}{\lambda}
$$

## Standard Errors from Fisher Information

One of the most important applications of Fisher information is providing standard errors for the MLE. Under regularity conditions, the MLE $\hat{\theta}_{\text{MLE}}$ is asymptotically normal:

$$
\hat{\theta}_{\text{MLE}} \overset{d}{\to} N\left(\theta, \frac{1}{nI(\theta)}\right) \quad \text{as } n \to \infty
$$

In practice, we estimate the standard error by evaluating the Fisher information at the MLE (the plug-in principle):

$$
\widehat{\text{SE}}(\hat{\theta}_{\text{MLE}}) = \frac{1}{\sqrt{nI(\hat{\theta}_{\text{MLE}})}}
$$

This approximation improves with sample size and forms the basis for constructing confidence intervals and Wald test statistics.

!!! example "Standard Error for the Bernoulli MLE"

    For $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$, the MLE is $\hat{p} = \bar{X}$. The estimated standard error is

    $$
    \widehat{\text{SE}}(\hat{p}) = \frac{1}{\sqrt{n \cdot \frac{1}{\hat{p}(1-\hat{p})}}} = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
    $$

    This is the familiar formula for the standard error of a sample proportion.

## Summary of Common Fisher Information Values

| Distribution | Parameter | Fisher Information $I(\theta)$ |
|---|---|---|
| $\text{Bernoulli}(p)$ | $p$ | $\dfrac{1}{p(1-p)}$ |
| $N(\mu, \sigma^2)$ (known $\sigma^2$) | $\mu$ | $\dfrac{1}{\sigma^2}$ |
| $\text{Poisson}(\lambda)$ | $\lambda$ | $\dfrac{1}{\lambda}$ |
| $\text{Exp}(\lambda)$ | $\lambda$ | $\dfrac{1}{\lambda^2}$ |

Each entry in this table can be verified using the second-derivative method demonstrated in the worked examples above.
