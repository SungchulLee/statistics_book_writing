# MLE for Normal Distribution

## Overview

Let $x^{(i)}$ be $m$ i.i.d. samples from $N(\mu, \sigma^2)$. Then, $\mu$ and $\sigma^2$ can be estimated by $\hat{\mu}$ and $\hat{\sigma}^2$ where:

$$
\begin{array}{lll}
\hat{\mu} &=& \displaystyle\frac{\sum_{i=1}^m x^{(i)}}{m} \\[12pt]
\hat{\sigma}^2 &=& \displaystyle\frac{\sum_{i=1}^m (x^{(i)} - \hat{\mu})^2}{m}
\end{array}
$$

## Derivation

### Data

$$
\{x^{(i)} : i = 1, \ldots, m\}
$$

### Model

$$
x^{(i)} \sim N(\mu, \sigma^2)
$$

### Likelihood Function

$$
L(\mu, \sigma^2) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{1}{2\sigma^2}(x^{(i)} - \mu)^2\right)
$$

### Log-Likelihood Function

$$
\ell(\mu, \sigma^2) = -\frac{1}{2\sigma^2}\sum_{i=1}^m (x^{(i)} - \mu)^2 - \frac{m}{2}\log\sigma^2 + \text{Constant}
$$

### Cost Function

$$
J(\mu, \sigma^2) = \frac{1}{2\sigma^2}\sum_{i=1}^m (x^{(i)} - \mu)^2 + \frac{m}{2}\log\sigma^2
$$

### Maximum Likelihood Principle

$$
\text{argmax}_{\mu, \sigma^2}\; L
\quad\Leftrightarrow\quad
\text{argmax}_{\mu, \sigma^2}\; \ell
\quad\Leftrightarrow\quad
\text{argmin}_{\mu, \sigma^2}\; J
$$

### MLE Solutions

$$
\begin{array}{llcll}
\displaystyle\frac{\partial J}{\partial \mu} = 0
&\Rightarrow&
\displaystyle\sum_{i=1}^m (x^{(i)} - \mu) = 0
&\Rightarrow&
\displaystyle\hat{\mu} = \frac{\sum_{i=1}^m x^{(i)}}{m} \\[16pt]
\displaystyle\frac{\partial J}{\partial \sigma^2} = 0
&\Rightarrow&
\cdots
&\Rightarrow&
\displaystyle\hat{\sigma}^2 = \frac{\sum_{i=1}^m (x^{(i)} - \hat{\mu})^2}{m}
\end{array}
$$

## Key Observations

| Estimator | MLE | Unbiased? |
|-----------|-----|-----------|
| $\hat{\mu}$ | $\frac{1}{m}\sum x^{(i)}$ | ✅ Yes |
| $\hat{\sigma}^2$ | $\frac{1}{m}\sum (x^{(i)} - \hat{\mu})^2$ | ❌ No (divides by $m$, not $m-1$) |

!!! note "MLE Bias for Variance"
    The MLE $\hat{\sigma}^2$ divides by $m$, making it a biased estimator of $\sigma^2$. The unbiased sample variance $S^2$ divides by $m - 1$ (Bessel's correction):

    $$
    S^2 = \frac{\sum_{i=1}^m (x^{(i)} - \hat{\mu})^2}{m - 1}
    $$

## Connection to Least Squares

The cost function for $\mu$ (with $\sigma^2$ fixed) is:

$$
J(\mu) \propto \sum_{i=1}^m (x^{(i)} - \mu)^2
$$

This is exactly the **least squares** objective. Thus, the MLE for the mean of a normal distribution is equivalent to the least squares estimate — a deep connection between MLE and regression.

## Exercises

**Exercise 1.**
Derive the MLE for $\mu$ from a sample $x_1, \ldots, x_n \sim N(\mu, \sigma^2)$ with $\sigma^2$ known, by differentiating the log-likelihood with respect to $\mu$.

??? success "Solution to Exercise 1"
    The log-likelihood is:

    $$
    \ell(\mu) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
    $$

    Differentiating with respect to $\mu$:

    $$
    \frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = \frac{1}{\sigma^2}\left(\sum x_i - n\mu\right)
    $$

    Setting to zero:

    $$
    \sum x_i - n\mu = 0 \implies \hat{\mu} = \frac{\sum x_i}{n} = \bar{x}
    $$

    The second derivative is $-n/\sigma^2 < 0$, confirming a maximum.

---

**Exercise 2.**
Show that the MLE $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$ is biased, and compute $E[\hat{\sigma}^2]$.

??? success "Solution to Exercise 2"
    Since $\sum(X_i - \bar{X})^2 = (n-1)S^2$ where $S^2$ is the unbiased sample variance with $E[S^2] = \sigma^2$:

    $$
    E[\hat{\sigma}^2] = E\!\left[\frac{1}{n}\sum(X_i - \bar{X})^2\right] = \frac{n-1}{n}E[S^2] \cdot \frac{n}{n-1} \cdot \frac{n-1}{n} = \frac{n-1}{n}\sigma^2
    $$

    The bias is $E[\hat{\sigma}^2] - \sigma^2 = -\sigma^2/n$. The MLE underestimates the true variance. This motivates using $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$ as the unbiased estimator.

---

**Exercise 3.**
Explain the connection between maximizing the normal log-likelihood with respect to $\mu$ and minimizing the sum of squared residuals. Why does this connection break down for non-normal distributions?

??? success "Solution to Exercise 3"
    For a normal distribution, the log-likelihood with respect to $\mu$ (treating $\sigma^2$ as fixed) is:

    $$
    \ell(\mu) = \text{const} - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
    $$

    Maximizing $\ell(\mu)$ is equivalent to minimizing $\sum(x_i - \mu)^2$ because the constant and the factor $-1/(2\sigma^2)$ do not change the argmax. This is the least squares objective.

    For non-normal distributions, the log-likelihood contains a different function of $(x_i - \mu)$. For example, for the Laplace distribution, $\ell(\mu) \propto -\sum |x_i - \mu|$, so the MLE minimizes the sum of absolute deviations (giving the median, not the mean). The least-squares connection is specific to the normal distribution's quadratic exponent.

---

**Exercise 4.**
A sample of 5 observations from a normal distribution gives values 3, 5, 7, 9, 11. Compute the MLE for both $\mu$ and $\sigma^2$.

??? success "Solution to Exercise 4"
    The MLE for $\mu$ is the sample mean:

    $$
    \hat{\mu} = \frac{3+5+7+9+11}{5} = \frac{35}{5} = 7
    $$

    The MLE for $\sigma^2$ divides by $n$ (not $n-1$):

    $$
    \hat{\sigma}^2 = \frac{1}{5}\sum(x_i - 7)^2 = \frac{(3-7)^2 + (5-7)^2 + (7-7)^2 + (9-7)^2 + (11-7)^2}{5}
    $$

    $$
    = \frac{16 + 4 + 0 + 4 + 16}{5} = \frac{40}{5} = 8
    $$

    Note: the unbiased estimate would be $S^2 = 40/4 = 10$.
