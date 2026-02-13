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
