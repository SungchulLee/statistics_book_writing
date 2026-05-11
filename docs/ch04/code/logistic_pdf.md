# Logistic Density Function (vs Normal)

## Overview

The **logistic distribution** with location $\mu$ and scale $s$ has PDF:

$$
f(x) = \frac{e^{-(x-\mu)/s}}{s\left(1 + e^{-(x-\mu)/s}\right)^2}
$$

| Property | Value |
|---|---|
| Mean | $\mu$ |
| Variance | $s^2\pi^2/3$ |
| Support | $(-\infty, \infty)$ |

The logistic distribution resembles the normal but has **heavier tails**, making it useful for modelling fat-tailed phenomena such as financial returns. Its CDF has the convenient closed form $F(x) = 1/(1 + e^{-(x-\mu)/s})$, which is the logistic (sigmoid) function used throughout machine learning.

---

## Comparison with Normal

To compare the two distributions fairly, we match their variances. If the logistic has scale $s$, its variance is $s^2\pi^2/3$. The matching normal has $\sigma = s\pi/\sqrt{3}$.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu, s = 0, 1
x = np.linspace(-8, 8, 400)

rv_logistic = stats.logistic(loc=mu, scale=s)
sigma = s * np.pi / np.sqrt(3)  # matching variance
rv_normal = stats.norm(loc=mu, scale=sigma)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(x, rv_logistic.pdf(x), label='Logistic(0, 1)')
ax.plot(x, rv_normal.pdf(x), '--', label=rf'Normal(0, {sigma:.2f}²) [same var]')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Logistic vs Normal Distribution — PDF')
ax.legend()
plt.tight_layout()
plt.show()
```

The logistic curve is slightly lower at the center and higher in the tails than the matched normal.

---

## Exercises

**Exercise 1.**
Show that the logistic CDF $F(x) = 1/(1 + e^{-(x-\mu)/s})$ is the antiderivative of the logistic PDF.

??? success "Solution to Exercise 1"
    Differentiate $F(x) = (1 + e^{-(x-\mu)/s})^{-1}$:

    $$
    F'(x) = \frac{e^{-(x-\mu)/s}/s}{(1 + e^{-(x-\mu)/s})^2} = f(x)
    $$

    confirming that $F'(x) = f(x)$. $\square$

---

**Exercise 2.**
Compute the logistic variance $s^2\pi^2/3$ and verify it is larger than the variance of a normal distribution with the same scale parameter $\sigma = s$.

??? success "Solution to Exercise 2"
    The logistic with scale $s$ has variance $s^2\pi^2/3 \approx 3.29 s^2$. The normal with $\sigma = s$ has variance $s^2$. So the logistic variance is $\pi^2/3 \approx 3.29$ times larger—consistent with its heavier tails.

---

**Exercise 3.**
The logistic distribution plays a central role in logistic regression. Explain how the CDF $F(x) = 1/(1 + e^{-x})$ serves as the link function that maps a linear predictor to a probability.

??? success "Solution to Exercise 3"
    In logistic regression, the model specifies $P(Y = 1 \mid \mathbf{x}) = \sigma(\mathbf{x}^\top\boldsymbol{\beta})$ where $\sigma(z) = 1/(1+e^{-z})$ is the logistic CDF. The linear predictor $z = \mathbf{x}^\top\boldsymbol{\beta}$ can take any value in $(-\infty, \infty)$, and the logistic function maps it to $(0, 1)$, producing a valid probability. The inverse, $z = \ln(p/(1-p))$ (the log-odds or logit), links the probability scale to the linear scale.

---

**Exercise 4.**
Compare the tail probabilities $P(|X| > 3)$ for the standard logistic ($\mu=0, s=1$) and the variance-matched normal. Which has more tail probability?

??? success "Solution to Exercise 4"
    **Logistic:** $P(|X| > 3) = 2 \cdot S(3) = 2/(1 + e^3) \approx 2 \times 0.0474 = 0.0949$.

    **Matched normal** ($\sigma = \pi/\sqrt{3} \approx 1.814$): $P(|X| > 3) = 2\mathcal{N}(-3/1.814) = 2\mathcal{N}(-1.654) \approx 2 \times 0.0491 = 0.0982$.

    The tail probabilities are similar at this threshold, but at more extreme values (e.g., $|X| > 5$) the logistic tails dominate because they decay exponentially ($\sim e^{-|x|/s}$) vs. the normal's Gaussian decay ($\sim e^{-x^2/(2\sigma^2)}$).
