# Normal Density Function with scipy.stats

## Overview

The **normal (Gaussian) distribution** is the most important continuous distribution in statistics. Its probability density function (PDF) is the familiar bell curve, symmetric about the mean $\mu$ with spread governed by the standard deviation $\sigma$.

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)
$$

About 99.7% of the probability mass lies within $\mu \pm 3\sigma$ (the 68–95–99.7 rule).

---

## scipy.stats Interface

SciPy represents the normal distribution via `stats.norm(loc=mu, scale=sigma)`, creating a **frozen** distribution object. The `loc` parameter is $\mu$ and `scale` is $\sigma$ (not $\sigma^2$).

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu = 1       # mean
sigma = 2    # standard deviation

# x-grid covering mu +/- 3*sigma
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

# Evaluate PDF
y = stats.norm(loc=mu, scale=sigma).pdf(x)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, y)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title(f"Normal({mu}, {sigma}²) PDF")
plt.show()
```

---

## Key Properties

| Property | Value |
|---|---|
| Support | $(-\infty, \infty)$ |
| Mean | $\mu$ |
| Variance | $\sigma^2$ |
| Mode | $\mu$ |
| Skewness | 0 |
| Kurtosis (excess) | 0 |

The normal distribution is the reference for symmetry (zero skewness) and tail weight (zero excess kurtosis). All other distributions are compared to it.

---

## Exercises

**Exercise 1.**
For $X \sim N(1, 4)$, compute $f(1)$ (the density at the mean) by hand.

??? success "Solution to Exercise 1"
    With $\mu = 1$ and $\sigma^2 = 4$ ($\sigma = 2$):

    $$
    f(1) = \frac{1}{2\sqrt{2\pi}} \exp(0) = \frac{1}{2\sqrt{2\pi}} \approx 0.1995
    $$

---

**Exercise 2.**
Show that the normal PDF integrates to 1. (Hint: compute $I = \int_{-\infty}^{\infty} e^{-x^2/2}\,dx$ by evaluating $I^2$ in polar coordinates.)

??? success "Solution to Exercise 2"
    Let $I = \int_{-\infty}^{\infty} e^{-x^2/2}\,dx$. Then:

    $$
    I^2 = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty} e^{-(x^2+y^2)/2}\,dx\,dy
    $$

    Switching to polar coordinates $x = r\cos\theta$, $y = r\sin\theta$:

    $$
    I^2 = \int_0^{2\pi}\int_0^{\infty} e^{-r^2/2}\,r\,dr\,d\theta = 2\pi \cdot \left[-e^{-r^2/2}\right]_0^{\infty} = 2\pi
    $$

    So $I = \sqrt{2\pi}$, and $\frac{1}{\sigma\sqrt{2\pi}}\int_{-\infty}^{\infty} e^{-(x-\mu)^2/(2\sigma^2)}\,dx = 1$. $\square$

---

**Exercise 3.**
Derive the inflection points of the normal PDF. At what values of $x$ does the curvature change sign?

??? success "Solution to Exercise 3"
    The inflection points occur where $f''(x) = 0$. Computing:

    $$
    f'(x) = -\frac{x - \mu}{\sigma^2} f(x)
    $$

    $$
    f''(x) = \left(\frac{(x-\mu)^2}{\sigma^4} - \frac{1}{\sigma^2}\right) f(x)
    $$

    Setting $f''(x) = 0$ (and noting $f(x) > 0$):

    $$
    (x - \mu)^2 = \sigma^2 \implies x = \mu \pm \sigma
    $$

    The inflection points are at $x = \mu - \sigma$ and $x = \mu + \sigma$, exactly one standard deviation from the mean.

---

**Exercise 4.**
Using SciPy, verify the 68–95–99.7 rule numerically for the standard normal distribution.

??? success "Solution to Exercise 4"
    ```python
    dist = stats.norm(0, 1)
    for k in [1, 2, 3]:
        prob = dist.cdf(k) - dist.cdf(-k)
        print(f"P(-{k} < Z < {k}) = {prob:.4f}")
    ```

    Output:

    - $P(-1 < Z < 1) = 0.6827$ (approximately 68%)
    - $P(-2 < Z < 2) = 0.9545$ (approximately 95%)
    - $P(-3 < Z < 3) = 0.9973$ (approximately 99.7%)
