# Log-Normal Density Function

## Overview

If $X \sim N(\mu, \sigma^2)$, then $Y = e^X$ follows the **log-normal distribution** with parameters $\mu$ and $\sigma$. Equivalently, $\ln Y \sim N(\mu, \sigma^2)$.

$$
f(y) = \frac{1}{y\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(\ln y - \mu)^2}{2\sigma^2}\right), \qquad y > 0
$$

| Property | Value |
|---|---|
| Support | $(0, \infty)$ |
| Mean | $e^{\mu + \sigma^2/2}$ |
| Median | $e^{\mu}$ |
| Variance | $(e^{\sigma^2} - 1)\,e^{2\mu + \sigma^2}$ |
| Mode | $e^{\mu - \sigma^2}$ |

The log-normal is widely used to model asset prices, incomes, and other strictly positive, right-skewed quantities.

---

## SciPy Parameterization

SciPy uses `stats.lognorm(s=sigma, scale=np.exp(mu))`, where `s` is the shape parameter $\sigma$ and `scale` is the median $e^\mu$.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu = 0
sigmas = [0.5, 1.0, 1.5, 2.0]
x = np.linspace(0.001, 8, 500)

fig, ax = plt.subplots(figsize=(12, 4))
for sigma in sigmas:
    rv = stats.lognorm(s=sigma, scale=np.exp(mu))
    ax.plot(x, rv.pdf(x), label=rf'$\sigma={sigma}$')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title(r'Log-Normal Distribution — PDF ($\mu=0$, varying $\sigma$)')
ax.legend()
ax.set_ylim(bottom=-0.02)
plt.tight_layout()
plt.show()
```

As $\sigma$ increases, the distribution becomes more right-skewed and the mode shifts toward zero.

---

## Exercises

**Exercise 1.**
Derive $E[Y]$ for $Y = e^X$ where $X \sim N(\mu, \sigma^2)$ using the moment generating function of the normal distribution.

??? success "Solution to Exercise 1"
    The MGF of $X \sim N(\mu, \sigma^2)$ is $M_X(t) = E[e^{tX}] = e^{\mu t + \sigma^2 t^2/2}$.

    Setting $t = 1$:

    $$
    E[Y] = E[e^X] = M_X(1) = e^{\mu + \sigma^2/2}
    $$

---

**Exercise 2.**
Show that the median of the log-normal is $e^\mu$, which is less than the mean $e^{\mu + \sigma^2/2}$ for any $\sigma > 0$.

??? success "Solution to Exercise 2"
    The median $m$ satisfies $P(Y \le m) = 0.5$:

    $$
    P(e^X \le m) = P(X \le \ln m) = \mathcal{N}\!\left(\frac{\ln m - \mu}{\sigma}\right) = 0.5
    $$

    This requires $(\ln m - \mu)/\sigma = 0$, so $m = e^\mu$.

    Since $\sigma^2/2 > 0$, we have $e^{\mu + \sigma^2/2} > e^\mu$, confirming mean > median. This reflects the right skewness: the mean is pulled upward by the heavy right tail.

---

**Exercise 3.**
If stock returns are log-normally distributed with $\mu = 0.05$ and $\sigma = 0.2$ (annualized), what is the probability that the stock loses more than 20% of its value?

??? success "Solution to Exercise 3"
    A 20% loss means $Y < 0.8$ (the stock is worth less than 80% of its initial value):

    $$
    P(Y < 0.8) = P(X < \ln 0.8) = \mathcal{N}\!\left(\frac{\ln 0.8 - 0.05}{0.2}\right) = \mathcal{N}\!\left(\frac{-0.2231 - 0.05}{0.2}\right) = \mathcal{N}(-1.366) \approx 0.086
    $$

    About 8.6% probability of losing more than 20%.

---

**Exercise 4.**
Prove that the product of independent log-normal random variables is again log-normal.

??? success "Solution to Exercise 4"
    Let $Y_1 = e^{X_1}$ and $Y_2 = e^{X_2}$ where $X_1 \sim N(\mu_1, \sigma_1^2)$ and $X_2 \sim N(\mu_2, \sigma_2^2)$ are independent. Then:

    $$
    Y_1 Y_2 = e^{X_1 + X_2}
    $$

    Since $X_1 + X_2 \sim N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$ (sum of independent normals), $Y_1 Y_2$ is log-normal with parameters $\mu_1 + \mu_2$ and $\sqrt{\sigma_1^2 + \sigma_2^2}$.

    By induction, any finite product of independent log-normals is log-normal. $\square$
