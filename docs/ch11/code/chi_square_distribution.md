# Chi Square Distribution

## Overview

The chi-squared distribution is one of the most fundamental distributions in inferential statistics, underpinning tests for variance, goodness of fit, and independence. It arises naturally as the distribution of a sum of squared standard normal random variables. This page defines the distribution, derives its key properties, visualizes its PDF and CDF, and demonstrates that sampling from $\chi^2(d)$ directly and constructing it from $Z^2$ sums produce identical distributions.

## Definition

If $Z_1, Z_2, \dots, Z_d$ are independent standard normal random variables, $Z_i \sim N(0,1)$, then the sum of their squares follows a chi-squared distribution with $d$ degrees of freedom:

$$
Q = \sum_{i=1}^{d} Z_i^2 \sim \chi^2(d)
$$

The probability density function for $x > 0$ is

$$
f(x;\, d) = \frac{1}{2^{d/2}\, \Gamma(d/2)}\, x^{d/2 - 1}\, e^{-x/2}
$$

where $\Gamma(\cdot)$ is the gamma function.

## Key Properties

| Property | Value |
|---|---|
| Mean | $d$ |
| Variance | $2d$ |
| Mode | $\max(d - 2,\, 0)$ |
| Skewness | $\sqrt{8/d}$ |
| MGF | $(1 - 2t)^{-d/2}$ for $t < 1/2$ |

As $d \to \infty$, the chi-squared distribution approaches a normal distribution by the Central Limit Theorem:

$$
\frac{Q - d}{\sqrt{2d}} \xrightarrow{d} N(0, 1)
$$

## Additivity Property

If $Q_1 \sim \chi^2(d_1)$ and $Q_2 \sim \chi^2(d_2)$ are independent, then

$$
Q_1 + Q_2 \sim \chi^2(d_1 + d_2)
$$

This follows directly from the definition: the sum of $d_1 + d_2$ independent squared standard normals has $d_1 + d_2$ degrees of freedom.

## Visualizing the PDF and CDF

The following code plots the PDF and CDF for a given degrees-of-freedom parameter:

```python
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

df = 5
x = np.linspace(0, 30, 100)
pdf = stats.chi2(df=df).pdf(x)
cdf = stats.chi2(df=df).cdf(x)

fig, ax = plt.subplots()
ax.plot(x, pdf, label="PDF")
ax.plot(x, cdf, label="CDF")
ax.legend()
ax.set_title(f"PDF and CDF of chi-squared({df})")
plt.show()
```

For small $d$, the PDF is right-skewed with a mode near zero. As $d$ increases, the distribution becomes more symmetric and shifts to the right.

## Construction from Squared Normals

The defining property can be verified empirically by comparing two approaches:

1. **Direct sampling:** draw 10,000 values from $\chi^2(d)$.
2. **Construction:** draw a $d \times 10{,}000$ matrix of $N(0,1)$ values, square each entry, and sum along columns.

```python
df, seed = 5, 1
data_direct = stats.chi2(df=df).rvs(10_000, random_state=seed)
data_from_norm = np.sum(
    stats.norm().rvs(size=(df, 10_000), random_state=seed) ** 2,
    axis=0
)
```

Overlaying histograms of both samples against the theoretical PDF confirms that they match, validating the definition $\sum Z_i^2 \sim \chi^2(d)$.

## Interpretation

- The degrees of freedom $d$ controls the shape: small $d$ gives a highly skewed distribution concentrated near zero, while large $d$ produces a nearly symmetric bell shape centered at $d$.
- The chi-squared distribution only takes positive values, consistent with its definition as a sum of squares.
- Critical values $\chi^2_{1-\alpha}(d)$ are widely used in hypothesis testing. For instance, the upper 5% critical value for $\chi^2(5)$ is approximately 11.07.

## Exercises

**Exercise 1.**
Compute $E[Q]$ and $\text{Var}(Q)$ for $Q \sim \chi^2(10)$ using the properties table above, and verify by computing $E[Z^2]$ and $\text{Var}(Z^2)$ for $Z \sim N(0,1)$.

??? success "Solution to Exercise 1"
    From the table, $E[Q] = d = 10$ and $\text{Var}(Q) = 2d = 20$.

    Verification: For $Z \sim N(0,1)$, $E[Z^2] = 1$ and $\text{Var}(Z^2) = E[Z^4] - (E[Z^2])^2 = 3 - 1 = 2$. Since $Q = \sum_{i=1}^{10} Z_i^2$ with independent terms,

    $$
    E[Q] = 10 \cdot 1 = 10, \qquad \text{Var}(Q) = 10 \cdot 2 = 20
    $$

    Both approaches agree.

---

**Exercise 2.**
Show that the moment generating function of $\chi^2(d)$ is $M_Q(t) = (1 - 2t)^{-d/2}$ for $t < 1/2$, starting from the MGF of $Z^2$ where $Z \sim N(0,1)$.

??? success "Solution to Exercise 2"
    For $Z \sim N(0,1)$, the MGF of $Z^2$ is

    $$
    M_{Z^2}(t) = E[e^{tZ^2}] = \int_{-\infty}^{\infty} e^{tz^2} \frac{1}{\sqrt{2\pi}} e^{-z^2/2}\, dz = \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}} e^{-z^2(1 - 2t)/2}\, dz
    $$

    This integral converges when $1 - 2t > 0$, i.e., $t < 1/2$. Completing the Gaussian integral gives $M_{Z^2}(t) = (1 - 2t)^{-1/2}$.

    Since $Q = \sum_{i=1}^{d} Z_i^2$ with independent $Z_i$, the MGF of $Q$ is the product of the individual MGFs:

    $$
    M_Q(t) = \prod_{i=1}^{d} (1 - 2t)^{-1/2} = (1 - 2t)^{-d/2}
    $$

    for $t < 1/2$. $\square$

---

**Exercise 3.**
If $Q_1 \sim \chi^2(3)$ and $Q_2 \sim \chi^2(7)$ are independent, find the distribution of $Q_1 + Q_2$ and compute $P(Q_1 + Q_2 > 18.31)$.

??? success "Solution to Exercise 3"
    By the additivity property, $Q_1 + Q_2 \sim \chi^2(3 + 7) = \chi^2(10)$.

    The value 18.31 is the upper 5% critical value of $\chi^2(10)$, so

    $$
    P(Q_1 + Q_2 > 18.31) = 0.05
    $$

---

**Exercise 4.**
Explain why the chi-squared distribution with $d = 2$ is an exponential distribution. Identify the rate parameter.

??? success "Solution to Exercise 4"
    Setting $d = 2$ in the PDF:

    $$
    f(x;\, 2) = \frac{1}{2^1 \Gamma(1)} x^{0} e^{-x/2} = \frac{1}{2} e^{-x/2}, \qquad x > 0
    $$

    This is exactly the PDF of an $\text{Exponential}(\lambda = 1/2)$ distribution (equivalently, $\text{Exponential}$ with mean 2). The connection is not surprising: $\chi^2(d)$ is a special case of the $\text{Gamma}(d/2, 1/2)$ distribution, and $\text{Gamma}(1, \lambda) = \text{Exponential}(\lambda)$.

---

**Exercise 5.**
A researcher wants to use the normal approximation $(Q - d)/\sqrt{2d} \approx N(0,1)$ to find the upper 5% critical value of $\chi^2(50)$. Compute the approximate critical value and compare it to the exact value of 67.50.

??? success "Solution to Exercise 5"
    Using the approximation with $d = 50$ and $z_{0.95} = 1.645$:

    $$
    Q \approx d + z_{0.95}\sqrt{2d} = 50 + 1.645\sqrt{100} = 50 + 16.45 = 66.45
    $$

    The exact value is 67.50, so the approximation underestimates by about 1.05 (a relative error of roughly 1.6%). The approximation is reasonable for $d = 50$ and improves further as $d$ increases. For higher accuracy, Wilson-Hilferty's cube-root transformation $\bigl(\frac{Q}{d}\bigr)^{1/3} \approx N\!\bigl(1 - \frac{2}{9d},\, \frac{2}{9d}\bigr)$ is often preferred.
