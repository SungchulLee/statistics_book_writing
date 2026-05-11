# Chi-Square Density Function

## Overview

The **chi-square distribution** with $k$ degrees of freedom arises as the distribution of a sum of $k$ independent squared standard normals:

$$
Q = Z_1^2 + Z_2^2 + \cdots + Z_k^2, \qquad Z_i \overset{\text{iid}}{\sim} N(0,1)
$$

It is fundamental in hypothesis testing (goodness-of-fit, independence tests) and in the construction of confidence intervals for the variance.

---

## PDF

$$
f(x; k) = \frac{1}{2^{k/2}\,\Gamma(k/2)}\, x^{k/2 - 1}\, e^{-x/2}, \qquad x \ge 0
$$

| Property | Value |
|---|---|
| Support | $[0, \infty)$ |
| Mean | $k$ |
| Variance | $2k$ |
| Mode | $\max(k - 2,\, 0)$ |

---

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

k = 5
chi2 = stats.chi2(df=k)

# Quantile-based x-range
x = np.linspace(chi2.ppf(1e-6), chi2.ppf(1 - 1e-6), 600)
y = chi2.pdf(x)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, y, lw=2, label=f"χ² PDF (k={k})")
ax.axvline(k, linestyle='--', alpha=0.8, label=f"mean = {k}")
ax.axvline(max(k - 2, 0), linestyle=':', alpha=0.8, label=f"mode = {max(k-2, 0)}")
ax.set_title("Chi-square Distribution — PDF")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.legend()
ax.grid(True, linestyle=":")
plt.tight_layout()
plt.show()
```

---

## Shape vs Degrees of Freedom

- **$k = 1, 2$:** Highly right-skewed, density peaks at or near 0.
- **$k \approx 10$:** Moderate skew, bell-like but asymmetric.
- **Large $k$:** By the CLT, $\chi^2_k \approx N(k, 2k)$.

!!! note "Connection to Normal"
    Since $\chi^2_k$ is a sum of $k$ i.i.d. random variables (each $Z_i^2$), the CLT guarantees approximate normality for large $k$.

---

## Exercises

**Exercise 1.**
Compute $E[Q]$ and $\text{Var}(Q)$ for $Q \sim \chi^2_k$ from the definition $Q = \sum_{i=1}^k Z_i^2$.

??? success "Solution to Exercise 1"
    For each $Z_i^2$: $E[Z_i^2] = 1$ and $\text{Var}(Z_i^2) = E[Z_i^4] - (E[Z_i^2])^2 = 3 - 1 = 2$.

    By independence:

    $$
    E[Q] = \sum_{i=1}^k E[Z_i^2] = k, \qquad \text{Var}(Q) = \sum_{i=1}^k \text{Var}(Z_i^2) = 2k
    $$

---

**Exercise 2.**
If $X \sim \chi^2_m$ and $Y \sim \chi^2_n$ are independent, show that $X + Y \sim \chi^2_{m+n}$.

??? success "Solution to Exercise 2"
    Write $X = \sum_{i=1}^m Z_i^2$ and $Y = \sum_{j=1}^n W_j^2$ where all $Z_i, W_j$ are independent $N(0,1)$. Then:

    $$
    X + Y = \sum_{i=1}^m Z_i^2 + \sum_{j=1}^n W_j^2
    $$

    This is a sum of $m + n$ independent squared standard normals, so $X + Y \sim \chi^2_{m+n}$ by definition. $\square$

---

**Exercise 3.**
Show that the mode of $\chi^2_k$ is $k - 2$ for $k \ge 2$ by differentiating the PDF and setting the result to zero.

??? success "Solution to Exercise 3"
    Taking the log of the PDF: $\ln f(x) = \text{const} + (k/2 - 1)\ln x - x/2$. Differentiating:

    $$
    \frac{d}{dx}\ln f(x) = \frac{k/2 - 1}{x} - \frac{1}{2} = 0
    $$

    Solving: $x = k - 2$. For $k \ge 2$, this is nonneg and lies in the support $[0, \infty)$, so the mode is $k - 2$. For $k < 2$, the derivative is always negative for $x > 0$, so the mode is at $x = 0$.

---

**Exercise 4.**
A random sample of size $n = 25$ from $N(\mu, \sigma^2)$ yields $s^2 = 12$. Construct a 95% confidence interval for $\sigma^2$ using the chi-square distribution.

??? success "Solution to Exercise 4"
    The pivotal quantity is $(n-1)s^2/\sigma^2 \sim \chi^2_{n-1}$. With $n-1 = 24$:

    $$
    P\!\left(\chi^2_{0.025} \le \frac{24 \cdot 12}{\sigma^2} \le \chi^2_{0.975}\right) = 0.95
    $$

    Using SciPy: $\chi^2_{0.025, 24} = 12.40$ and $\chi^2_{0.975, 24} = 39.36$.

    $$
    \frac{24 \times 12}{39.36} \le \sigma^2 \le \frac{24 \times 12}{12.40}
    $$

    $$
    7.32 \le \sigma^2 \le 23.23
    $$
