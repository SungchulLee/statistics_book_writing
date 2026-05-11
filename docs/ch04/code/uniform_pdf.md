# Uniform Density Function

## Overview

The **continuous uniform distribution** on the interval $[a, b]$ assigns equal probability density to every point in the interval:

$$
f(x) = \frac{1}{b - a}, \qquad a \le x \le b
$$

| Property | Value |
|---|---|
| Support | $[a, b]$ |
| Mean | $(a + b)/2$ |
| Variance | $(b - a)^2/12$ |
| CDF | $(x - a)/(b - a)$ for $x \in [a, b]$ |

The uniform distribution is the maximum-entropy distribution on a bounded interval — it makes the fewest assumptions about where values fall.

---

## SciPy Parameterization

SciPy uses `stats.uniform(loc=a, scale=b-a)`, where `loc` is the left endpoint and `scale` is the interval width.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

intervals = [(0, 1), (-2, 2), (1, 5)]
x = np.linspace(-3, 6, 500)

fig, ax = plt.subplots(figsize=(12, 4))
for a, b in intervals:
    rv = stats.uniform(loc=a, scale=b - a)
    ax.plot(x, rv.pdf(x), label=f'Uniform({a}, {b})')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Uniform Distribution — PDF')
ax.legend()
ax.set_ylim(bottom=-0.05)
plt.tight_layout()
plt.show()
```

Wider intervals produce shorter (but wider) rectangles, since the total area must equal 1.

---

## Exercises

**Exercise 1.**
Derive the mean and variance of $X \sim \text{Uniform}(a, b)$ by direct integration.

??? success "Solution to Exercise 1"
    **Mean:**

    $$
    E[X] = \int_a^b \frac{x}{b-a}\,dx = \frac{1}{b-a}\cdot\frac{b^2 - a^2}{2} = \frac{a+b}{2}
    $$

    **Second moment:**

    $$
    E[X^2] = \int_a^b \frac{x^2}{b-a}\,dx = \frac{b^3 - a^3}{3(b-a)} = \frac{a^2 + ab + b^2}{3}
    $$

    **Variance:**

    $$
    \text{Var}(X) = E[X^2] - (E[X])^2 = \frac{a^2+ab+b^2}{3} - \frac{(a+b)^2}{4} = \frac{(b-a)^2}{12}
    $$

---

**Exercise 2.**
Show that if $U \sim \text{Uniform}(0,1)$, then $X = a + (b-a)U \sim \text{Uniform}(a,b)$.

??? success "Solution to Exercise 2"
    The CDF of $X$ for $a \le x \le b$:

    $$
    P(X \le x) = P(a + (b-a)U \le x) = P\!\left(U \le \frac{x-a}{b-a}\right) = \frac{x-a}{b-a}
    $$

    This is the CDF of $\text{Uniform}(a, b)$. $\square$

---

**Exercise 3.**
The uniform distribution is the maximum-entropy distribution on $[a, b]$. State what this means and explain intuitively why it makes sense.

??? success "Solution to Exercise 3"
    Among all continuous distributions supported on $[a, b]$, the uniform distribution has the highest differential entropy $h(X) = \ln(b - a)$. Maximum entropy means maximum uncertainty: we know the range but nothing else about where values are likely to fall. Intuitively, any non-uniform density would concentrate probability in some subregion, implying more knowledge about the distribution than just its support.

---

**Exercise 4.**
If $X \sim \text{Uniform}(0, 1)$, find the distribution of $Y = -\ln(X)$ and identify it.

??? success "Solution to Exercise 4"
    For $y > 0$:

    $$
    P(Y \le y) = P(-\ln X \le y) = P(X \ge e^{-y}) = 1 - e^{-y}
    $$

    This is the CDF of $\text{Exponential}(\lambda = 1)$. So $Y = -\ln(U) \sim \text{Exp}(1)$, which is the basis of inverse transform sampling for the exponential distribution.
