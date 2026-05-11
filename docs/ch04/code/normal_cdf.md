# Normal Cumulative Distribution and Quantiles

## Overview

The **cumulative distribution function** (CDF) of a random variable $X$ gives the probability that $X$ takes a value less than or equal to $x$:

$$
F(x) = P(X \le x) = \int_{-\infty}^{x} f(t)\,dt
$$

For the normal distribution, this integral has no closed-form expression and must be evaluated numerically. SciPy provides `stats.norm.cdf()` for this purpose.

---

## CDF and PDF Together

Plotting the CDF and PDF on the same figure (using dual y-axes) makes their relationship explicit: the CDF at any point equals the area under the PDF to the left of that point.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu, sigma = 1, 2
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 400)

dist = stats.norm(loc=mu, scale=sigma)
y_cdf = dist.cdf(x)
y_pdf = dist.pdf(x)

fig, ax_cdf = plt.subplots(figsize=(12, 3))

# CDF on left axis
ax_cdf.plot(x, y_cdf, lw=2, label="CDF P(X ≤ x)")
ax_cdf.set_xlabel("x")
ax_cdf.set_ylabel("P(X ≤ x)")
ax_cdf.set_ylim(-0.02, 1.02)

# Reference points
for xv in [mu - sigma, mu, mu + sigma]:
    yv = dist.cdf(xv)
    ax_cdf.axvline(xv, linestyle='--', color='gray', alpha=0.7)
    ax_cdf.text(xv, yv + 0.05, f"P(X≤{xv:.0f})={yv:.3f}",
                ha='center', fontsize=9)

# PDF on right axis
ax_pdf = ax_cdf.twinx()
ax_pdf.plot(x, y_pdf, lw=2, color='tab:red', label="PDF (density)")
ax_pdf.set_ylabel("Density", color='tab:red')

ax_cdf.set_title(f"Normal({mu}, {sigma}) — CDF with PDF Overlay")
plt.tight_layout()
plt.show()
```

---

## Key CDF Values for the Standard Normal

| $x$ | $\mathcal{N}(x) = P(Z \le x)$ |
|---|---|
| $-1.96$ | $0.025$ |
| $-1$ | $0.159$ |
| $0$ | $0.500$ |
| $1$ | $0.841$ |
| $1.96$ | $0.975$ |

The symmetry property $\mathcal{N}(-x) = 1 - \mathcal{N}(x)$ means only one half of the table is needed.

---

## Exercises

**Exercise 1.**
For $X \sim N(0, 1)$, compute $P(-1.96 \le X \le 1.96)$ using the CDF.

??? success "Solution to Exercise 1"
    $$
    P(-1.96 \le X \le 1.96) = \mathcal{N}(1.96) - \mathcal{N}(-1.96) = 0.975 - 0.025 = 0.950
    $$

    This is the basis for the 95% confidence interval: the central 95% of the standard normal lies between $\pm 1.96$.

---

**Exercise 2.**
Prove the symmetry property $\mathcal{N}(-x) = 1 - \mathcal{N}(x)$ for the standard normal CDF.

??? success "Solution to Exercise 2"
    The standard normal PDF satisfies $\varphi(-t) = \varphi(t)$ (symmetry about 0). Then:

    $$
    \mathcal{N}(-x) = \int_{-\infty}^{-x} \varphi(t)\,dt
    $$

    Substituting $u = -t$ (so $du = -dt$):

    $$
    \mathcal{N}(-x) = \int_{\infty}^{x} \varphi(-u)(-du) = \int_x^{\infty} \varphi(u)\,du = 1 - \mathcal{N}(x)
    $$

    $\square$

---

**Exercise 3.**
If $X \sim N(5, 9)$, find $P(X > 8)$ by standardizing.

??? success "Solution to Exercise 3"
    Standardize: $Z = (X - 5)/3$. Then:

    $$
    P(X > 8) = P\!\left(Z > \frac{8-5}{3}\right) = P(Z > 1) = 1 - \mathcal{N}(1) \approx 1 - 0.8413 = 0.1587
    $$

---

**Exercise 4.**
Show that $F'(x) = f(x)$ (the derivative of the CDF is the PDF) using the Fundamental Theorem of Calculus. Explain what this means graphically.

??? success "Solution to Exercise 4"
    By definition, $F(x) = \int_{-\infty}^x f(t)\,dt$. By the Fundamental Theorem of Calculus:

    $$
    F'(x) = \frac{d}{dx}\int_{-\infty}^x f(t)\,dt = f(x)
    $$

    Graphically: the slope of the CDF at any point $x$ equals the height of the PDF at that point. Where the PDF is highest (at the mode), the CDF is steepest. Where the PDF is near zero (in the tails), the CDF is nearly flat.
