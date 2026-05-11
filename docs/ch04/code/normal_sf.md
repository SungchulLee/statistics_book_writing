# Normal Survival Function

## Overview

The **survival function** (SF) is the complement of the CDF:

$$
S(x) = P(X > x) = 1 - F(x)
$$

It gives the probability that $X$ exceeds a given threshold. The survival function is used extensively in reliability engineering, actuarial science, and clinical trials where the quantity of interest is time-to-event or exceedance probability.

---

## Code

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu, sigma = 0, 1
dist = stats.norm(loc=mu, scale=sigma)

x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 400)
cdf = dist.cdf(x)
sf = dist.sf(x)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, cdf, lw=2, label='CDF  P(X ≤ x)')
ax.plot(x, sf, lw=2, label='SF   P(X > x)')
ax.axvline(0, ls=':', color='gray', alpha=0.6)
ax.axhline(0.5, ls=':', color='gray', alpha=0.6)
ax.annotate("CDF + SF = 1", xy=(1.2, 0.5), fontsize=12,
            bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='gray'))
ax.set_xlabel('x')
ax.set_ylabel('Probability')
ax.set_ylim(-0.03, 1.03)
ax.legend(loc='center left', frameon=False)
ax.set_title(f"Normal({mu}, {sigma}) — CDF vs Survival Function")
ax.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
```

---

## Why Use the Survival Function

For extreme upper-tail probabilities, computing $1 - F(x)$ directly can suffer from floating-point cancellation when $F(x)$ is very close to 1. The dedicated `sf()` method avoids this by computing the tail probability directly.

```python
# Poor: floating-point cancellation
p_bad = 1 - stats.norm.cdf(8)    # may give 0.0

# Good: numerically stable
p_good = stats.norm.sf(8)         # gives ~6.22e-16
```

---

## Exercises

**Exercise 1.**
For $Z \sim N(0,1)$, compute $P(Z > 2)$ using both the CDF and the SF. Verify they agree.

??? success "Solution to Exercise 1"
    Via CDF: $P(Z > 2) = 1 - \mathcal{N}(2) = 1 - 0.9772 = 0.0228$.

    Via SF: `stats.norm.sf(2) = 0.0228`.

    Both give the same result. For extreme values like $P(Z > 8)$, the SF method is numerically preferable.

---

**Exercise 2.**
Show that $S(x) + F(x) = 1$ for all $x$ and any continuous random variable.

??? success "Solution to Exercise 2"
    By definition:

    $$
    F(x) + S(x) = P(X \le x) + P(X > x) = P(X \in (-\infty, x]) + P(X \in (x, \infty))
    $$

    Since $(-\infty, x]$ and $(x, \infty)$ are a partition of $\mathbb{R}$, their probabilities sum to 1. $\square$

---

**Exercise 3.**
A component fails when stress $X \sim N(500, 2500)$ exceeds a threshold of 600. What is the probability of failure?

??? success "Solution to Exercise 3"
    Standardize: $Z = (600 - 500)/50 = 2$.

    $$
    P(X > 600) = P(Z > 2) = S(2) \approx 0.0228
    $$

    About 2.3% of components will fail.

---

**Exercise 4.**
The **hazard function** is defined as $h(x) = f(x)/S(x)$. For the standard normal, compute $h(0)$ and explain why $h(x)$ is increasing for $x > 0$.

??? success "Solution to Exercise 4"
    At $x = 0$: $f(0) = 1/\sqrt{2\pi} \approx 0.3989$ and $S(0) = 0.5$.

    $$
    h(0) = \frac{0.3989}{0.5} \approx 0.7979
    $$

    For $x > 0$, $S(x)$ decreases faster than $f(x)$ because the denominator shrinks (fewer values remain above $x$) while the numerator (the density) also decreases but more slowly in relative terms. This makes $h(x)$ increasing — the conditional probability of "failure" at $x$ given survival past $x$ grows with $x$. The normal distribution has an **increasing failure rate**.
