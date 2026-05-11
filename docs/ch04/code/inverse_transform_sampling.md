# Inverse Transform Sampling

## Overview

**Inverse transform sampling** is a method for generating random samples from any distribution whose inverse CDF (quantile function) is available. The key result is:

If $U \sim \text{Uniform}(0, 1)$ and $F$ is a CDF with inverse $F^{-1}$, then:

$$
X = F^{-1}(U) \sim F
$$

This single idea underlies much of computational statistics and Monte Carlo simulation.

---

## Proof

For any $x$:

$$
P(X \le x) = P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)
$$

The last equality uses the fact that $U \sim \text{Uniform}(0,1)$, so $P(U \le p) = p$. $\square$

---

## Example 1: Exponential Distribution

The exponential CDF is $F(x) = 1 - e^{-\lambda x}$. Inverting:

$$
F^{-1}(u) = -\frac{\ln(1 - u)}{\lambda}
$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
n = 10_000
lam = 1.0

u = np.random.uniform(0, 1, n)
x_exp = -np.log(1 - u) / lam

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(u, bins=50, density=True, color="lightgray", edgecolor="black")
axes[0].set_title("Step 1: U ~ Uniform(0,1)")

t = np.linspace(0, np.max(x_exp), 300)
axes[1].hist(x_exp, bins=50, density=True, color="steelblue",
             edgecolor="white", alpha=0.7, label="Transformed")
axes[1].plot(t, stats.expon.pdf(t, scale=1/lam), "r-", lw=2.5, label="Exp PDF")
axes[1].set_title("Step 2: X = -ln(1-U)/λ")
axes[1].legend()

u_grid = np.linspace(0.001, 0.999, 300)
axes[2].plot(u_grid, -np.log(1 - u_grid) / lam, "b-", lw=2)
axes[2].set_title("Inverse CDF: F⁻¹(u)")
axes[2].set_xlabel("u")
axes[2].set_ylabel("x")

plt.tight_layout()
plt.show()
```

---

## Example 2: Cauchy Distribution

The standard Cauchy CDF is $F(x) = \frac{1}{2} + \frac{1}{\pi}\arctan(x)$. Inverting:

$$
F^{-1}(u) = \tan\!\left(\pi\!\left(u - \frac{1}{2}\right)\right)
$$

```python
u = np.random.uniform(0, 1, n)
x_cauchy = np.tan(np.pi * (u - 0.5))

fig, ax = plt.subplots(figsize=(10, 4))
x_clipped = np.clip(x_cauchy, -20, 20)
ax.hist(x_clipped, bins=80, density=True, color="coral",
        edgecolor="white", alpha=0.7, label="Transformed")
t2 = np.linspace(-20, 20, 500)
ax.plot(t2, stats.cauchy.pdf(t2), "k-", lw=2.5, label="Cauchy PDF")
ax.set_title("Cauchy via Inverse Transform")
ax.set_xlim(-20, 20)
ax.legend()
plt.tight_layout()
plt.show()
```

The Cauchy example illustrates that inverse transform sampling works even for heavy-tailed distributions with no finite mean.

---

## When to Use

| Situation | Recommendation |
|---|---|
| Closed-form $F^{-1}$ available | Inverse transform (fast, exact) |
| $F^{-1}$ expensive to compute | Consider rejection sampling or MCMC |
| Discrete distribution | Use cumulative PMF thresholds |
| Multivariate distribution | Use conditional decomposition or specialized algorithms |

---

## Exercises

**Exercise 1.**
Derive the inverse CDF for the $\text{Uniform}(a, b)$ distribution and write the inverse transform formula.

??? success "Solution to Exercise 1"
    The CDF is $F(x) = (x - a)/(b - a)$. Setting $u = F(x)$ and solving:

    $$
    x = a + (b - a)u = F^{-1}(u)
    $$

    So if $U \sim \text{Uniform}(0,1)$, then $X = a + (b-a)U \sim \text{Uniform}(a, b)$.

---

**Exercise 2.**
Use inverse transform sampling to generate samples from a $\text{Bernoulli}(p)$ distribution. Explain how the same idea extends to any discrete distribution.

??? success "Solution to Exercise 2"
    Generate $U \sim \text{Uniform}(0,1)$. Set $X = 1$ if $U \le p$, else $X = 0$.

    For a general discrete distribution with values $x_1, x_2, \ldots$ and probabilities $p_1, p_2, \ldots$: compute the cumulative probabilities $c_k = \sum_{i=1}^k p_i$. Set $X = x_k$ where $k$ is the smallest index such that $U \le c_k$. This partitions $[0,1]$ into intervals of length $p_k$, each mapped to $x_k$.

---

**Exercise 3.**
Prove the inverse transform result: if $U \sim \text{Uniform}(0,1)$ and $F$ is a continuous, strictly increasing CDF, then $F^{-1}(U) \sim F$.

??? success "Solution to Exercise 3"
    For any $x \in \mathbb{R}$:

    $$
    P(F^{-1}(U) \le x) = P(U \le F(x))
    $$

    The second equality uses the fact that $F$ is strictly increasing, so $F^{-1}(u) \le x \iff u \le F(x)$. Since $U \sim \text{Uniform}(0,1)$:

    $$
    P(U \le F(x)) = F(x)
    $$

    Therefore $P(F^{-1}(U) \le x) = F(x)$, which means $F^{-1}(U)$ has CDF $F$. $\square$

---

**Exercise 4.**
The Rayleigh distribution has CDF $F(x) = 1 - e^{-x^2/(2\sigma^2)}$ for $x \ge 0$. Derive the inverse CDF and write code to generate Rayleigh samples via inverse transform sampling.

??? success "Solution to Exercise 4"
    Invert $u = 1 - e^{-x^2/(2\sigma^2)}$:

    $$
    e^{-x^2/(2\sigma^2)} = 1 - u \implies x = \sigma\sqrt{-2\ln(1-u)}
    $$

    Code:

    ```python
    sigma = 1.0
    u = np.random.uniform(0, 1, 10000)
    x = sigma * np.sqrt(-2 * np.log(1 - u))
    ```

    This is closely related to one component of the Box-Muller transform: $R = \sqrt{-2\ln U}$ has a Rayleigh distribution.

---

**Exercise 5.**
Explain why $1 - U$ can replace $U$ in all inverse transform formulas without changing the distribution of the output. Why is this useful in practice?

??? success "Solution to Exercise 5"
    If $U \sim \text{Uniform}(0,1)$, then $1 - U \sim \text{Uniform}(0,1)$ as well (the uniform is symmetric about 0.5). Therefore replacing $U$ with $1 - U$ in $F^{-1}(U)$ produces the same distribution.

    This is useful because it simplifies formulas. For the exponential, $-\ln(1-U)/\lambda$ can be replaced by $-\ln(U)/\lambda$, avoiding one subtraction. In practice, this also avoids the edge case $U = 0$ (which gives $\ln(0) = -\infty$) since $1 - U = 0$ has probability zero.
