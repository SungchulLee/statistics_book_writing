# Moment Generating Functions

## Overview

The **moment generating function (MGF)** is a powerful tool that encodes all the moments of a random variable into a single function. It provides an elegant way to compute expectations, prove limit theorems, and characterize distributions. If two random variables have the same MGF, they have the same distribution.

---

## Definition

The **moment generating function** of a random variable $X$ is defined as:

$$
M_X(t) = E\left[e^{tX}\right] =
\begin{cases}
\displaystyle\sum_x e^{tx} \cdot P(X = x), & \text{discrete} \\[10pt]
\displaystyle\int_{-\infty}^{\infty} e^{tx} f(x) \, dx, & \text{continuous}
\end{cases}
$$

The MGF exists if $M_X(t)$ is finite for all $t$ in some open interval containing 0.

---

## Why "Moment Generating"?

The Taylor expansion of $e^{tX}$ reveals the connection to moments:

$$
M_X(t) = E\left[e^{tX}\right] = E\left[\sum_{k=0}^{\infty} \frac{(tX)^k}{k!}\right] = \sum_{k=0}^{\infty} \frac{t^k}{k!} E[X^k]
$$

Taking derivatives and evaluating at $t = 0$ extracts individual moments:

$$
M_X^{(n)}(0) = \frac{d^n}{dt^n} M_X(t) \bigg|_{t=0} = E[X^n]
$$

Specifically:

$$
\begin{aligned}
M_X'(0) &= E[X] \\
M_X''(0) &= E[X^2] \\
\text{Var}(X) &= M_X''(0) - \left[M_X'(0)\right]^2
\end{aligned}
$$

---

## Key Properties

### Uniqueness

If two random variables $X$ and $Y$ have MGFs that exist and are equal in an open interval around 0:

$$
M_X(t) = M_Y(t) \quad \Longrightarrow \quad X \stackrel{d}{=} Y
$$

This makes the MGF a tool for **identifying distributions**.

### Linear Transformation

For $Y = aX + b$:

$$
M_Y(t) = e^{bt} M_X(at)
$$

### Sum of Independent Variables

If $X \perp\!\!\!\perp Y$:

$$
M_{X+Y}(t) = M_X(t) \cdot M_Y(t)
$$

This extends to $n$ independent variables: $M_{S_n}(t) = \prod_{i=1}^n M_{X_i}(t)$.

---

## Common MGFs

| Distribution | $M_X(t)$ | Parameters |
|:---|:---|:---|
| Bernoulli$(p)$ | $1 - p + pe^t$ | $p \in (0,1)$ |
| Binomial$(n, p)$ | $(1 - p + pe^t)^n$ | $n \in \mathbb{N},\ p \in (0,1)$ |
| Poisson$(\lambda)$ | $\exp\left(\lambda(e^t - 1)\right)$ | $\lambda > 0$ |
| Geometric$(p)$ | $\dfrac{pe^t}{1 - (1-p)e^t}$, $t < -\ln(1-p)$ | $p \in (0,1)$ |
| Exponential$(\lambda)$ | $\dfrac{\lambda}{\lambda - t}$, $t < \lambda$ | $\lambda > 0$ |
| Normal$(\mu, \sigma^2)$ | $\exp\left(\mu t + \frac{\sigma^2 t^2}{2}\right)$ | $\mu \in \mathbb{R},\ \sigma^2 > 0$ |

---

## Examples

### Example: MGF of the Normal Distribution

For $X \sim N(\mu, \sigma^2)$, the MGF is:

$$
M_X(t) = \exp\left(\mu t + \frac{\sigma^2 t^2}{2}\right)
$$

Extracting moments:

$$
\begin{aligned}
M_X'(t) &= \left(\mu + \sigma^2 t\right) M_X(t) \\
M_X'(0) &= \mu = E[X] \\[6pt]
M_X''(t) &= \left(\sigma^2 + (\mu + \sigma^2 t)^2\right) M_X(t) \\
M_X''(0) &= \sigma^2 + \mu^2 = E[X^2] \\[6pt]
\text{Var}(X) &= (\sigma^2 + \mu^2) - \mu^2 = \sigma^2
\end{aligned}
$$

### Example: Sum of Independent Normals

If $X_1 \sim N(\mu_1, \sigma_1^2)$ and $X_2 \sim N(\mu_2, \sigma_2^2)$ are independent:

$$
M_{X_1 + X_2}(t) = \exp\left((\mu_1 + \mu_2)t + \frac{(\sigma_1^2 + \sigma_2^2)t^2}{2}\right)
$$

By uniqueness, $X_1 + X_2 \sim N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$.

### Example: Proving the CLT (Sketch)

For i.i.d. $X_i$ with mean $\mu$, variance $\sigma^2$, let $Z_n = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}}$. The MGF of $Z_n$ satisfies:

$$
M_{Z_n}(t) = \left[M_{\frac{X_i - \mu}{\sigma}}\left(\frac{t}{\sqrt{n}}\right)\right]^n \to e^{t^2/2} \quad \text{as } n \to \infty
$$

The limit $e^{t^2/2}$ is the MGF of $N(0,1)$, proving convergence in distribution.

---

## Python Exploration

```python
import numpy as np
from scipy.misc import derivative

def mgf_normal(t, mu, sigma2):
    """MGF of Normal(mu, sigma2)."""
    return np.exp(mu * t + sigma2 * t**2 / 2)

# Extract moments via numerical differentiation
mu, sigma2 = 3.0, 4.0

E_X = derivative(lambda t: mgf_normal(t, mu, sigma2), 0, n=1, dx=1e-6)
E_X2 = derivative(lambda t: mgf_normal(t, mu, sigma2), 0, n=2, dx=1e-6)
Var_X = E_X2 - E_X**2

print(f"E[X] = {E_X:.4f} (theoretical: {mu})")
print(f"E[X²] = {E_X2:.4f} (theoretical: {sigma2 + mu**2})")
print(f"Var(X) = {Var_X:.4f} (theoretical: {sigma2})")
```

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_mgf_comparison():
    """Plot MGFs of several distributions."""
    t = np.linspace(-1.5, 1.5, 300)

    fig, ax = plt.subplots(figsize=(12, 4))

    # Normal(0, 1)
    ax.plot(t, np.exp(t**2 / 2), label='N(0, 1)', lw=2)

    # Exponential(1)
    t_exp = t[t < 1]
    ax.plot(t_exp, 1 / (1 - t_exp), label='Exp(1)', lw=2)

    # Poisson(3)
    lam = 3
    ax.plot(t, np.exp(lam * (np.exp(t) - 1)), label='Poisson(3)', lw=2)

    # Bernoulli(0.5)
    p = 0.5
    ax.plot(t, 1 - p + p * np.exp(t), label='Bernoulli(0.5)', lw=2)

    ax.set_xlabel('t')
    ax.set_ylabel('M_X(t)')
    ax.set_title('Moment Generating Functions')
    ax.set_ylim(0, 15)
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

plot_mgf_comparison()
```

```python
import numpy as np

def verify_sum_of_normals(n_simulations=100_000):
    """Verify that sum of independent normals is normal via simulation."""
    np.random.seed(42)
    mu1, sigma1 = 2, 3
    mu2, sigma2 = 5, 4

    X1 = np.random.normal(mu1, sigma1, n_simulations)
    X2 = np.random.normal(mu2, sigma2, n_simulations)
    S = X1 + X2

    print(f"E[X1+X2] = {S.mean():.4f} (theoretical: {mu1 + mu2})")
    print(f"Var(X1+X2) = {S.var():.4f} (theoretical: {sigma1**2 + sigma2**2})")

verify_sum_of_normals()
```

---

## Key Takeaways

- The MGF $M_X(t) = E[e^{tX}]$ encodes all moments: the $n$-th derivative at 0 gives $E[X^n]$.
- If two distributions have the same MGF (in a neighborhood of 0), they are identical.
- For independent variables, the MGF of the sum is the product of individual MGFs.
- MGFs provide elegant proofs for results like the distribution of sums of normals and the Central Limit Theorem.
