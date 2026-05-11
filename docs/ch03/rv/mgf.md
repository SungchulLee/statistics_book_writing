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

## Exercises

**Exercise 1.**
$X \sim \mathrm{Exp}(\lambda)$. (a) Derive $M_X(t)$ and its domain. (b) Compute $\mathbb{E}[X], \mathbb{E}[X^2]$. (c) Verify $\mathrm{Var}(X) = 1/\lambda^2$.

??? success "Solution to Exercise 1"
    (a) $M_X(t) = \int_0^\infty e^{tx} \lambda e^{-\lambda x} dx = \lambda \int_0^\infty e^{-(\lambda - t) x} dx = \lambda/(\lambda - t)$ for $t < \lambda$.

    (b) $M_X'(t) = \lambda/(\lambda - t)^2$. $M_X'(0) = 1/\lambda = \mathbb{E}[X]$. $M_X''(t) = 2\lambda/(\lambda - t)^3$. $M_X''(0) = 2/\lambda^2 = \mathbb{E}[X^2]$.

    (c) $\mathrm{Var}(X) = 2/\lambda^2 - (1/\lambda)^2 = 1/\lambda^2$. Mean and SD both equal $1/\lambda$.

---

**Exercise 2.**
Prove $M_{aX + b}(t) = e^{bt} M_X(at)$ and $M_{X + Y}(t) = M_X(t) M_Y(t)$ when $X$ and $Y$ are independent.

??? success "Solution to Exercise 2"
    **Linear transformation:**

    $$
    M_{aX + b}(t) = \mathbb{E}[e^{t(aX + b)}] = e^{bt} \mathbb{E}[e^{(at)X}] = e^{bt} M_X(at)
    $$

    **Sum of independent:** for independent $X$ and $Y$, $e^{tX}$ and $e^{tY}$ are independent (functions of independent variables), so

    $$
    M_{X + Y}(t) = \mathbb{E}[e^{t(X + Y)}] = \mathbb{E}[e^{tX} \cdot e^{tY}] = \mathbb{E}[e^{tX}] \mathbb{E}[e^{tY}] = M_X(t) M_Y(t)
    $$

    The second formula generalizes: $M_{S_n}(t) = \prod_i M_{X_i}(t)$ for $S_n = X_1 + \cdots + X_n$ with mutual independence.

    These two properties are why MGFs are so useful for analyzing transformations and sums of random variables.

---

**Exercise 3.**
Use the MGF approach to derive the **distribution of a sum of independent Poisson random variables**: if $X_i \sim \mathrm{Poisson}(\lambda_i)$ independently, what is the distribution of $\sum_i X_i$?

??? success "Solution to Exercise 3"
    Poisson MGF: $M_X(t) = e^{\lambda (e^t - 1)}$.

    Sum: $M_{\sum X_i}(t) = \prod_i M_{X_i}(t) = \prod_i e^{\lambda_i (e^t - 1)} = e^{(\sum \lambda_i)(e^t - 1)}$.

    This is the MGF of $\mathrm{Poisson}(\sum \lambda_i)$. By uniqueness, $\sum X_i \sim \mathrm{Poisson}(\sum \lambda_i)$.

    **Implication:** Poisson is "closed under addition" — sums of independent Poissons are Poisson with sum of rates. This is the foundation of count-data models: total event counts from independent sources are Poisson with combined rate, allowing modular analysis.

---

**Exercise 4.**
**Skewness and kurtosis from MGF.** Show that the third and fourth standardized cumulants (skewness and excess kurtosis) come from derivatives of $\ln M_X(t)$, not $M_X(t)$ itself.

??? success "Solution to Exercise 4"
    The **cumulant generating function** is $K_X(t) = \ln M_X(t)$. Its Taylor expansion is

    $$
    K_X(t) = \kappa_1 t + \kappa_2 \frac{t^2}{2} + \kappa_3 \frac{t^3}{6} + \kappa_4 \frac{t^4}{24} + \cdots
    $$

    where $\kappa_n$ is the $n$-th cumulant. The first few:

    - $\kappa_1 = \mathbb{E}[X] = \mu$ (mean)
    - $\kappa_2 = \mathrm{Var}(X) = \sigma^2$
    - $\kappa_3 = \mathbb{E}[(X - \mu)^3]$ (third central moment)
    - $\kappa_4 = \mathbb{E}[(X - \mu)^4] - 3\sigma^4$ (fourth central moment minus $3\sigma^4$)

    Then **skewness** $= \kappa_3/\kappa_2^{3/2}$ and **excess kurtosis** $= \kappa_4/\kappa_2^2$. Both come from cumulants, which are derivatives of $K_X(t)$, not $M_X(t)$.

    **Why cumulants are nicer than moments:** cumulants of independent sums add: $\kappa_n(X + Y) = \kappa_n(X) + \kappa_n(Y)$ for independent $X, Y$. Moments do not. For example, $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$ (which is $\kappa_2$ addition), and similarly for higher cumulants. This makes cumulants the natural language for the CLT and Edgeworth expansions.

---

**Exercise 5.**
**MGF non-existence.** Show that the Cauchy distribution does not have an MGF. What is the **characteristic function** alternative, and why does it always exist?

??? success "Solution to Exercise 5"
    Cauchy density: $f(x) = 1/(\pi(1 + x^2))$.

    $M(t) = \int_{-\infty}^\infty e^{tx} / (\pi (1 + x^2)) dx$. For any $t \ne 0$, $e^{tx}$ grows exponentially in one direction and the integrand is not integrable. The MGF is undefined except at $t = 0$ (where it trivially equals 1).

    **Characteristic function (CF):** $\phi_X(t) = \mathbb{E}[e^{itX}]$. Always exists because $|e^{itX}| = 1$, so $\mathbb{E}[|e^{itX}|] = 1 < \infty$.

    For the Cauchy: $\phi(t) = e^{-|t|}$ — a clean closed form despite the MGF not existing.

    **Practical implication:** when working with heavy-tailed distributions, switch from MGFs to characteristic functions. Most theoretical results (Lévy continuity theorem, inversion formulas) are stated in terms of characteristic functions because they cover the broader class. MGFs remain useful for distributions with all moments — the "nice" majority of applied statistics.

---

**Exercise 6.**
**The MGF determines the distribution** (uniqueness theorem) — but only when it exists in a neighborhood of zero. Construct two distinct distributions whose moments coincide for all orders. (This is the **moment problem** failure.)

??? success "Solution to Exercise 6"
    Consider the **lognormal** distribution with PDF $f_1(x) = \frac{1}{x\sqrt{2\pi}} e^{-(\ln x)^2/2}$ for $x > 0$, and a perturbed version

    $$
    f_2(x) = f_1(x) \cdot (1 + \sin(2\pi \ln x))
    $$

    Both are valid PDFs (with $f_2$ chosen to remain non-negative for the perturbation amplitude).

    Moments: $\mathbb{E}_2[X^k] = \int_0^\infty x^k f_1(x)(1 + \sin(2\pi \ln x)) dx = \mathbb{E}_1[X^k] + \int x^k f_1(x) \sin(2\pi \ln x) dx$.

    Substitute $u = \ln x$: the second integral becomes $\int e^{(k+1)u - u^2/2}\sin(2\pi u)/\sqrt{2\pi}\, du$. This integral equals zero for *every* integer $k$ — the integrand combines a Gaussian factor with a sinusoidal modulator that integrates to zero. So all moments agree, but $f_1 \ne f_2$.

    **Why this works:** the lognormal's MGF does *not* converge in any neighborhood of 0 (the integral $\int e^{tx} f_1(x) dx$ diverges for $t > 0$). So the MGF doesn't exist, and the moment sequence does not determine the distribution.

    **Lesson:** when the MGF exists in an open interval, moments uniquely determine the distribution. When the MGF doesn't exist (heavy tails, lognormal), moments may not suffice — multiple distributions can share all moments. This is the **Stieltjes moment problem**: the moments uniquely determine the distribution iff a Carleman-type condition holds.
