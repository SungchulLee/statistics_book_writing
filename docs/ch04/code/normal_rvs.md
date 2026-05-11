# Normal Random Variates

## Overview

Drawing **random variates** (random samples) from a distribution is fundamental to simulation-based statistics. For the normal distribution, `stats.norm.rvs()` generates samples from $N(\mu, \sigma^2)$. As the sample size grows, the histogram of the samples converges to the theoretical PDF—a direct illustration of the Law of Large Numbers.

---

## Generating Samples

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu = 1
sigma = 2

# Theoretical PDF
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 200)
y = stats.norm(loc=mu, scale=sigma).pdf(x)

# Random samples
n_samples = 10_000
samples = stats.norm(loc=mu, scale=sigma).rvs(size=n_samples)

# Plot
fig, ax = plt.subplots(figsize=(12, 3))
ax.hist(samples, bins=50, density=True, alpha=0.3, color='C0',
        label='Random Samples (rvs)')
ax.plot(x, y, 'r-', lw=2, label='Theoretical PDF')
ax.set_title(f"Normal(μ={mu}, σ={sigma}) — PDF vs. Random Samples")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, linestyle=':')
plt.show()
```

The key option is `density=True` in `hist()`, which normalizes the histogram so its total area equals 1, making it directly comparable to the PDF curve.

---

## Why Histograms Approximate the PDF

For a bin of width $\Delta x$ centered at $x_0$, the expected fraction of samples falling in that bin is approximately $f(x_0)\,\Delta x$, where $f$ is the PDF. With `density=True`, the histogram height at $x_0$ estimates $f(x_0)$. By the Law of Large Numbers, this estimate converges to the true density as $n \to \infty$.

---

## Exercises

**Exercise 1.**
Generate 100, 1000, and 10000 samples from $N(0, 1)$ and overlay their histograms on the same plot. Describe how the fit improves with sample size.

??? success "Solution to Exercise 1"
    ```python
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    x = np.linspace(-4, 4, 200)
    pdf = stats.norm.pdf(x)
    for ax, n in zip(axes, [100, 1000, 10000]):
        samples = stats.norm.rvs(size=n)
        ax.hist(samples, bins=30, density=True, alpha=0.4)
        ax.plot(x, pdf, 'r-', lw=2)
        ax.set_title(f"n = {n}")
    ```

    With $n = 100$, the histogram is jagged and deviates noticeably. At $n = 1000$, the shape is recognizably bell-curved but still has some bumps. At $n = 10000$, the histogram closely traces the PDF. The convergence rate is approximately $O(1/\sqrt{n})$.

---

**Exercise 2.**
If $X \sim N(\mu, \sigma^2)$ and you draw $n$ samples $X_1, \ldots, X_n$, what are $E[\bar{X}]$ and $\text{Var}(\bar{X})$? Verify numerically by generating 10000 sample means of size $n = 50$.

??? success "Solution to Exercise 2"
    $E[\bar{X}] = \mu$ and $\text{Var}(\bar{X}) = \sigma^2/n$.

    ```python
    mu, sigma, n = 5, 3, 50
    means = [stats.norm(mu, sigma).rvs(n).mean() for _ in range(10000)]
    print(f"E[X_bar] ≈ {np.mean(means):.4f}  (theory: {mu})")
    print(f"Var(X_bar) ≈ {np.var(means):.4f}  (theory: {sigma**2/n:.4f})")
    ```

---

**Exercise 3.**
Explain the difference between `stats.norm.rvs(size=n)` and `np.random.normal(0, 1, n)`. When would you prefer one over the other?

??? success "Solution to Exercise 3"
    Both generate standard normal random variates. `np.random.normal` calls NumPy's random number generator directly—it is slightly faster for simple cases. `stats.norm.rvs` uses the frozen distribution interface, which is more flexible: you can create a distribution object once and call `.rvs()`, `.pdf()`, `.cdf()` etc. on the same object. Use the SciPy interface when you need multiple methods on the same distribution; use NumPy's direct call for maximum speed in tight loops.

---

**Exercise 4.**
Using the Box-Muller transform, generate standard normal samples from uniform random variables. The transform is: given $U_1, U_2 \sim \text{Uniform}(0,1)$ independently,

$$
Z_1 = \sqrt{-2\ln U_1}\,\cos(2\pi U_2), \qquad Z_2 = \sqrt{-2\ln U_1}\,\sin(2\pi U_2)
$$

Verify that $Z_1$ and $Z_2$ are approximately $N(0,1)$.

??? success "Solution to Exercise 4"
    ```python
    n = 10000
    U1 = np.random.uniform(0, 1, n)
    U2 = np.random.uniform(0, 1, n)
    Z1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    Z2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
    ```

    Check: `Z1.mean() ≈ 0`, `Z1.std() ≈ 1`, and a histogram of `Z1` matches the standard normal PDF. The same holds for `Z2`. The two outputs are independent standard normals by construction, which can be verified by checking that their sample correlation is near zero.
