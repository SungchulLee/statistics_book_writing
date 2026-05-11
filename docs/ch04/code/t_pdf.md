# Student-t Density Function

## Overview

The **Student's $t$-distribution** arises naturally when estimating the mean of a normally distributed population with unknown variance. It has heavier tails than the normal, making it more robust to outliers and more appropriate for small-sample inference.

The PDF with $\nu$ degrees of freedom, location $\mu$, and scale $\sigma$ is:

$$
f(x) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sigma\sqrt{\nu\pi}\;\Gamma\!\left(\frac{\nu}{2}\right)} \left(1 + \frac{1}{\nu}\left(\frac{x-\mu}{\sigma}\right)^2\right)^{-(\nu+1)/2}
$$

---

## Moments

| Property | Condition | Value |
|---|---|---|
| Mean | $\nu > 1$ | $\mu$ |
| Variance | $\nu > 2$ | $\dfrac{\nu}{\nu - 2}\,\sigma^2$ |
| Variance | $1 < \nu \le 2$ | $\infty$ |
| Mean | $\nu \le 1$ | undefined |

As $\nu \to \infty$, the $t$-distribution converges to $N(\mu, \sigma^2)$.

---

## Code: Comparing t and Normal

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

nu = 5       # degrees of freedom
mu = 0
sigma = 1

t_dist = stats.t(df=nu, loc=mu, scale=sigma)
n_dist = stats.norm(loc=mu, scale=sigma)

# Quantile-based x-range for robust plotting
x = np.linspace(t_dist.ppf(1e-4), t_dist.ppf(1 - 1e-4), 600)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, t_dist.pdf(x), lw=2, label=f"t PDF (ν={nu})")
ax.plot(x, n_dist.pdf(x), lw=1.8, linestyle='--', label="Normal PDF")
ax.set_title("Student's t Distribution — PDF")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.legend()
ax.grid(True, linestyle=":")
plt.tight_layout()
plt.show()
```

The plot reveals that the $t$-distribution has more probability in the tails and less at the center than the normal, with the difference becoming more pronounced as $\nu$ decreases.

---

## Exercises

**Exercise 1.**
For the $t_5$ distribution, compute the variance using the formula $\nu/(\nu-2)$. How much larger is it than the standard normal variance?

??? success "Solution to Exercise 1"
    $$
    \text{Var}(T) = \frac{5}{5 - 2} = \frac{5}{3} \approx 1.667
    $$

    This is 67% larger than the standard normal variance of 1. The extra variance comes entirely from the heavier tails.

---

**Exercise 2.**
Explain why the $t$-distribution is used instead of the normal when constructing confidence intervals for the mean with unknown $\sigma$. What changes as $n$ grows?

??? success "Solution to Exercise 2"
    When $\sigma$ is unknown, we replace it with the sample standard deviation $s$. The resulting pivotal quantity $(\bar{X} - \mu)/(s/\sqrt{n})$ follows a $t_{n-1}$ distribution, not a standard normal, because $s$ introduces additional randomness. Using the $t$-distribution accounts for this extra uncertainty by producing wider confidence intervals.

    As $n$ grows, $s \to \sigma$ and $t_{n-1} \to N(0,1)$, so the $t$-based and $z$-based intervals converge. For $n > 30$, the practical difference is small.

---

**Exercise 3.**
Prove that the $t_1$ distribution is the standard Cauchy distribution by showing their PDFs are identical.

??? success "Solution to Exercise 3"
    The $t_\nu$ PDF with $\nu = 1$, $\mu = 0$, $\sigma = 1$ is:

    $$
    f(x) = \frac{\Gamma(1)}{\sqrt{\pi}\;\Gamma(1/2)} \left(1 + x^2\right)^{-1}
    $$

    Using $\Gamma(1) = 1$ and $\Gamma(1/2) = \sqrt{\pi}$:

    $$
    f(x) = \frac{1}{\pi(1 + x^2)}
    $$

    This is exactly the standard Cauchy PDF. The Cauchy distribution therefore has no finite mean or variance, consistent with the $t$-distribution moment conditions ($\nu \le 1$). $\square$

---

**Exercise 4.**
Using SciPy, plot the $t$-distribution PDF for $\nu = 1, 5, 30, \infty$ (using `stats.norm` for $\infty$) on the same axes. Describe the convergence.

??? success "Solution to Exercise 4"
    ```python
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.linspace(-5, 5, 500)
    for nu in [1, 5, 30]:
        ax.plot(x, stats.t(nu).pdf(x), label=f"t({nu})")
    ax.plot(x, stats.norm.pdf(x), '--', label="N(0,1)")
    ax.legend()
    ```

    At $\nu = 1$ (Cauchy), the tails are extremely heavy. At $\nu = 5$, the shape is noticeably bell-curved but still wider. At $\nu = 30$, the $t$ and normal curves are nearly indistinguishable. The convergence $t_\nu \to N(0,1)$ is monotonic: each increase in $\nu$ brings the tails closer to the normal.
