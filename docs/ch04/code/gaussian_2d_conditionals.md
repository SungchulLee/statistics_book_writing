# Gaussian 2D Conditional Distributions

## Overview

A powerful property of the bivariate normal is that **conditional distributions are also normal**. If $(a, b)^\top$ follows a standard bivariate normal with correlation $\rho$, then:

$$
b \mid a = a_0 \;\sim\; N\!\left(\rho\, a_0,\; 1 - \rho^2\right)
$$

The conditional mean is a linear function of $a_0$, and the conditional variance $1 - \rho^2$ does not depend on $a_0$—it depends only on $\rho$.

---

## General Case

For a bivariate normal with means $\mu_a, \mu_b$, variances $\sigma_a^2, \sigma_b^2$, and correlation $\rho$:

$$
b \mid a = a_0 \;\sim\; N\!\left(\mu_b + \rho\frac{\sigma_b}{\sigma_a}(a_0 - \mu_a),\; \sigma_b^2(1 - \rho^2)\right)
$$

!!! tip "Key Insight"
    The conditional mean is exactly the **regression line** of $b$ on $a$. The conditional variance is the residual variance after accounting for the linear relationship.

---

## Code

```python
import numpy as np
import matplotlib.pyplot as plt

def bivariate_gaussian_pdf(x, y, rho):
    return (np.exp(-(x**2 - 2*rho*x*y + y**2) / (2*(1 - rho**2)))
            / (2 * np.pi * np.sqrt(1 - rho**2)))

def conditional_pdf(x0, y, rho):
    sigma_cond = np.sqrt(1 - rho**2)
    mu_cond = rho * x0
    return (1 / (np.sqrt(2*np.pi) * sigma_cond)
            * np.exp(-0.5 * ((y - mu_cond) / sigma_cond)**2))

x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

rho_vals = [0.0, 0.5, 0.9]
cond_val = 1.0  # condition on a = 1

fig, axes = plt.subplots(len(rho_vals), 2, figsize=(10, 4 * len(rho_vals)))

for i, rho in enumerate(rho_vals):
    Z = bivariate_gaussian_pdf(X, Y, rho)
    mu_cond = rho * cond_val
    sigma_cond = np.sqrt(1 - rho**2)

    # Joint contour with slice
    ax = axes[i, 0]
    ax.contourf(X, Y, Z, levels=20, cmap="Blues", alpha=0.4)
    ax.axvline(cond_val, color="red", linestyle="--", lw=2, label=f"a = {cond_val}")
    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_title(f"Joint PDF (ρ = {rho})")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # Conditional PDF
    ax = axes[i, 1]
    cond_y = conditional_pdf(cond_val, y, rho)
    ax.plot(y, cond_y, "b-", lw=2)
    ax.axvline(mu_cond, color="red", linestyle="--", lw=1.5,
               label=f"E[b|a=1] = {mu_cond:.2f}")
    ax.fill_between(y, cond_y, alpha=0.15, color="blue")
    ax.set_xlabel("b")
    ax.set_ylabel("f(b | a = 1)")
    ax.set_title(f"Conditional PDF (σ = {sigma_cond:.3f})")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()
```

---

## Interpretation

| $\rho$ | $E[b \mid a=1]$ | $\text{Var}(b \mid a=1)$ | Effect |
|---|---|---|---|
| 0 | 0 | 1 | Conditioning on $a$ provides no information about $b$ |
| 0.5 | 0.5 | 0.75 | Moderate reduction in uncertainty |
| 0.9 | 0.9 | 0.19 | Knowing $a$ nearly determines $b$ |

As $|\rho| \to 1$, the conditional distribution concentrates around the regression line, and the conditional variance approaches zero.

---

## Exercises

**Exercise 1.**
For the standard bivariate normal with $\rho = 0.8$, compute the conditional mean and variance of $b$ given $a = 2$.

??? success "Solution to Exercise 1"
    $$
    E[b \mid a = 2] = \rho \cdot 2 = 0.8 \times 2 = 1.6
    $$

    $$
    \text{Var}(b \mid a = 2) = 1 - \rho^2 = 1 - 0.64 = 0.36
    $$

    So $b \mid a = 2 \sim N(1.6, 0.36)$, with conditional standard deviation $0.6$.

---

**Exercise 2.**
Prove the conditional distribution formula for the standard bivariate normal: $b \mid a = a_0 \sim N(\rho a_0, 1 - \rho^2)$.

??? success "Solution to Exercise 2"
    The joint density is:

    $$
    f(a, b) = \frac{1}{2\pi\sqrt{1-\rho^2}}\exp\!\left(-\frac{a^2 - 2\rho ab + b^2}{2(1-\rho^2)}\right)
    $$

    The marginal of $a$ is $f_a(a) = \frac{1}{\sqrt{2\pi}}e^{-a^2/2}$. So:

    $$
    f(b \mid a = a_0) = \frac{f(a_0, b)}{f_a(a_0)} \propto \exp\!\left(-\frac{(b - \rho a_0)^2}{2(1-\rho^2)}\right)
    $$

    after completing the square in $b$. This is the kernel of $N(\rho a_0, 1-\rho^2)$. $\square$

---

**Exercise 3.**
Explain the connection between the conditional mean $E[b \mid a] = \rho \cdot a$ and simple linear regression of $b$ on $a$.

??? success "Solution to Exercise 3"
    In simple linear regression of $b$ on $a$ with standardized variables (zero mean, unit variance), the regression line is $\hat{b} = \rho \cdot a$, where $\rho$ is the correlation. For the bivariate normal, this regression line is not just the best linear predictor — it is the **conditional expectation** $E[b \mid a]$. In general, $E[Y \mid X]$ can be nonlinear, but for the bivariate normal it is exactly linear.

---

**Exercise 4.**
If $\rho = 0$, what does the conditional distribution become? Relate this to the concept of independence.

??? success "Solution to Exercise 4"
    When $\rho = 0$: $E[b \mid a = a_0] = 0$ and $\text{Var}(b \mid a = a_0) = 1$. The conditional distribution $b \mid a \sim N(0, 1)$ does not depend on $a_0$ at all — it equals the marginal distribution of $b$.

    This is the definition of independence: $f(b \mid a) = f(b)$ for all $a$. For the bivariate normal, $\rho = 0$ is both necessary and sufficient for independence.
