# Bivariate Normal Distribution

## Overview

The **bivariate normal distribution** extends the univariate normal to two dimensions. A random vector $(X_1, X_2)^\top$ is bivariate normal with mean $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$ if its PDF is:

$$
f(\mathbf{x}) = \frac{1}{2\pi|\boldsymbol{\Sigma}|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)
$$

The shape of the density contours (ellipses) is entirely determined by $\boldsymbol{\Sigma}$.

---

## Effect of Covariance Structure

We visualize four configurations as 3D surfaces and contour plots:

| Configuration | $\boldsymbol{\Sigma}$ | Correlation $\rho$ |
|---|---|---|
| Independent | $\begin{pmatrix}4&0\\0&4\end{pmatrix}$ | 0 |
| Positive correlation | $\begin{pmatrix}4&2.8\\2.8&4\end{pmatrix}$ | 0.7 |
| Negative correlation | $\begin{pmatrix}4&-2.8\\-2.8&4\end{pmatrix}$ | $-0.7$ |
| Unequal variances | $\begin{pmatrix}7&0\\0&15\end{pmatrix}$ | 0 |

---

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

configs = [
    {"label": "Independent (ρ=0)", "mu": [0, 0], "cov": [[4, 0], [0, 4]]},
    {"label": "Positive corr (ρ=0.7)", "mu": [0, 0], "cov": [[4, 2.8], [2.8, 4]]},
    {"label": "Negative corr (ρ=−0.7)", "mu": [0, 0], "cov": [[4, -2.8], [-2.8, 4]]},
    {"label": "Unequal variances", "mu": [0, 0], "cov": [[7, 0], [0, 15]]},
]

x = np.linspace(-10, 10, 200)
X, Y = np.meshgrid(x, x)

fig = plt.figure(figsize=(18, 12))
for i, cfg in enumerate(configs):
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mean=cfg["mu"], cov=cfg["cov"])
    Z = rv.pdf(pos)

    # 3D surface
    ax = fig.add_subplot(2, 4, i + 1, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, edgecolor="none")
    ax.set_title(cfg["label"], fontsize=9)

    # Contour
    ax2 = fig.add_subplot(2, 4, i + 5)
    ax2.contourf(X, Y, Z, levels=20, cmap="viridis")
    ax2.contour(X, Y, Z, levels=8, colors="white", linewidths=0.5)
    ax2.set_title(cfg["label"], fontsize=9)
    ax2.set_aspect("equal")

plt.suptitle("Bivariate Normal: 3D Surface (top) and Contour (bottom)")
plt.tight_layout()
plt.show()
```

---

## Interpretation

- **$\rho = 0$, equal variances:** Circular contours — $X_1$ and $X_2$ are independent with identical spread.
- **$\rho > 0$:** Ellipses tilted along the $X_1 = X_2$ diagonal — positive association.
- **$\rho < 0$:** Ellipses tilted along $X_1 = -X_2$ — negative association.
- **Unequal variances:** Ellipses elongated along the axis with larger variance.

The contour ellipses satisfy $(\mathbf{x} - \boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}) = c$ for constant $c$. Their axes align with the eigenvectors of $\boldsymbol{\Sigma}$, and the axis lengths are proportional to $\sqrt{\lambda_i}$ (the square roots of the eigenvalues).

---

## Exercises

**Exercise 1.**
For the bivariate normal with $\boldsymbol{\Sigma} = \begin{pmatrix}4&2.8\\2.8&4\end{pmatrix}$, compute the correlation $\rho$.

??? success "Solution to Exercise 1"
    $$
    \rho = \frac{\text{Cov}(X_1, X_2)}{\sigma_1 \sigma_2} = \frac{2.8}{\sqrt{4}\sqrt{4}} = \frac{2.8}{4} = 0.7
    $$

---

**Exercise 2.**
Show that for the bivariate normal, zero correlation implies independence.

??? success "Solution to Exercise 2"
    When $\rho = 0$, the covariance matrix is diagonal: $\boldsymbol{\Sigma} = \text{diag}(\sigma_1^2, \sigma_2^2)$. Then:

    $$
    f(x_1, x_2) = \frac{1}{2\pi\sigma_1\sigma_2}\exp\!\left(-\frac{x_1^2}{2\sigma_1^2} - \frac{x_2^2}{2\sigma_2^2}\right) = f_1(x_1)\cdot f_2(x_2)
    $$

    The joint density factors into the product of the marginal densities, so $X_1$ and $X_2$ are independent.

    !!! warning "Bivariate Normal Only"
        Zero correlation implies independence **only** for the bivariate normal. In general, uncorrelated random variables can be dependent.

    $\square$

---

**Exercise 3.**
Compute the eigenvalues and eigenvectors of $\boldsymbol{\Sigma} = \begin{pmatrix}4&2.8\\2.8&4\end{pmatrix}$ and describe the orientation of the contour ellipses.

??? success "Solution to Exercise 3"
    The characteristic equation is $(4-\lambda)^2 - 2.8^2 = 0$:

    $$
    \lambda^2 - 8\lambda + 16 - 7.84 = 0 \implies \lambda^2 - 8\lambda + 8.16 = 0
    $$

    $$
    \lambda = \frac{8 \pm \sqrt{64 - 32.64}}{2} = \frac{8 \pm 5.6}{2}
    $$

    So $\lambda_1 = 6.8$ and $\lambda_2 = 1.2$.

    For $\lambda_1 = 6.8$: eigenvector is $(1, 1)^\top/\sqrt{2}$ (the $X_1 = X_2$ direction).
    For $\lambda_2 = 1.2$: eigenvector is $(1, -1)^\top/\sqrt{2}$ (the $X_1 = -X_2$ direction).

    The major axis of the ellipse points along $(1,1)$ with half-length $\sqrt{6.8} \approx 2.61$, and the minor axis along $(1,-1)$ with half-length $\sqrt{1.2} \approx 1.10$.

---

**Exercise 4.**
Prove that the marginal distribution of $X_1$ from a bivariate normal $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ is $N(\mu_1, \sigma_1^2)$.

??? success "Solution to Exercise 4"
    Integrate out $X_2$ from the joint density. Write the exponent as a quadratic form in $(x_1, x_2)$ and complete the square in $x_2$. The $x_2$ integral is a Gaussian integral that evaluates to a constant, leaving:

    $$
    f_{X_1}(x_1) = \frac{1}{\sigma_1\sqrt{2\pi}}\exp\!\left(-\frac{(x_1 - \mu_1)^2}{2\sigma_1^2}\right)
    $$

    Alternatively, note that $X_1 = (1, 0)\mathbf{X}$, and any linear transformation of a multivariate normal is normal with mean $(1,0)\boldsymbol{\mu} = \mu_1$ and variance $(1,0)\boldsymbol{\Sigma}(1,0)^\top = \sigma_1^2$. $\square$
