# Gaussian 2D Eigendecomposition

## Overview

The covariance matrix $\boldsymbol{\Sigma}$ of a bivariate Gaussian can be decomposed as:

$$
\boldsymbol{\Sigma} = \mathbf{U}\mathbf{D}\mathbf{U}^\top
$$

where $\mathbf{U}$ is the matrix of eigenvectors (principal directions) and $\mathbf{D} = \text{diag}(\lambda_1, \lambda_2)$ is the diagonal matrix of eigenvalues. The eigenvectors point along the axes of the probability ellipses, and $\sqrt{\lambda_i}$ gives the standard deviation in each principal direction.

---

## Geometric Interpretation

The constant-density contours of the bivariate normal satisfy:

$$
(\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) = c
$$

These are ellipses whose:

- **axes directions** align with the eigenvectors of $\boldsymbol{\Sigma}$
- **axis half-lengths** are proportional to $\sqrt{\lambda_i}$

This is a direct consequence of the eigendecomposition: in the rotated coordinate system defined by $\mathbf{U}$, the covariance matrix becomes diagonal, and the ellipses become axis-aligned.

---

## Code

```python
import numpy as np
import matplotlib.pyplot as plt

def bivariate_normal_pdf(X, Y, inv_Sigma, det_Sigma):
    return (np.exp(-(inv_Sigma[0,0]*X**2 + 2*inv_Sigma[0,1]*X*Y
                     + inv_Sigma[1,1]*Y**2) / 2)
            / (2 * np.pi * np.sqrt(det_Sigma)))

configs = [
    {"label": "Σ = [[0.5, 0.3], [0.3, 0.5]]",
     "Sigma": np.array([[0.5, 0.3], [0.3, 0.5]])},
    {"label": "Σ = [[1.0, 0.0], [0.0, 0.3]]",
     "Sigma": np.array([[1.0, 0.0], [0.0, 0.3]])},
    {"label": "Σ = [[0.2, 0.14], [0.14, 0.8]]",
     "Sigma": np.array([[0.2, 0.14], [0.14, 0.8]])},
]

x = np.linspace(-2.5, 2.5, 200)
X, Y = np.meshgrid(x, x)

fig, axes = plt.subplots(len(configs), 2, figsize=(12, 5 * len(configs)))

for i, cfg in enumerate(configs):
    Sigma = cfg["Sigma"]
    inv_Sigma = np.linalg.inv(Sigma)
    det_Sigma = np.linalg.det(Sigma)
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

    Z = bivariate_normal_pdf(X, Y, inv_Sigma, det_Sigma)

    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 3D surface
    axes[i, 0].remove()
    ax3d = fig.add_subplot(len(configs), 2, 2*i + 1, projection="3d")
    ax3d.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, edgecolor="none")
    ax3d.set_title(cfg["label"], fontsize=10)

    # Contour with eigenvectors
    ax = axes[i, 1]
    ax.contourf(X, Y, Z, levels=20, cmap="Blues", alpha=0.5)
    colors_ev = ["red", "darkgreen"]
    for j in range(2):
        scale = np.sqrt(eigenvalues[j])
        dx = eigenvectors[0, j] * scale
        dy = eigenvectors[1, j] * scale
        ax.annotate("", xy=(dx, dy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=colors_ev[j], lw=2.5))
    ax.set_title("Contour + Eigenvectors", fontsize=10)
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()
```

---

## Interpretation

For $\boldsymbol{\Sigma} = \begin{pmatrix}0.5 & 0.3 \\ 0.3 & 0.5\end{pmatrix}$:

- Eigenvalues: $\lambda_1 = 0.8$, $\lambda_2 = 0.2$
- The major eigenvector points along $(1, 1)/\sqrt{2}$ (the positive correlation direction)
- The ratio $\sqrt{\lambda_1/\lambda_2} = 2$ gives the eccentricity of the ellipse

When $\boldsymbol{\Sigma}$ is diagonal (no correlation), the eigenvectors align with the coordinate axes, and the contours are axis-aligned ellipses (or circles if the variances are equal).

---

## Exercises

**Exercise 1.**
Compute the eigenvalues and eigenvectors of $\boldsymbol{\Sigma} = \begin{pmatrix}1 & 0 \\ 0 & 0.3\end{pmatrix}$ and describe the contour shapes.

??? success "Solution to Exercise 1"
    Since $\boldsymbol{\Sigma}$ is diagonal, the eigenvalues are $\lambda_1 = 1$ and $\lambda_2 = 0.3$, with eigenvectors $\mathbf{e}_1 = (1, 0)^\top$ and $\mathbf{e}_2 = (0, 1)^\top$. The contours are axis-aligned ellipses, elongated along the $x_1$ axis (since $\lambda_1 > \lambda_2$). The ratio of axis lengths is $\sqrt{1/0.3} \approx 1.83$.

---

**Exercise 2.**
Prove that the eigenvalues of a covariance matrix are always nonneg.

??? success "Solution to Exercise 2"
    A covariance matrix $\boldsymbol{\Sigma}$ is positive semidefinite: $\mathbf{v}^\top\boldsymbol{\Sigma}\mathbf{v} \ge 0$ for all $\mathbf{v}$. If $\lambda$ is an eigenvalue with eigenvector $\mathbf{u}$ ($\|\mathbf{u}\| = 1$), then:

    $$
    0 \le \mathbf{u}^\top\boldsymbol{\Sigma}\mathbf{u} = \mathbf{u}^\top(\lambda\mathbf{u}) = \lambda
    $$

    So $\lambda \ge 0$. $\square$

---

**Exercise 3.**
Show that $\text{tr}(\boldsymbol{\Sigma}) = \lambda_1 + \lambda_2$ and $|\boldsymbol{\Sigma}| = \lambda_1\lambda_2$. Verify both for $\boldsymbol{\Sigma} = \begin{pmatrix}0.5 & 0.3 \\ 0.3 & 0.5\end{pmatrix}$.

??? success "Solution to Exercise 3"
    Since $\boldsymbol{\Sigma} = \mathbf{U}\mathbf{D}\mathbf{U}^\top$:

    $$
    \text{tr}(\boldsymbol{\Sigma}) = \text{tr}(\mathbf{U}\mathbf{D}\mathbf{U}^\top) = \text{tr}(\mathbf{D}) = \lambda_1 + \lambda_2
    $$

    $$
    |\boldsymbol{\Sigma}| = |\mathbf{U}||\mathbf{D}||\mathbf{U}^\top| = \lambda_1\lambda_2
    $$

    For the given matrix: $\text{tr} = 0.5 + 0.5 = 1.0$ and $\lambda_1 + \lambda_2 = 0.8 + 0.2 = 1.0$. Also $|\boldsymbol{\Sigma}| = 0.25 - 0.09 = 0.16$ and $\lambda_1\lambda_2 = 0.8 \times 0.2 = 0.16$. Both identities hold.

---

**Exercise 4.**
The **Mahalanobis distance** from a point $\mathbf{x}$ to the mean $\boldsymbol{\mu}$ is $d_M = \sqrt{(\mathbf{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$. Show that in the principal component coordinate system (eigenvector basis), this reduces to the Euclidean distance with each axis scaled by $1/\sqrt{\lambda_i}$.

??? success "Solution to Exercise 4"
    In the rotated coordinates $\mathbf{z} = \mathbf{U}^\top(\mathbf{x} - \boldsymbol{\mu})$:

    $$
    d_M^2 = \mathbf{z}^\top \mathbf{D}^{-1} \mathbf{z} = \frac{z_1^2}{\lambda_1} + \frac{z_2^2}{\lambda_2}
    $$

    This is the squared Euclidean distance with each component divided by $\sqrt{\lambda_i}$. The Mahalanobis distance "standardizes" each principal direction by its standard deviation, making it scale- and correlation-invariant. $\square$
