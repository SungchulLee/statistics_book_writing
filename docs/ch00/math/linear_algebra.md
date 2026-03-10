# Linear Algebra Notation and Conventions

Linear algebra is the language of multivariate statistics. This section establishes the vector and matrix notation used throughout the book and reviews the essential operations and results.

## Definition

Vectors are column vectors $\mathbf{x} \in \mathbb{R}^n$ in bold lowercase; matrices are $\mathbf{A} \in \mathbb{R}^{m \times n}$ in bold uppercase. The core operations are:

$$
\mathbf{x}^T\mathbf{y} = \sum_{i=1}^n x_i y_i, \qquad \|\mathbf{x}\| = \sqrt{\mathbf{x}^T\mathbf{x}}, \qquad [\mathbf{AB}]_{ij} = \sum_{\ell} a_{i\ell} b_{\ell j}
$$

The **design matrix** $\mathbf{X} \in \mathbb{R}^{n \times p}$ stacks $n$ observations of $p$ features row-wise.

## Explanation

**Key matrix properties**: The trace $\text{tr}(\mathbf{A}) = \sum a_{ii}$ is cyclic ($\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})$). The determinant $\det(\mathbf{A})$ is nonzero iff $\mathbf{A}$ is invertible. The rank equals the number of linearly independent columns, and full column rank of $\mathbf{X}$ is required for the OLS estimator to exist.

**Eigenvalues and eigenvectors**: $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$. For symmetric matrices (including covariance matrices), all eigenvalues are real, eigenvectors are orthogonal, and the spectral decomposition $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ always exists.

**Positive definiteness**: $\mathbf{A} \succ 0$ iff $\mathbf{x}^T\mathbf{A}\mathbf{x} > 0$ for all nonzero $\mathbf{x}$, equivalently all eigenvalues positive. Covariance matrices are positive semi-definite; they are positive definite when no variable is a deterministic linear combination of others.

**Hat matrix and projections**: $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is symmetric and idempotent with $\text{tr}(\mathbf{H}) = p$. It produces $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$, and $\mathbf{I} - \mathbf{H}$ projects onto the residual space.

**Matrix calculus for OLS**: Minimizing $(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$ yields

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

## Examples

```python
import numpy as np

np.random.seed(42)
n, p = 50, 3
X = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])
beta_true = np.array([2.0, -1.0, 0.5])
y = X @ beta_true + np.random.randn(n) * 0.5

# OLS via normal equations
beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
print("True beta:", beta_true)
print("OLS beta: ", beta_hat.round(4))

# Hat matrix properties
H = X @ np.linalg.inv(X.T @ X) @ X.T
print(f"Symmetric: {np.allclose(H, H.T)}")
print(f"Idempotent: {np.allclose(H @ H, H)}")
print(f"tr(H) = {np.trace(H):.1f} (should be {p})")

# Eigenvalues of X'X
eigvals = np.linalg.eigvalsh(X.T @ X)
print(f"Eigenvalues of X'X: {eigvals.round(2)}")
print(f"Condition number: {eigvals.max()/eigvals.min():.2f}")
```
