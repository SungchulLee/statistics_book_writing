# Positive Definite Matrices

Positive definiteness characterizes matrices whose quadratic forms are strictly positive, making them the natural mathematical objects for covariance matrices and optimization in statistics.

## Definition

A symmetric matrix $A$ is **positive definite** (written $A \succ 0$) if for all nonzero $\mathbf{x} \in \mathbb{R}^n$:

$$
\mathbf{x}^T A \mathbf{x} > 0
$$

Equivalently, $A$ is positive definite if and only if all its eigenvalues are strictly positive: $\lambda_i > 0$ for all $i$.

A matrix is **positive semi-definite** ($A \succeq 0$) if $\mathbf{x}^T A \mathbf{x} \geq 0$ for all $\mathbf{x}$.

## Explanation

Several equivalent characterizations exist:

- All eigenvalues positive.
- All leading principal minors positive (Sylvester's criterion).
- There exists an invertible matrix $L$ such that $A = L^T L$ (Cholesky decomposition).
- The quadratic form $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ has a unique minimum at $\mathbf{x} = \mathbf{0}$.

In statistics, the population covariance matrix $\Sigma$ is positive semi-definite by construction. It is positive definite when no linear combination of the variables is deterministic (i.e., no perfect multicollinearity). The sample covariance is positive definite when $n > p$ and the data are in general position.

Positive definiteness of $X^TX$ is required for the OLS solution $\hat{\boldsymbol{\beta}} = (X^TX)^{-1}X^T\mathbf{y}$ to exist uniquely.

## Examples

```python
import numpy as np

# Build a positive definite matrix via A = L^T L
L = np.array([[2, 0, 0],
              [1, 3, 0],
              [0.5, 0.5, 1]])
A = L.T @ L
print("A:\n", A)

eigenvalues = np.linalg.eigvalsh(A)
print("Eigenvalues:", eigenvalues.round(4))
print("All positive:", np.all(eigenvalues > 0))

# Cholesky decomposition (only works for PD matrices)
L_chol = np.linalg.cholesky(A)
print("Cholesky factor matches:", np.allclose(L_chol @ L_chol.T, A))

# Quadratic form is positive for random vectors
x = np.random.randn(3)
print(f"x'Ax = {x @ A @ x:.4f} > 0")
```
