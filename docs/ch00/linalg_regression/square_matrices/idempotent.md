# Idempotent Matrices

Idempotent matrices are the algebraic backbone of projection in regression. Applying the matrix once projects a vector onto a subspace; applying it again changes nothing.

## Definition

A square matrix $A$ is **idempotent** if

$$
A^2 = A
$$

The eigenvalues of an idempotent matrix are either 0 or 1, and $\text{rank}(A) = \text{tr}(A)$.

## Explanation

Idempotency captures the idea that once you project, projecting again has no further effect. In regression:

- The **hat matrix** $H = X(X^TX)^{-1}X^T$ is idempotent: it projects $\mathbf{y}$ onto the column space of $X$ to produce $\hat{\mathbf{y}}$.
- The **residual maker** $M = I - H$ is also idempotent: it projects onto the orthogonal complement, producing $\hat{\boldsymbol{\varepsilon}} = M\mathbf{y}$.

Since the only eigenvalues are 0 and 1, the rank equals the number of eigenvalues equal to 1, which equals the trace. For $H$, this gives $\text{tr}(H) = p$ (number of parameters), and for $M$, $\text{tr}(M) = n - p$ (residual degrees of freedom).

This trace-rank identity is why $\text{RSS}/\sigma^2 \sim \chi^2(n-p)$: the quadratic form $\mathbf{Z}^T M \mathbf{Z}$ through an idempotent matrix of rank $n-p$ yields a chi-squared with $n-p$ degrees of freedom.

## Examples

```python
import numpy as np

np.random.seed(0)
n, p = 50, 3
X = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])

# Hat matrix
H = X @ np.linalg.inv(X.T @ X) @ X.T
M = np.eye(n) - H

# Verify idempotency
print("H idempotent:", np.allclose(H @ H, H))
print("M idempotent:", np.allclose(M @ M, M))

# Trace equals rank
print(f"tr(H) = {np.trace(H):.1f}, expected p = {p}")
print(f"tr(M) = {np.trace(M):.1f}, expected n-p = {n - p}")

# Eigenvalues are 0 or 1
eigs_H = np.linalg.eigvalsh(H)
print("H eigenvalues (unique, rounded):", np.unique(eigs_H.round(8)))
```
