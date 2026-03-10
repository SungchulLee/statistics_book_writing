# Orthogonal Projection Matrices

The orthogonal projection onto a column space is the key geometric operation in regression, producing the hat matrix that maps observed responses to fitted values.

## Definition

A matrix $P$ is an **orthogonal projection** if it is both symmetric and idempotent:

$$
P^2 = P \quad \text{and} \quad P^T = P
$$

Given a full-rank $n \times p$ matrix $X$, the orthogonal projection onto $\text{col}(X)$ is

$$
H = X(X^TX)^{-1}X^T
$$

This is the **hat matrix** of regression.

## Explanation

The hat matrix $H$ satisfies three defining properties:

1. **Idempotent**: $H^2 = H$ (projecting twice changes nothing).
2. **Symmetric**: $H^T = H$ (the projection is orthogonal, not oblique).
3. **Column space**: $\text{col}(H) = \text{col}(X)$.

The decomposition $\mathbf{y} = H\mathbf{y} + (I - H)\mathbf{y} = \hat{\mathbf{y}} + \hat{\boldsymbol{\varepsilon}}$ is the Pythagorean theorem in $\mathbb{R}^n$:

$$
\|\mathbf{y}\|^2 = \|\hat{\mathbf{y}}\|^2 + \|\hat{\boldsymbol{\varepsilon}}\|^2
$$

The diagonal entries $h_{ii}$ are called **leverages** and satisfy $0 \leq h_{ii} \leq 1$ with $\sum h_{ii} = p$. Points with high leverage have outsized influence on the fitted regression.

## Examples

```python
import numpy as np

np.random.seed(42)
n, p = 50, 3
X = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])

H = X @ np.linalg.inv(X.T @ X) @ X.T

# Verify properties
print("Symmetric:", np.allclose(H, H.T))
print("Idempotent:", np.allclose(H @ H, H))
print(f"Trace (= p): {np.trace(H):.1f}")

# Leverages
leverages = np.diag(H)
print(f"Leverage range: [{leverages.min():.4f}, {leverages.max():.4f}]")
print(f"Sum of leverages: {leverages.sum():.1f} (should be {p})")

# Pythagorean decomposition
y = np.random.randn(n)
y_hat = H @ y
e_hat = y - y_hat
print(f"||y||^2 = {y @ y:.4f}")
print(f"||y_hat||^2 + ||e||^2 = {y_hat @ y_hat + e_hat @ e_hat:.4f}")
```
