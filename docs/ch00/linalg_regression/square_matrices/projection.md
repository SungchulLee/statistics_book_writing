# Projection Matrices

Projection matrices map vectors onto subspaces and are the geometric foundation of least-squares fitting, where predicted values are projections of the response onto the column space of the design matrix.

## Definition

A square matrix $P$ is a **projection matrix** if it is idempotent:

$$
P^2 = P
$$

$P$ projects any vector $\mathbf{v}$ onto $\text{col}(P)$, the column space of $P$. The complementary projection $I - P$ maps onto the null space of $P$.

## Explanation

Every projection decomposes a vector into two orthogonal components when the projection is orthogonal (i.e., $P$ is also symmetric). In general, idempotent matrices that are not symmetric produce oblique projections where the two components are not perpendicular.

In regression, the hat matrix $H = X(X^TX)^{-1}X^T$ is the orthogonal projection onto $\text{col}(X)$:

$$
\hat{\mathbf{y}} = H\mathbf{y}, \qquad \hat{\boldsymbol{\varepsilon}} = (I - H)\mathbf{y}
$$

Key properties of projection matrices:

- Eigenvalues are 0 or 1 (from idempotency).
- $\text{rank}(P) = \text{tr}(P)$.
- $P$ and $I - P$ project onto complementary subspaces.

## Examples

```python
import numpy as np

np.random.seed(0)
n = 5

# Projection onto span of first two standard basis vectors
P = np.zeros((n, n))
P[0, 0] = 1
P[1, 1] = 1

v = np.random.randn(n)
projected = P @ v
residual = v - projected

print("v:", v.round(3))
print("Projection:", projected.round(3))
print("Residual:", residual.round(3))
print("Idempotent:", np.allclose(P @ P, P))
print("Rank = trace:", np.isclose(np.trace(P), np.linalg.matrix_rank(P)))

# Orthogonality of projected and residual
print("Orthogonal:", np.isclose(projected @ residual, 0))
```
