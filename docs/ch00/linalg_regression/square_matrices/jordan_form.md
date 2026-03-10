# Jordan Canonical Form

The Jordan form generalizes eigendecomposition to matrices that are not diagonalizable, providing a complete canonical form for every square matrix over the complex numbers.

## Definition

Every $n \times n$ matrix $A$ (over $\mathbb{C}$) is similar to a **Jordan matrix**

$$
J = P^{-1} A P = \text{diag}(J_1, J_2, \ldots, J_k)
$$

where each **Jordan block** $J_i$ is an $m_i \times m_i$ matrix of the form

$$
J_i = \begin{pmatrix} \lambda_i & 1 & & 0 \\ & \lambda_i & \ddots & \\ & & \ddots & 1 \\ 0 & & & \lambda_i \end{pmatrix}
$$

The Jordan form is unique up to the ordering of the blocks.

## Explanation

When a matrix has repeated eigenvalues but not enough independent eigenvectors (it is *defective*), it cannot be diagonalized. The Jordan form handles this by introducing 1s on the superdiagonal within each Jordan block, corresponding to generalized eigenvectors.

Key properties:

- A matrix is diagonalizable if and only if all Jordan blocks are $1 \times 1$.
- The number of Jordan blocks for eigenvalue $\lambda$ equals the geometric multiplicity of $\lambda$.
- The sum of block sizes for eigenvalue $\lambda$ equals the algebraic multiplicity.

In statistics, the Jordan form is rarely needed directly because covariance matrices are symmetric (hence always diagonalizable). However, it provides theoretical completeness when analyzing general linear transformations or state-space models.

## Examples

```python
import numpy as np
from scipy.linalg import jordan_normal_form

# A defective matrix: eigenvalue 2 with algebraic mult 2 but geometric mult 1
A = np.array([[2, 1],
              [0, 2]])

J, P = jordan_normal_form(A)
print("Jordan form:\n", J)
print("Transformation matrix:\n", P)
print("Verify P^{-1} A P = J:", np.allclose(np.linalg.inv(P) @ A @ P, J))

# A diagonalizable matrix for comparison
B = np.array([[3, 1],
              [0, 2]])
J_B, P_B = jordan_normal_form(B)
print("\nJordan form of B:\n", J_B)
print("B is diagonalizable: all blocks are 1x1")
```
