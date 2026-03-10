# Similar Matrices

Matrix similarity is the equivalence relation underlying eigendecomposition and canonical forms. Similar matrices represent the same linear transformation in different bases.

## Definition

Two $n \times n$ matrices $A$ and $B$ are **similar** if there exists an invertible matrix $P$ such that

$$
B = P^{-1} A P
$$

Similar matrices share the same eigenvalues, determinant, trace, rank, and characteristic polynomial.

## Explanation

Similarity corresponds to a change of basis: if $A$ represents a linear map $T$ in one basis, then $B = P^{-1}AP$ represents $T$ in the basis given by the columns of $P$. The invariants under similarity are precisely the properties intrinsic to the transformation, not to the choice of coordinates.

The practical goal is to find a similar matrix with simple structure -- ideally diagonal. When $A$ is diagonalizable, $D = P^{-1}AP$ with $D$ diagonal. When it is not, the simplest similar matrix is the Jordan form.

In statistics, changing from raw predictors to principal components is a similarity transformation applied to the Gram matrix $X^TX$.

## Examples

```python
import numpy as np

A = np.array([[4, 1],
              [2, 3]])

# Change-of-basis matrix
P = np.array([[1, 1],
              [1, -2]])
P_inv = np.linalg.inv(P)

B = P_inv @ A @ P
print("A:\n", A)
print("B = P^{-1} A P:\n", B.round(10))

# Verify shared invariants
print(f"det(A) = {np.linalg.det(A):.2f}, det(B) = {np.linalg.det(B):.2f}")
print(f"tr(A)  = {np.trace(A):.2f}, tr(B)  = {np.trace(B):.2f}")
print(f"eig(A) = {sorted(np.linalg.eigvals(A).real)}")
print(f"eig(B) = {sorted(np.linalg.eigvals(B).real)}")
```
