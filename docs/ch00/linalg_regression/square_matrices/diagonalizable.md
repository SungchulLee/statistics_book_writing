# Diagonalizable Matrices

A diagonalizable matrix can be decomposed into its eigenvalues and eigenvectors, enabling direct computation of matrix powers, exponentials, and quadratic forms that appear throughout statistics.

## Definition

An $n \times n$ matrix $A$ is **diagonalizable** if there exists an invertible matrix $P$ and a diagonal matrix $D$ such that

$$
A = P D P^{-1}
$$

where the columns of $P$ are eigenvectors of $A$ and the diagonal entries of $D$ are the corresponding eigenvalues. Equivalently, $A$ is diagonalizable if and only if it has $n$ linearly independent eigenvectors.

## Explanation

A matrix is diagonalizable when its eigenspaces span the full space. Sufficient conditions include:

- **Distinct eigenvalues**: If all $n$ eigenvalues are distinct, the eigenvectors are automatically linearly independent.
- **Symmetric matrices**: Every real symmetric matrix is diagonalizable with an orthogonal eigenvector matrix ($P^{-1} = P^T$), by the spectral theorem.

The decomposition $A = PDP^{-1}$ simplifies many computations. For instance, $A^k = PD^kP^{-1}$, where $D^k$ simply raises each diagonal entry to the $k$-th power.

In statistics, covariance matrices are real symmetric and therefore always diagonalizable. Their eigendecomposition reveals the principal directions of variation (PCA).

## Examples

```python
import numpy as np

# Symmetric matrix (always diagonalizable)
A = np.array([[4, 2],
              [2, 3]])

eigenvalues, P = np.linalg.eigh(A)
D = np.diag(eigenvalues)

# Verify decomposition: A = P D P^T
A_reconstructed = P @ D @ P.T
print("Original:\n", A)
print("Reconstructed:\n", A_reconstructed.round(10))
print("Eigenvalues:", eigenvalues)

# Compute A^5 via eigendecomposition
A5_direct = np.linalg.matrix_power(A, 5)
A5_eigen = P @ np.diag(eigenvalues**5) @ P.T
print("A^5 match:", np.allclose(A5_direct, A5_eigen))
```
