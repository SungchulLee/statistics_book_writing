# Symmetric Matrices

Symmetric matrices are the most important class of matrices in statistics. Every covariance matrix is symmetric, and the spectral theorem guarantees a clean eigendecomposition with real eigenvalues and orthogonal eigenvectors.

## Definition

A matrix $A$ is **symmetric** if $A = A^T$. The **spectral theorem** states that every real symmetric matrix has a decomposition

$$
A = Q \Lambda Q^T
$$

where $Q$ is orthogonal ($Q^TQ = I$) and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ contains real eigenvalues.

## Explanation

The spectral theorem provides three guarantees for real symmetric matrices:

1. **All eigenvalues are real** -- no complex eigenvalues arise.
2. **Eigenvectors are orthogonal** -- eigenvectors for distinct eigenvalues are perpendicular, and one can choose orthonormal eigenvectors for repeated eigenvalues.
3. **Diagonalizable** -- the matrix is always diagonalizable (even with repeated eigenvalues), unlike the general case.

In statistics, the sample covariance matrix $S = \frac{1}{n-1}(X - \bar{X})^T(X - \bar{X})$ is always symmetric. Its spectral decomposition is the foundation of principal component analysis (PCA): the eigenvectors are the principal directions, and the eigenvalues are the variances along those directions.

## Examples

```python
import numpy as np

# Sample covariance matrix (always symmetric)
np.random.seed(42)
X = np.random.randn(100, 3) @ np.array([[2, 1, 0],
                                          [1, 3, 0.5],
                                          [0, 0.5, 1]])
S = np.cov(X, rowvar=False)
print("Symmetric:", np.allclose(S, S.T))

eigenvalues, Q = np.linalg.eigh(S)
print("Eigenvalues:", eigenvalues.round(4))
print("Orthogonal Q:", np.allclose(Q.T @ Q, np.eye(3)))

# Verify spectral decomposition
S_reconstructed = Q @ np.diag(eigenvalues) @ Q.T
print("Reconstruction error:", np.max(np.abs(S - S_reconstructed)))
```
