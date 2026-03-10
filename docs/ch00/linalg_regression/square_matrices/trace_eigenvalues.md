# Trace and Eigenvalues

The trace of a matrix equals the sum of its eigenvalues, a compact identity used throughout statistics for computing degrees of freedom and expected values of quadratic forms.

## Definition

For an $n \times n$ matrix $A$ with eigenvalues $\lambda_1, \ldots, \lambda_n$ (counted with algebraic multiplicity),

$$
\text{tr}(A) = \sum_{i=1}^n a_{ii} = \sum_{i=1}^n \lambda_i
$$

Similarly, $\det(A) = \prod_{i=1}^n \lambda_i$.

## Explanation

The trace is invariant under similarity: $\text{tr}(P^{-1}AP) = \text{tr}(A)$. Since every matrix is similar to its Jordan form (which has eigenvalues on the diagonal), the trace equals the sum of eigenvalues.

A key cyclic property is $\text{tr}(ABC) = \text{tr}(CAB) = \text{tr}(BCA)$. This is used constantly in regression:

- $\text{tr}(H) = \text{tr}(X(X^TX)^{-1}X^T) = \text{tr}((X^TX)^{-1}X^TX) = \text{tr}(I_p) = p$
- For idempotent $A$: $\text{rank}(A) = \text{tr}(A)$ since each eigenvalue is 0 or 1.
- Expected value of quadratic forms: $E[\mathbf{X}^TA\mathbf{X}] = \text{tr}(A\Sigma) + \boldsymbol{\mu}^TA\boldsymbol{\mu}$.

## Examples

```python
import numpy as np

A = np.array([[5, 2, 1],
              [0, 3, 1],
              [1, 0, 4]])

eigenvalues = np.linalg.eigvals(A)
print(f"Trace (diagonal sum): {np.trace(A):.4f}")
print(f"Sum of eigenvalues:   {eigenvalues.sum().real:.4f}")
print(f"Determinant:          {np.linalg.det(A):.4f}")
print(f"Product of eigenvalues: {eigenvalues.prod().real:.4f}")

# Cyclic property
B = np.random.randn(3, 3)
C = np.random.randn(3, 3)
print(f"\ntr(ABC) = {np.trace(A @ B @ C):.6f}")
print(f"tr(BCA) = {np.trace(B @ C @ A):.6f}")
print(f"tr(CAB) = {np.trace(C @ A @ B):.6f}")
```
