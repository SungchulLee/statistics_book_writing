# Chi-Squared Distribution and Quadratic Forms

Quadratic forms of normal random vectors arise throughout hypothesis testing and confidence intervals. This page connects matrix algebra to the chi-squared distribution via projection matrices.

## Definition

Let $\mathbf{Z} \sim N(\mathbf{0}, I_n)$ be a standard normal vector and let $A$ be an $n \times n$ symmetric idempotent matrix of rank $r$. Then the quadratic form

$$
Q = \mathbf{Z}^T A \mathbf{Z} \sim \chi^2(r)
$$

More generally, if $\mathbf{X} \sim N(\boldsymbol{\mu}, \Sigma)$ with $\Sigma$ positive definite, then

$$
(\mathbf{X} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{X} - \boldsymbol{\mu}) \sim \chi^2(n)
$$

## Explanation

The key insight is that symmetric idempotent matrices are orthogonal projections. Since $A$ has eigenvalues 0 and 1 with multiplicity $r$ for eigenvalue 1, the spectral decomposition gives $A = P \Lambda P^T$ where $\Lambda$ has exactly $r$ ones on the diagonal. The quadratic form $\mathbf{Z}^T A \mathbf{Z}$ reduces to a sum of $r$ independent squared standard normals, which is the definition of $\chi^2(r)$.

This result underpins:

- **One-sample chi-squared test**: $(n-1)S^2/\sigma^2 \sim \chi^2(n-1)$ because the centering matrix $I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$ is idempotent with rank $n-1$.
- **Cochran's theorem**: Partitions a sum of squared normals into independent chi-squared components corresponding to orthogonal projections.
- **F-tests**: Ratios of independent chi-squared variables (each from a quadratic form) yield $F$ distributions.

## Examples

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n = 50
r = 10

# Build a rank-r symmetric idempotent matrix
U = np.linalg.qr(np.random.randn(n, r))[0]
A = U @ U.T  # orthogonal projection onto col(U)

# Verify idempotency and rank
assert np.allclose(A @ A, A), "A is not idempotent"
assert np.isclose(np.trace(A), r), f"rank should be {r}"

# Simulate quadratic form Z'AZ
n_sims = 100_000
Q_vals = np.array([
    np.random.randn(n) @ A @ np.random.randn(n)
    for _ in range(n_sims)
])

# Compare to chi-squared(r)
print(f"Simulated mean: {Q_vals.mean():.2f}, expected: {r}")
print(f"Simulated var:  {Q_vals.var():.2f}, expected: {2*r}")

# KS test against chi-squared(r)
stat, pval = stats.kstest(Q_vals, 'chi2', args=(r,))
print(f"KS p-value: {pval:.4f} (large => good fit)")
```
