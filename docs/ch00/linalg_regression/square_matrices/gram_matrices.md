# Gram Matrices

Gram matrices of the form $X^T X$ appear in every least-squares problem and carry the geometric and algebraic information needed to solve the normal equations.

## Definition

Given an $n \times p$ matrix $X$, the **Gram matrix** is

$$
G = X^T X
$$

It is a $p \times p$ symmetric positive semi-definite matrix. If $X$ has full column rank ($\text{rank}(X) = p$), then $G$ is positive definite and invertible.

## Explanation

The Gram matrix encodes inner products between the columns of $X$: the $(i,j)$ entry of $X^T X$ equals $\mathbf{x}_i^T \mathbf{x}_j$, the dot product of the $i$-th and $j$-th predictor vectors. This makes $X^T X$ the "information" matrix in regression.

Key properties:

- **Positive semi-definiteness**: For any $\mathbf{v} \in \mathbb{R}^p$, $\mathbf{v}^T X^T X \mathbf{v} = \|X\mathbf{v}\|^2 \geq 0$.
- **Normal equations**: The OLS solution satisfies $X^T X \hat{\boldsymbol{\beta}} = X^T \mathbf{y}$.
- **Covariance of OLS**: $\text{Cov}(\hat{\boldsymbol{\beta}}) = \sigma^2 (X^T X)^{-1}$, so ill-conditioning of $X^T X$ inflates estimator variance.
- **Eigenvalues**: The eigenvalues of $X^T X$ are the squared singular values of $X$. Their ratio (condition number) measures multicollinearity.

## Examples

```python
import numpy as np

np.random.seed(42)
n, p = 100, 3
X = np.column_stack([np.ones(n), np.random.randn(n, p - 1)])

G = X.T @ X
print("Gram matrix:\n", G.round(2))

# Verify symmetry and positive definiteness
eigenvalues = np.linalg.eigvalsh(G)
print("Eigenvalues of X'X:", eigenvalues.round(4))
print("All positive:", np.all(eigenvalues > 0))

# Condition number
cond = eigenvalues.max() / eigenvalues.min()
print(f"Condition number: {cond:.2f}")

# Solve normal equations
y = X @ np.array([1, -2, 3]) + np.random.randn(n) * 0.5
beta_hat = np.linalg.solve(G, X.T @ y)
print("OLS coefficients:", beta_hat.round(4))
```
