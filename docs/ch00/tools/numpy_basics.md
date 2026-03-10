# NumPy Arrays

NumPy provides the `ndarray`, a fast multidimensional array supporting vectorized arithmetic and broadcasting. Nearly every scientific Python library is built on top of NumPy.

## Definition

An **ndarray** is a homogeneous, fixed-size, N-dimensional array. Key attributes: `shape` (dimensions), `dtype` (element type), `ndim` (number of axes). Arrays are created from lists (`np.array([1,2,3])`), constructors (`np.zeros`, `np.eye`, `np.linspace`), or random generators (`np.random.default_rng`).

## Explanation

**Vectorization** replaces explicit Python loops with C-level operations, yielding 10--100x speedups. Element-wise arithmetic (`a + b`, `a * b`, `a ** 2`) and comparisons (`a > 3`) work directly on arrays.

**Broadcasting** automatically expands arrays of different shapes: a scalar broadcasts to any shape, and a `(3,1)` array broadcasts with a `(3,)` array to produce a `(3,3)` result. This enables concise standardization: `(X - X.mean(axis=0)) / X.std(axis=0)`.

**Aggregation** functions (`sum`, `mean`, `std`, `var`) accept an `axis` parameter: `axis=0` aggregates down rows (column-wise), `axis=1` across columns.

**Linear algebra** via `np.linalg`: matrix multiply with `@`, solve $Ax=b$ with `np.linalg.solve`, eigendecompose with `np.linalg.eigh` (symmetric) or `np.linalg.eig` (general), invert with `np.linalg.inv`.

## Examples

```python
import numpy as np

rng = np.random.default_rng(42)

# OLS estimator: beta_hat = (X'X)^{-1} X'y
n, p = 50, 3
X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
beta_true = np.array([2.0, -1.0, 0.5])
y = X @ beta_true + rng.standard_normal(n) * 0.5

beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
print("True:     ", beta_true)
print("Estimated:", beta_hat.round(4))

# Broadcasting: standardize columns
X_std = (X[:, 1:] - X[:, 1:].mean(axis=0)) / X[:, 1:].std(axis=0)
print("Column means after standardization:", X_std.mean(axis=0).round(10))
print("Column stds after standardization: ", X_std.std(axis=0).round(10))
```
