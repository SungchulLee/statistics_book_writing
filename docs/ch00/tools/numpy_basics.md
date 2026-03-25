# NumPy Arrays

## Overview

**NumPy** (Numerical Python) is the foundational library for numerical computing in Python. At its core is the `ndarray`—a fast, memory-efficient, multidimensional array that supports vectorized arithmetic and broadcasting. Nearly every scientific Python library (Pandas, SciPy, Scikit-Learn, Matplotlib) is built on top of NumPy arrays.

```python
import numpy as np
```

## Creating Arrays

### From Python Lists

```python
# 1-D array
a = np.array([1, 2, 3, 4, 5])
print(a)          # [1 2 3 4 5]
print(a.shape)    # (5,)
print(a.dtype)    # int64

# 2-D array (matrix)
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(M.shape)    # (2, 3)
```

### With Built-in Constructors

```python
np.zeros((3, 4))          # 3×4 matrix of zeros
np.ones((2, 2))           # 2×2 matrix of ones
np.full((3, 3), 7)        # 3×3 matrix filled with 7
np.eye(4)                 # 4×4 identity matrix
np.arange(0, 10, 2)       # array([0, 2, 4, 6, 8])
np.linspace(0, 1, 5)      # array([0.  , 0.25, 0.5 , 0.75, 1.  ])
```

### Random Arrays

```python
rng = np.random.default_rng(42)          # reproducible generator

rng.standard_normal((3, 3))              # 3×3 standard normal draws
rng.uniform(0, 1, size=(2, 5))           # 2×5 Uniform(0,1) draws
rng.integers(0, 10, size=6)              # 6 random ints in [0, 10)
```

## Array Attributes

| Attribute | Description | Example |
|---|---|---|
| `a.shape` | Dimensions | `(2, 3)` |
| `a.ndim` | Number of dimensions | `2` |
| `a.size` | Total number of elements | `6` |
| `a.dtype` | Element data type | `float64` |
| `a.nbytes` | Memory consumption in bytes | `48` |

## Indexing and Slicing

### One-Dimensional

```python
a = np.array([10, 20, 30, 40, 50])

a[0]        # 10        — first element
a[-1]       # 50        — last element
a[1:4]      # [20 30 40] — slice (start inclusive, stop exclusive)
a[::2]      # [10 30 50] — every other element
```

### Two-Dimensional

```python
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

M[0, 1]       # 2          — row 0, col 1
M[1, :]       # [4 5 6]    — entire row 1
M[:, 2]       # [3 6 9]    — entire col 2
M[:2, :2]     # [[1 2]     — upper-left 2×2 sub-matrix
              #  [4 5]]
```

### Boolean (Fancy) Indexing

```python
a = np.array([3, 1, 4, 1, 5, 9])

mask = a > 3
print(mask)       # [False False  True False  True  True]
print(a[mask])    # [4 5 9]
```

## Vectorized Operations

NumPy performs element-wise arithmetic without explicit loops, which is both faster and more readable than pure Python.

```python
a = np.array([1, 2, 3, 4, 5])

# Element-wise operations
print(a + 10)      # [11 12 13 14 15]
print(a * 2)       # [ 2  4  6  8 10]
print(a ** 2)       # [ 1  4  9 16 25]

# Element-wise between two arrays
b = np.array([10, 20, 30, 40, 50])
print(a + b)       # [11 22 33 44 55]
print(a * b)       # [ 10  40  90 160 250]
```

### Comparison with Pure Python

```python
import time

size = 1_000_000
py_list = list(range(size))
np_arr  = np.arange(size)

# Pure Python
start = time.time()
result_py = [x ** 2 for x in py_list]
print(f"Python list: {time.time() - start:.4f}s")

# NumPy
start = time.time()
result_np = np_arr ** 2
print(f"NumPy array: {time.time() - start:.4f}s")
```

NumPy is typically **10–100× faster** for array operations because it delegates computation to optimized C and Fortran routines.

## Broadcasting

Broadcasting is NumPy's mechanism for performing arithmetic on arrays of different shapes.

**Rules (simplified):**

1. If the arrays differ in the number of dimensions, the shape of the smaller array is padded with ones on the left.
2. Arrays with size 1 along a particular dimension act as if they had the size of the array with the largest shape along that dimension.

```python
# Scalar broadcast
a = np.array([1, 2, 3])
print(a + 100)    # [101 102 103]

# Column vector + row vector → matrix
col = np.array([[1], [2], [3]])   # shape (3, 1)
row = np.array([10, 20, 30])      # shape (3,)

print(col + row)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
```

### Statistical Application: Standardization

Broadcasting makes it natural to standardize a data matrix (subtract column means, divide by column standard deviations):

```python
# X has shape (n, p) — n observations, p features
X = rng.standard_normal((100, 5))

X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)

# Verify: each column now has mean ≈ 0 and std ≈ 1
print(X_standardized.mean(axis=0))  # ≈ [0, 0, 0, 0, 0]
print(X_standardized.std(axis=0))   # ≈ [1, 1, 1, 1, 1]
```

## Aggregation Functions

```python
a = np.array([4, 1, 7, 3, 9, 2])

a.sum()       # 26
a.mean()      # 4.333...
a.std()       # 2.687...
a.var()       # 7.222...
a.min()       # 1
a.max()       # 9
a.argmin()    # 1   — index of min
a.argmax()    # 4   — index of max
np.median(a)  # 3.5
```

### Aggregation Along an Axis

For a 2-D array, `axis=0` aggregates *down* the rows (column-wise) and `axis=1` aggregates *across* the columns (row-wise).

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])

M.sum(axis=0)    # [5 7 9]   — column sums
M.sum(axis=1)    # [6 15]    — row sums
M.mean(axis=0)   # [2.5 3.5 4.5]
```

## Linear Algebra

NumPy provides a comprehensive set of linear algebra operations essential for statistics.

### Matrix Multiplication

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[2, 3],
              [0, 1]])

# Three equivalent ways
C = A @ B                   # preferred syntax
C = np.matmul(A, B)
C = np.dot(A, B)            # identical for 2-D arrays

print(C)
# [[ 2  5]
#  [ 6 13]]
```

### Other Operations

```python
# Transpose
print(A.T)

# Determinant
print(np.linalg.det(A))       # -2.0

# Inverse
print(np.linalg.inv(A))

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Solve Ax = b
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print(x)                       # [1. 2.]
```

### Statistical Application: OLS Estimator

The ordinary-least-squares estimator $\hat{\beta} = (X^T X)^{-1} X^T y$ can be computed directly:

```python
n, p = 50, 3
rng = np.random.default_rng(0)
X = rng.standard_normal((n, p))
X = np.column_stack([np.ones(n), X])   # add intercept column
beta_true = np.array([2, -1, 0.5, 3])
y = X @ beta_true + rng.standard_normal(n) * 0.5

# OLS estimate
beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
print(beta_hat)   # ≈ [2, -1, 0.5, 3]
```

## Reshaping and Stacking

```python
a = np.arange(12)

# Reshape into 3×4
M = a.reshape(3, 4)

# Flatten back
flat = M.ravel()

# Stack arrays
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

np.vstack([v1, v2])    # [[1 2 3], [4 5 6]]
np.hstack([v1, v2])    # [1 2 3 4 5 6]
np.column_stack([v1, v2])  # [[1 4], [2 5], [3 6]]
```

## Random Number Generation for Statistics

```python
rng = np.random.default_rng(seed=42)

# Standard distributions
rng.normal(loc=0, scale=1, size=1000)         # Normal(μ=0, σ=1)
rng.uniform(low=0, high=1, size=1000)         # Uniform(0, 1)
rng.binomial(n=10, p=0.3, size=1000)          # Binomial(10, 0.3)
rng.poisson(lam=5, size=1000)                 # Poisson(λ=5)
rng.exponential(scale=2, size=1000)           # Exponential(β=2)

# Sampling
population = np.arange(100)
sample = rng.choice(population, size=10, replace=False)

# Permutation
rng.shuffle(population)         # in-place shuffle
perm = rng.permutation(100)     # returns shuffled copy
```

## Summary

| Concept | Key Takeaway |
|---|---|
| `ndarray` | Core data structure; homogeneous, fixed-size, N-dimensional |
| Vectorization | Element-wise operations replace explicit loops for speed |
| Broadcasting | Automatic shape expansion enables concise array arithmetic |
| Aggregation | `sum`, `mean`, `std`, `var` with optional `axis` parameter |
| Linear algebra | `@` for matrix multiplication; `np.linalg` for decompositions and solvers |
| Random generation | Use `default_rng(seed)` for reproducible random sampling |
