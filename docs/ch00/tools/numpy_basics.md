# NumPy Arrays

## Overview

**NumPy** (Numerical Python) is the foundational library for numerical computing in Python. At its core is the `ndarray` — a homogeneous, fixed-size, multidimensional array backed by contiguous memory and operated on by routines written in C, Fortran, and (where available) BLAS/LAPACK. Three properties give NumPy its leverage:

1. **Vectorization**: arithmetic operates on whole arrays at once, so Python loops disappear.
2. **Broadcasting**: arrays of different shapes are made conformable by an automatic rule, enabling concise expressions like `(X - X.mean(axis=0)) / X.std(axis=0)` for standardization.
3. **Memory locality**: a 1-D array of one million `float64`s occupies 8 MB of contiguous memory, so vector arithmetic is cache-friendly.

Every scientific Python library — pandas, SciPy, scikit-learn, statsmodels, Matplotlib — is built on `ndarray` as its data interchange format. Mastering NumPy is therefore the prerequisite to using the rest of the ecosystem.

```python
import numpy as np
```

## Creating Arrays

### From Python lists

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

### With built-in constructors

```python
np.zeros((3, 4))          # 3×4 matrix of zeros
np.ones((2, 2))           # 2×2 matrix of ones
np.full((3, 3), 7)        # 3×3 matrix filled with 7
np.eye(4)                 # 4×4 identity matrix
np.arange(0, 10, 2)       # array([0, 2, 4, 6, 8])
np.linspace(0, 1, 5)      # array([0., 0.25, 0.5, 0.75, 1.])
```

`arange` mirrors Python's `range` (step-based, may miss the endpoint). `linspace` distributes a fixed number of points across an inclusive interval — usually preferable when plotting a function over a domain.

### Random arrays (modern API)

The legacy `np.random.*` functions still work, but the **`Generator`** API introduced in NumPy 1.17 is preferred: it is faster, supports parallel streams, and isolates state from global mutations.

```python
rng = np.random.default_rng(seed=42)         # reproducible generator
rng.standard_normal((3, 3))                  # 3×3 standard normal draws
rng.uniform(0, 1, size=(2, 5))               # 2×5 Uniform(0,1) draws
rng.integers(0, 10, size=6)                  # 6 ints in [0, 10)
```

## Array Attributes

| Attribute | Description | Example |
|---|---|---|
| `a.shape` | Dimensions | `(2, 3)` |
| `a.ndim` | Number of dimensions | `2` |
| `a.size` | Total number of elements | `6` |
| `a.dtype` | Element data type | `float64` |
| `a.nbytes` | Memory consumption in bytes | `48` |

The product of `a.shape` always equals `a.size`. `a.nbytes` equals `a.size * a.dtype.itemsize`.

## Indexing and Slicing

### One-dimensional

```python
a = np.array([10, 20, 30, 40, 50])

a[0]        # 10        — first element
a[-1]       # 50        — last element
a[1:4]      # [20 30 40] — slice (start inclusive, stop exclusive)
a[::2]      # [10 30 50] — every other element
a[::-1]     # [50 40 30 20 10] — reversed
```

### Two-dimensional

```python
M = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

M[0, 1]       # 2          — row 0, col 1
M[1, :]       # [4 5 6]    — entire row 1
M[:, 2]       # [3 6 9]    — entire col 2
M[:2, :2]     # [[1 2],    — upper-left 2×2 sub-matrix
              #  [4 5]]
```

!!! note "Views vs. copies"
    Basic slicing (`a[1:4]`, `M[:2, :2]`) returns a **view** into the same memory; modifying the slice modifies the original. Boolean indexing and fancy indexing return **copies**. Use `arr.copy()` to be explicit when in doubt.

### Boolean (fancy) indexing

```python
a = np.array([3, 1, 4, 1, 5, 9])

mask = a > 3
print(mask)       # [False False  True False  True  True]
print(a[mask])    # [4 5 9]

# Combine masks with bit-wise &, |, ~
print(a[(a > 2) & (a < 6)])    # [3 4 5]
```

Boolean indexing is the natural way to filter without writing loops.

## Vectorized Operations

NumPy performs element-wise arithmetic without explicit loops, which is both faster and more readable than pure Python:

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

a + 10        # [11 12 13 14 15]
a * 2         # [ 2  4  6  8 10]
a ** 2        # [ 1  4  9 16 25]
a + b         # [11 22 33 44 55]
a * b         # [ 10  40  90 160 250]
np.sqrt(a)    # [1.    1.414 1.732 2.    2.236]
```

### Why this is faster

```python
import time

size = 1_000_000
py_list = list(range(size))
np_arr  = np.arange(size)

t0 = time.perf_counter()
[x ** 2 for x in py_list]
print(f"Python list: {time.perf_counter() - t0:.4f} s")

t0 = time.perf_counter()
np_arr ** 2
print(f"NumPy array: {time.perf_counter() - t0:.4f} s")
```

NumPy is typically **10–100× faster** for array operations because the inner loop is in C and the data is stored contiguously, enabling SIMD instructions and cache-friendly access.

## Broadcasting

Broadcasting is NumPy's mechanism for performing arithmetic on arrays of different shapes without making intermediate copies. The rules:

1. If the arrays differ in number of dimensions, prepend `1`s to the smaller shape.
2. Two shapes are compatible in a dimension if they are equal **or** one of them is 1; the broadcast result takes the larger size.

```python
# Scalar broadcast (shape () broadcasts to anything)
a = np.array([1, 2, 3])
a + 100            # [101 102 103]

# Column vector (3,1) + row vector (3,) → (3,3)
col = np.array([[1], [2], [3]])
row = np.array([10, 20, 30])
print(col + row)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
```

### Statistical application: standardization

```python
rng = np.random.default_rng(42)
X = rng.standard_normal((100, 5))            # (n=100) × (p=5)

X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
print(X_std.mean(axis=0).round(8))           # ≈ zeros
print(X_std.std(axis=0, ddof=1).round(8))    # ≈ ones
```

Without broadcasting, this would require a loop over columns; with it, the statistical idea ("center each column, then scale by its standard deviation") translates directly into code.

## Aggregation

```python
a = np.array([4, 1, 7, 3, 9, 2])

a.sum()        # 26
a.mean()       # 4.333...
a.std()        # 2.687...  (default ddof=0)
a.std(ddof=1)  # 2.943...  (Bessel-corrected sample std)
a.var()        # 7.222...
a.min()        # 1
a.max()        # 9
a.argmin()     # 1   — index of min
a.argmax()     # 4   — index of max
np.median(a)   # 3.5
```

### Aggregation along an axis

For a 2-D array, `axis=0` collapses *rows* (gives a per-column result) and `axis=1` collapses *columns* (gives a per-row result).

```python
M = np.array([[1, 2, 3],
              [4, 5, 6]])

M.sum(axis=0)    # [5 7 9]   — column sums
M.sum(axis=1)    # [6 15]    — row sums
M.mean(axis=0)   # [2.5 3.5 4.5]
```

!!! warning "Default `ddof=0`"
    NumPy's `var` and `std` default to `ddof=0` (population variance, dividing by $n$). Sample variance with Bessel's correction is `ddof=1`, dividing by $n - 1$. Pandas defaults to `ddof=1`. Mixing the two in the same analysis is a classic source of subtle off-by-one bugs.

## Linear Algebra

```python
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[2, 3],
              [0, 1]])

A @ B                  # preferred — matrix multiplication
np.matmul(A, B)
np.dot(A, B)           # identical for 2-D arrays

A.T                    # transpose
np.linalg.det(A)       # -2.0
np.linalg.inv(A)       # inverse (use solve() when possible)
np.linalg.eig(A)       # eigenvalues + eigenvectors
np.linalg.eigh(A)      # for symmetric/Hermitian — faster and stable

b = np.array([5, 11])
np.linalg.solve(A, b)  # [1. 2.] — solves Ax = b
```

### Statistical application: OLS

The ordinary-least-squares estimator $\hat{\boldsymbol\beta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ translates directly:

```python
rng = np.random.default_rng(0)
n, p = 50, 3
X = np.column_stack([np.ones(n), rng.standard_normal((n, p))])
beta_true = np.array([2, -1, 0.5, 3])
y = X @ beta_true + rng.standard_normal(n) * 0.5

beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
print(beta_hat.round(3))     # ≈ [2., -1., 0.5, 3.]
```

Use `np.linalg.solve(X.T @ X, X.T @ y)` rather than `np.linalg.inv(X.T @ X) @ X.T @ y` — solving is more numerically stable and faster than forming the inverse. Even better is `np.linalg.lstsq(X, y, rcond=None)`, which handles rank-deficient $\mathbf{X}$ via SVD.

## Reshaping and stacking

```python
a = np.arange(12)
M = a.reshape(3, 4)        # 3×4 view
flat = M.ravel()           # back to 1-D view

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
np.vstack([v1, v2])        # [[1 2 3], [4 5 6]]
np.hstack([v1, v2])        # [1 2 3 4 5 6]
np.column_stack([v1, v2])  # [[1 4], [2 5], [3 6]]
```

## Summary

| Concept | Key takeaway |
|---|---|
| `ndarray` | Homogeneous, fixed-size, $N$-dimensional contiguous block |
| Vectorization | Replaces explicit loops with array-level expressions |
| Broadcasting | Automatic shape expansion via two simple rules |
| Aggregation | `sum`, `mean`, `std`, `var` with an `axis` parameter; mind `ddof` |
| Linear algebra | `@` for matmul; prefer `solve` over `inv`; `eigh` for symmetric |
| Random generation | `default_rng(seed)` is the modern reproducible interface |

## Exercises

**Exercise 1.**
Verify the OLS normal-equation solution by hand and in NumPy for

$$
\mathbf{X} = \begin{pmatrix} 1 & 2 \\ 1 & 4 \\ 1 & 6 \end{pmatrix}, \qquad \mathbf{y} = (5, 9, 13)^T
$$

Compute $\mathbf{X}^T\mathbf{X}$, $\mathbf{X}^T\mathbf{y}$, $\hat{\boldsymbol\beta}$, and the residual vector.

??? success "Solution to Exercise 1"
    ```python
    import numpy as np

    X = np.array([[1, 2], [1, 4], [1, 6]])
    y = np.array([5, 9, 13])

    XtX = X.T @ X
    Xty = X.T @ y
    beta_hat = np.linalg.solve(XtX, Xty)
    y_hat = X @ beta_hat
    resid = y - y_hat

    print("X^T X =", XtX, sep="\n")
    print("X^T y =", Xty)
    print("beta_hat =", beta_hat)
    print("residuals =", resid)
    ```

    Expected: $\hat{\boldsymbol\beta} = (1, 2)^T$ and zero residuals (perfect collinearity in the three points).

---

**Exercise 2.**
Without a Python `for` loop, generate a $1000 \times 5$ matrix of standard normal draws and standardize each column to have sample mean 0 and sample variance 1. Verify with `mean(axis=0)` and `var(axis=0, ddof=1)`.

??? success "Solution to Exercise 2"
    ```python
    rng = np.random.default_rng(42)
    X = rng.standard_normal((1000, 5))
    X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    print(X_std.mean(axis=0).round(8))           # ≈ zeros
    print(X_std.var(axis=0, ddof=1).round(8))    # ≈ ones
    ```

    Broadcasting subtracts the row vector of column means (shape `(5,)`) from `X` (shape `(1000, 5)`) and likewise divides by the row vector of column standard deviations.

---

**Exercise 3.**
Write a vectorized function `pairwise_distances(X)` that returns the $n \times n$ matrix of Euclidean distances between rows of `X`. Use only broadcasting and `np.sqrt` — no explicit loops.

??? success "Solution to Exercise 3"
    ```python
    def pairwise_distances(X):
        # X has shape (n, p). diff has shape (n, n, p) after broadcasting:
        # X[:, None, :] is (n, 1, p), X[None, :, :] is (1, n, p)
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

    X = np.array([[0, 0], [3, 4], [6, 8]])
    print(pairwise_distances(X))
    # [[ 0.  5. 10.]
    #  [ 5.  0.  5.]
    #  [10.  5.  0.]]
    ```

    The two broadcast steps insert a singleton dimension that pairs each row with every other row. The result is symmetric with zero diagonal.

---

**Exercise 4.**
Why does `np.linalg.solve(A, b)` give more accurate results than `np.linalg.inv(A) @ b`? Construct an example with $\mathbf{A}$ near-singular and report the two answers.

??? success "Solution to Exercise 4"
    `solve` factors $\mathbf{A}$ once (LU with partial pivoting) and back-substitutes for $\mathbf{b}$; it never forms $\mathbf{A}^{-1}$ explicitly. Forming the inverse multiplies all entries by $1/\det(\mathbf{A})$, amplifying rounding error when $\det(\mathbf{A})$ is small. Backslash-style routines also avoid an unnecessary $O(n^3)$ matrix multiply.

    ```python
    A = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]])  # near-singular
    b = np.array([2.0, 2.0 + 1e-12])

    x_solve = np.linalg.solve(A, b)
    x_inv = np.linalg.inv(A) @ b
    print("solve:", x_solve)
    print("inv:  ", x_inv)
    ```

    Both should be close to $(1, 1)^T$, but `inv` exhibits visibly larger error. On truly singular matrices `solve` raises; `inv` may return garbage instead.

---

**Exercise 5.**
Two arrays have shapes $(5, 1)$ and $(1, 3)$. What is the shape of their elementwise product? What if the shapes were $(5,)$ and $(3,)$ — does it work?

??? success "Solution to Exercise 5"
    Following the broadcasting rules: align shapes on the right, expand dimensions of size 1. $(5, 1)$ and $(1, 3)$ broadcast to $(5, 3)$. Multiplication produces a $5 \times 3$ outer-product-style matrix.

    For $(5,)$ and $(3,)$: align right, prepend ones to make $(1, 5)$ and $(1, 3)$ — wait, **actually** they become $(5,)$ and $(3,)$ aligned at the last axis. Last-axis sizes are $5$ and $3$, neither equal nor 1, so broadcasting **fails** with `ValueError`. To produce a $5 \times 3$ outer product you must explicitly insert axes: `a[:, None] * b[None, :]` or `np.outer(a, b)`.

---

**Exercise 6.**
The default `np.var(x)` divides by $n$, while `np.var(x, ddof=1)` divides by $n - 1$. Which one is an unbiased estimator of $\mathrm{Var}(X)$ when $x$ is an i.i.d. sample? Demonstrate the bias empirically by drawing $10^4$ samples of size $n = 5$ from $N(0, 1)$ and comparing the average of `var(ddof=0)` and `var(ddof=1)` across replications.

??? success "Solution to Exercise 6"
    `ddof=1` is unbiased: $\mathbb{E}[S^2] = \sigma^2$ when dividing by $n - 1$. `ddof=0` underestimates $\sigma^2$ by a factor of $(n-1)/n$.

    ```python
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((10_000, 5))
    print("Mean of var(ddof=0):", samples.var(axis=1, ddof=0).mean())  # ≈ 0.80
    print("Mean of var(ddof=1):", samples.var(axis=1, ddof=1).mean())  # ≈ 1.00
    ```

    With $n = 5$, the population-style divisor produces $\approx 4/5 = 0.8$ on average — exactly the predicted bias factor. The Bessel-corrected version hovers around $1.0$ as expected. This bias matters most when $n$ is small; for $n$ in the thousands the difference is negligible.
