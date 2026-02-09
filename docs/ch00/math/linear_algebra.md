# Linear Algebra Notation and Conventions

Linear algebra is the language of multivariate statistics. Regression, principal component analysis, ANOVA decompositions, and multivariate distributions all rely on matrix and vector operations. This section establishes the notation used throughout the book and reviews the essential results.

## Vectors

### Notation

Vectors are denoted by lowercase bold letters. Unless stated otherwise, all vectors are **column vectors**:

$$
\mathbf{x} = \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix} \in \mathbb{R}^n
$$

A row vector is written as the transpose $\mathbf{x}^T = (x_1, x_2, \dots, x_n)$.

### Basic Operations

| Operation | Notation | Result |
|---|---|---|
| Scalar multiplication | $c\mathbf{x}$ | $(cx_1, \dots, cx_n)^T$ |
| Addition | $\mathbf{x} + \mathbf{y}$ | $(x_1 + y_1, \dots, x_n + y_n)^T$ |
| Dot (inner) product | $\mathbf{x}^T \mathbf{y}$ | $\sum_{i=1}^n x_i y_i \in \mathbb{R}$ |
| Euclidean norm | $\lVert \mathbf{x} \rVert$ | $\sqrt{\mathbf{x}^T \mathbf{x}}$ |

### Statistical Interpretation

In data analysis, an observation vector $\mathbf{x}_i \in \mathbb{R}^p$ represents the $p$ measured features of the $i$-th observation. Stacking $n$ such observations row-wise produces the **design matrix** $\mathbf{X} \in \mathbb{R}^{n \times p}$.

## Matrices

### Notation

Matrices are denoted by uppercase bold letters:

$$
\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix} \in \mathbb{R}^{m \times n}
$$

The element in row $i$, column $j$ is $a_{ij}$ or $[\mathbf{A}]_{ij}$.

### Special Matrices

| Matrix | Notation | Definition |
|---|---|---|
| Identity | $\mathbf{I}_n$ | $[\mathbf{I}]_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$ |
| Zero matrix | $\mathbf{0}$ | All entries zero |
| Diagonal | $\text{diag}(d_1, \dots, d_n)$ | $a_{ij} = 0$ for $i \neq j$ |
| Symmetric | $\mathbf{A} = \mathbf{A}^T$ | $a_{ij} = a_{ji}$ |
| Ones vector | $\mathbf{1}_n$ | $(1, 1, \dots, 1)^T \in \mathbb{R}^n$ |

## Matrix Operations

### Multiplication

For $\mathbf{A} \in \mathbb{R}^{m \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times n}$:

$$
[\mathbf{AB}]_{ij} = \sum_{\ell=1}^{k} a_{i\ell}\, b_{\ell j}
$$

The result $\mathbf{AB} \in \mathbb{R}^{m \times n}$. Matrix multiplication is **not commutative** in general: $\mathbf{AB} \neq \mathbf{BA}$.

### Transpose

$$
[\mathbf{A}^T]_{ij} = a_{ji}
$$

**Properties:**

- $(\mathbf{A}^T)^T = \mathbf{A}$
- $(\mathbf{AB})^T = \mathbf{B}^T \mathbf{A}^T$
- $(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T$

### Trace

For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$:

$$
\text{tr}(\mathbf{A}) = \sum_{i=1}^n a_{ii}
$$

**Properties:**

- $\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})$
- $\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})$ (cyclic property)
- $\text{tr}(c\mathbf{A}) = c\,\text{tr}(\mathbf{A})$

The trace appears in expressions for the sum of squared residuals and in the expected value of quadratic forms.

### Determinant

The **determinant** $\det(\mathbf{A})$ or $|\mathbf{A}|$ is a scalar that encodes whether a matrix is invertible.

For a $2 \times 2$ matrix:

$$
\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc
$$

**Properties:**

- $\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$
- $\det(\mathbf{A}^T) = \det(\mathbf{A})$
- $\det(c\mathbf{A}) = c^n \det(\mathbf{A})$ for $\mathbf{A} \in \mathbb{R}^{n \times n}$
- $\mathbf{A}$ is invertible iff $\det(\mathbf{A}) \neq 0$

The determinant appears in the density of the multivariate normal distribution: $f(\mathbf{x}) \propto |\boldsymbol{\Sigma}|^{-1/2} \exp\!\bigl(-\tfrac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\bigr)$.

### Inverse

The **inverse** of a square matrix $\mathbf{A}$ (when it exists) satisfies:

$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}
$$

**Properties:**

- $(\mathbf{AB})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$
- $(\mathbf{A}^T)^{-1} = (\mathbf{A}^{-1})^T$

## Rank and Linear Independence

- A set of vectors $\{\mathbf{v}_1, \dots, \mathbf{v}_k\}$ is **linearly independent** if no vector can be written as a linear combination of the others.
- The **column rank** of $\mathbf{A}$ is the maximum number of linearly independent columns. For any matrix, column rank equals row rank, so we simply say **rank**.
- $\mathbf{A} \in \mathbb{R}^{n \times n}$ is **full rank** iff $\text{rank}(\mathbf{A}) = n$ iff $\mathbf{A}$ is invertible.

In regression, the design matrix $\mathbf{X}$ must have full column rank for the OLS estimator $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ to exist. Multicollinearity (Chapter 12) is precisely the situation where $\mathbf{X}$ is close to rank-deficient.

## Eigenvalues and Eigenvectors

For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, a scalar $\lambda$ and nonzero vector $\mathbf{v}$ satisfying

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}
$$

are an **eigenvalue–eigenvector pair**.

### Properties for Symmetric Matrices

When $\mathbf{A} = \mathbf{A}^T$ (which includes covariance matrices):

- All eigenvalues are real.
- Eigenvectors corresponding to distinct eigenvalues are orthogonal.
- $\mathbf{A}$ admits the **spectral decomposition**: $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$, where $\mathbf{Q}$ is orthogonal and $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_n)$.

### Connection to Statistics

- $\text{tr}(\mathbf{A}) = \sum_i \lambda_i$ and $\det(\mathbf{A}) = \prod_i \lambda_i$.
- A symmetric matrix is **positive definite** ($\mathbf{A} \succ 0$) iff all eigenvalues are strictly positive. Covariance matrices are positive semi-definite.
- Principal Component Analysis rotates data into the eigenvector basis of the sample covariance matrix.

## Positive Definite Matrices

A symmetric matrix $\mathbf{A}$ is:

| Type | Condition | Eigenvalues |
|---|---|---|
| Positive definite ($\mathbf{A} \succ 0$) | $\mathbf{x}^T \mathbf{A} \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$ | All $\lambda_i > 0$ |
| Positive semi-definite ($\mathbf{A} \succeq 0$) | $\mathbf{x}^T \mathbf{A} \mathbf{x} \geq 0$ for all $\mathbf{x}$ | All $\lambda_i \geq 0$ |

Covariance matrices $\boldsymbol{\Sigma}$ are always positive semi-definite. If no feature is a deterministic linear combination of others, $\boldsymbol{\Sigma}$ is positive definite and invertible.

## Quadratic Forms

A **quadratic form** associated with a symmetric matrix $\mathbf{A}$ is

$$
Q(\mathbf{x}) = \mathbf{x}^T \mathbf{A} \mathbf{x} = \sum_{i=1}^n \sum_{j=1}^n a_{ij} x_i x_j
$$

Quadratic forms appear frequently in statistics:

- **Sum of squares**: $\text{SSR} = (\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}})^T(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}})$
- **Mahalanobis distance**: $(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})$
- **Chi-square statistics**: sums of squared standardized residuals

## Projection Matrices

The **orthogonal projection** onto the column space of $\mathbf{X}$ (assuming full column rank) is

$$
\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T
$$

This is the **hat matrix** in regression, so named because $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$.

**Properties:**

- $\mathbf{H}^2 = \mathbf{H}$ (idempotent)
- $\mathbf{H}^T = \mathbf{H}$ (symmetric)
- $\text{tr}(\mathbf{H}) = p$ (number of parameters)
- $\mathbf{I} - \mathbf{H}$ projects onto the orthogonal complement (residual space)

## Matrix Calculus (Quick Reference)

The following derivative identities are used in deriving estimators:

| Expression | Derivative w.r.t. $\mathbf{x}$ |
|---|---|
| $\mathbf{a}^T \mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^T \mathbf{A} \mathbf{x}$ | $(\mathbf{A} + \mathbf{A}^T)\mathbf{x}$; if $\mathbf{A}$ symmetric: $2\mathbf{A}\mathbf{x}$ |
| $\mathbf{x}^T \mathbf{x}$ | $2\mathbf{x}$ |

**Deriving OLS:** Minimize $(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$ with respect to $\boldsymbol{\beta}$:

$$
\frac{\partial}{\partial \boldsymbol{\beta}}\bigl[\mathbf{y}^T\mathbf{y} - 2\boldsymbol{\beta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}\bigr] = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{0}
$$

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

## Notational Conventions Used in This Book

| Symbol | Meaning |
|---|---|
| $\mathbf{x}, \mathbf{y}, \boldsymbol{\beta}$ | Column vectors (bold lowercase) |
| $\mathbf{X}, \mathbf{A}, \boldsymbol{\Sigma}$ | Matrices (bold uppercase) |
| $x_i$, $a_{ij}$ | Scalar entries (plain lowercase) |
| $\mathbf{I}_n$ | $n \times n$ identity matrix |
| $\mathbf{1}_n$ | $n$-vector of ones |
| $\mathbf{0}$ | Zero vector or matrix (size from context) |
| $\mathbf{A}^T$ | Transpose |
| $\mathbf{A}^{-1}$ | Inverse |
| $\text{tr}(\mathbf{A})$ | Trace |
| $\det(\mathbf{A})$ or $\lvert\mathbf{A}\rvert$ | Determinant |
| $\text{diag}(\cdot)$ | Diagonal matrix |
| $\lVert \mathbf{x} \rVert$ | Euclidean ($\ell_2$) norm, unless otherwise specified |

## Summary

| Concept | Where It Appears |
|---|---|
| Vectors and dot products | Feature vectors, inner products in regression |
| Matrix multiplication | Design matrix operations, covariance computation |
| Inverse | OLS formula $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ |
| Eigendecomposition | PCA, spectral properties of covariance matrices |
| Positive definiteness | Covariance matrices, quadratic form positivity |
| Projection matrices | Hat matrix, residual decomposition in regression |
| Matrix calculus | Deriving MLE and least-squares estimators |
