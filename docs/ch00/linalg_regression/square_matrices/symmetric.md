# Symmetric Matrices

Symmetric matrices -- those satisfying $\mathbf{A} = \mathbf{A}^T$ -- are the most important class of matrices in statistics. Every covariance matrix is symmetric, every hat matrix is symmetric, and the normal equations in regression involve the symmetric matrix $\mathbf{X}^T\mathbf{X}$. The Spectral Theorem guarantees that symmetric matrices have real eigenvalues and an orthonormal eigenbasis, which makes them diagonalizable by an orthogonal transformation. This special structure underlies principal component analysis, the chi-squared distribution of quadratic forms, and the geometry of confidence ellipsoids.

## Definition

!!! info "Definition -- Symmetric Matrix"
    A square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is **symmetric** if

    $$
    \mathbf{A} = \mathbf{A}^T
    $$

    That is, $a_{ij} = a_{ji}$ for all $i, j$.

A symmetric matrix is completely determined by its entries on and above the main diagonal: there are $n(n+1)/2$ free entries rather than $n^2$.

## The Spectral Theorem

The Spectral Theorem is the single most important result about symmetric matrices.

!!! tip "Theorem -- Spectral Theorem (Real Symmetric Matrices)"
    Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be symmetric. Then:

    1. All eigenvalues of $\mathbf{A}$ are **real**.
    2. Eigenvectors corresponding to **distinct** eigenvalues are **orthogonal**.
    3. $\mathbf{A}$ admits the **spectral decomposition** (eigendecomposition):

    $$
    \mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T
    $$

    where $\mathbf{Q}$ is an orthogonal matrix ($\mathbf{Q}^T\mathbf{Q} = \mathbf{Q}\mathbf{Q}^T = \mathbf{I}$) whose columns are orthonormal eigenvectors, and $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ contains the eigenvalues.

### Proof Sketch (Eigenvalues Are Real)

Let $\lambda$ be a (possibly complex) eigenvalue with eigenvector $\mathbf{v} \neq \mathbf{0}$, so $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$. Taking the conjugate transpose of both sides:

$$
\bar{\mathbf{v}}^T\mathbf{A}^T = \bar{\lambda}\bar{\mathbf{v}}^T
$$

Since $\mathbf{A} = \mathbf{A}^T$ (and $\mathbf{A}$ is real, so $\bar{\mathbf{A}} = \mathbf{A}$):

$$
\bar{\mathbf{v}}^T\mathbf{A} = \bar{\lambda}\bar{\mathbf{v}}^T
$$

Multiply on the right by $\mathbf{v}$:

$$
\bar{\mathbf{v}}^T\mathbf{A}\mathbf{v} = \bar{\lambda}\bar{\mathbf{v}}^T\mathbf{v}
$$

But $\bar{\mathbf{v}}^T\mathbf{A}\mathbf{v} = \bar{\mathbf{v}}^T(\lambda\mathbf{v}) = \lambda\bar{\mathbf{v}}^T\mathbf{v}$. Therefore $\lambda\bar{\mathbf{v}}^T\mathbf{v} = \bar{\lambda}\bar{\mathbf{v}}^T\mathbf{v}$. Since $\bar{\mathbf{v}}^T\mathbf{v} = \lVert\mathbf{v}\rVert^2 > 0$, we conclude $\lambda = \bar{\lambda}$, so $\lambda$ is real. $\square$

### Proof Sketch (Orthogonality of Eigenvectors)

Let $\mathbf{A}\mathbf{u} = \alpha\mathbf{u}$ and $\mathbf{A}\mathbf{v} = \beta\mathbf{v}$ with $\alpha \neq \beta$. Then

$$
\alpha\,\mathbf{u}^T\mathbf{v} = (\mathbf{A}\mathbf{u})^T\mathbf{v} = \mathbf{u}^T\mathbf{A}^T\mathbf{v} = \mathbf{u}^T\mathbf{A}\mathbf{v} = \beta\,\mathbf{u}^T\mathbf{v}
$$

So $(\alpha - \beta)\mathbf{u}^T\mathbf{v} = 0$. Since $\alpha \neq \beta$, we must have $\mathbf{u}^T\mathbf{v} = 0$. $\square$

## Outer-Product Form of the Spectral Decomposition

Writing the spectral decomposition column-by-column gives the **outer-product form**:

$$
\mathbf{A} = \sum_{i=1}^n \lambda_i\,\mathbf{q}_i\mathbf{q}_i^T
$$

where $\mathbf{q}_1, \dots, \mathbf{q}_n$ are the orthonormal eigenvectors. Each term $\lambda_i\mathbf{q}_i\mathbf{q}_i^T$ is a rank-1 symmetric matrix that projects onto the eigenvector $\mathbf{q}_i$ and scales by $\lambda_i$.

## Properties of Symmetric Matrices

### Orthogonal Diagonalization

A matrix is orthogonally diagonalizable (i.e., $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ with $\mathbf{Q}$ orthogonal) if and only if it is symmetric. This is a stronger statement than mere diagonalizability.

### Inverse

If $\mathbf{A}$ is symmetric and invertible, then $\mathbf{A}^{-1}$ is also symmetric:

$$
(\mathbf{A}^{-1})^T = (\mathbf{A}^T)^{-1} = \mathbf{A}^{-1}
$$

Via the spectral decomposition: $\mathbf{A}^{-1} = \mathbf{Q}\boldsymbol{\Lambda}^{-1}\mathbf{Q}^T = \mathbf{Q}\operatorname{diag}(1/\lambda_1, \dots, 1/\lambda_n)\mathbf{Q}^T$.

### Powers and Functions

For any integer $k$ and any function $f$ defined on the eigenvalues:

$$
\mathbf{A}^k = \mathbf{Q}\boldsymbol{\Lambda}^k\mathbf{Q}^T, \qquad f(\mathbf{A}) = \mathbf{Q}\operatorname{diag}\!\bigl(f(\lambda_1), \dots, f(\lambda_n)\bigr)\mathbf{Q}^T
$$

In particular, the matrix square root $\mathbf{A}^{1/2} = \mathbf{Q}\operatorname{diag}(\sqrt{\lambda_1}, \dots, \sqrt{\lambda_n})\mathbf{Q}^T$ exists when all eigenvalues are nonnegative.

### Quadratic Forms

For a symmetric matrix $\mathbf{A}$, the quadratic form $\mathbf{x}^T\mathbf{A}\mathbf{x}$ can be diagonalized by the change of variables $\mathbf{z} = \mathbf{Q}^T\mathbf{x}$:

$$
\mathbf{x}^T\mathbf{A}\mathbf{x} = \mathbf{z}^T\boldsymbol{\Lambda}\mathbf{z} = \sum_{i=1}^n \lambda_i z_i^2
$$

This decomposition into a weighted sum of squares is the starting point for analyzing chi-squared distributions of quadratic forms.

## Example

Consider the covariance matrix

$$
\boldsymbol{\Sigma} = \begin{pmatrix} 5 & 2 \\ 2 & 2 \end{pmatrix}
$$

The characteristic polynomial is $\det(\boldsymbol{\Sigma} - \lambda\mathbf{I}) = (5-\lambda)(2-\lambda) - 4 = \lambda^2 - 7\lambda + 6 = (\lambda - 6)(\lambda - 1)$, giving eigenvalues $\lambda_1 = 6$ and $\lambda_2 = 1$.

For $\lambda_1 = 6$: $\mathbf{v}_1 = (2, 1)^T / \sqrt{5}$.

For $\lambda_2 = 1$: $\mathbf{v}_2 = (-1, 2)^T / \sqrt{5}$.

The orthogonal matrix and spectral decomposition are

$$
\mathbf{Q} = \frac{1}{\sqrt{5}}\begin{pmatrix} 2 & -1 \\ 1 & 2 \end{pmatrix}, \qquad \boldsymbol{\Sigma} = \mathbf{Q}\begin{pmatrix} 6 & 0 \\ 0 & 1 \end{pmatrix}\mathbf{Q}^T
$$

**Verification:** $\operatorname{tr}(\boldsymbol{\Sigma}) = 7 = 6 + 1$ and $\det(\boldsymbol{\Sigma}) = 6 = 6 \times 1$. The total variance is 7, with 6 units along the first principal direction and 1 along the second.

## Connection to Statistics

### Covariance Matrices

For a random vector $\mathbf{X} \in \mathbb{R}^p$ with mean $\boldsymbol{\mu}$, the covariance matrix $\boldsymbol{\Sigma} = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$ is always symmetric (and positive semi-definite). Its spectral decomposition $\boldsymbol{\Sigma} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ defines the principal component directions ($\mathbf{Q}$) and the variance along each direction ($\boldsymbol{\Lambda}$).

### Normal Equations

The OLS normal equations $\mathbf{X}^T\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$ involve the symmetric matrix $\mathbf{X}^T\mathbf{X}$. Its eigenvalues determine the numerical stability of the least-squares solution: when eigenvalues span many orders of magnitude (ill-conditioning), the solution is sensitive to perturbations in the data.

### Confidence Ellipsoids

The confidence region for a parameter vector $\boldsymbol{\beta}$ in multivariate normal theory is an ellipsoid $\{(\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})^T\boldsymbol{\Sigma}^{-1}(\boldsymbol{\beta} - \hat{\boldsymbol{\beta}}) \leq c\}$. The eigenvectors of $\boldsymbol{\Sigma}$ define the axes of the ellipsoid, and the eigenvalues determine their lengths.

## Summary

Symmetric matrices have real eigenvalues, orthogonal eigenvectors, and admit orthogonal diagonalization $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$. This structure makes them far better behaved than general matrices: powers, inverses, and functions all reduce to operations on eigenvalues. In statistics, covariance matrices, Gram matrices $\mathbf{X}^T\mathbf{X}$, and projection matrices are all symmetric, making the Spectral Theorem a foundational tool for principal component analysis, quadratic forms, and regression theory.
