# Diagonal Form of Diagonalizable Matrices

Among all the similarity transformations one can apply to a matrix, the most useful outcome is a diagonal matrix. A diagonalizable matrix can be factored into $\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$, which makes computing powers, exponentials, and quadratic forms straightforward. In statistics, covariance matrices are always diagonalizable (they are symmetric), and their diagonal form reveals principal components. This section defines diagonalizability, states the key existence theorem, and illustrates the computation.

## Definition

!!! info "Definition -- Diagonalizable Matrix"
    A square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is **diagonalizable** if it is similar to a diagonal matrix. That is, there exist an invertible matrix $\mathbf{P} \in \mathbb{R}^{n \times n}$ and a diagonal matrix $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ such that

    $$
    \mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}
    $$

    Equivalently, $\boldsymbol{\Lambda} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$.

The columns of $\mathbf{P}$ are the eigenvectors of $\mathbf{A}$, and the diagonal entries of $\boldsymbol{\Lambda}$ are the corresponding eigenvalues. Writing $\mathbf{P} = (\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_n)$, the decomposition $\mathbf{A}\mathbf{P} = \mathbf{P}\boldsymbol{\Lambda}$ is equivalent to $\mathbf{A}\mathbf{v}_i = \lambda_i \mathbf{v}_i$ for each $i$.

## When Is a Matrix Diagonalizable

!!! tip "Theorem -- Diagonalizability Criterion"
    A matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is diagonalizable if and only if it possesses $n$ linearly independent eigenvectors.

**Proof sketch.** If $\mathbf{A}$ has $n$ linearly independent eigenvectors $\mathbf{v}_1, \dots, \mathbf{v}_n$, place them as columns of $\mathbf{P}$. Then $\mathbf{P}$ is invertible (its columns are linearly independent), and by the eigenvalue relation,

$$
\mathbf{A}\mathbf{P} = \mathbf{A}(\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_n) = (\lambda_1\mathbf{v}_1 \mid \cdots \mid \lambda_n\mathbf{v}_n) = \mathbf{P}\boldsymbol{\Lambda}
$$

Multiplying both sides on the left by $\mathbf{P}^{-1}$ gives $\boldsymbol{\Lambda} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$.

Conversely, if $\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$, then $\mathbf{A}\mathbf{P} = \mathbf{P}\boldsymbol{\Lambda}$, so each column of $\mathbf{P}$ is an eigenvector of $\mathbf{A}$. Since $\mathbf{P}$ is invertible, these $n$ eigenvectors are linearly independent. $\square$

### Sufficient Conditions

Several important sufficient conditions guarantee diagonalizability:

- **Distinct eigenvalues.** If $\mathbf{A}$ has $n$ distinct eigenvalues, then the corresponding eigenvectors are linearly independent, so $\mathbf{A}$ is diagonalizable.
- **Symmetric matrices.** Every real symmetric matrix is diagonalizable (Spectral Theorem). Moreover, the eigenvectors can be chosen to be orthonormal, so $\mathbf{P}$ is orthogonal.
- **Algebraic multiplicity equals geometric multiplicity.** For each eigenvalue $\lambda_i$, the geometric multiplicity (dimension of the eigenspace) equals the algebraic multiplicity (multiplicity as a root of the characteristic polynomial).

## Powers and Exponentials

The diagonal form simplifies matrix powers dramatically:

$$
\mathbf{A}^k = \mathbf{P}\boldsymbol{\Lambda}^k\mathbf{P}^{-1} = \mathbf{P}\operatorname{diag}(\lambda_1^k, \dots, \lambda_n^k)\mathbf{P}^{-1}
$$

Similarly, the matrix exponential is

$$
e^{\mathbf{A}} = \mathbf{P}\operatorname{diag}(e^{\lambda_1}, \dots, e^{\lambda_n})\mathbf{P}^{-1}
$$

More generally, for any function $f$ that is defined on the eigenvalues:

$$
f(\mathbf{A}) = \mathbf{P}\operatorname{diag}\!\bigl(f(\lambda_1), \dots, f(\lambda_n)\bigr)\mathbf{P}^{-1}
$$

## Example -- A Diagonalizable Matrix

Consider

$$
\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}
$$

The characteristic polynomial is $\det(\mathbf{A} - \lambda\mathbf{I}) = (2 - \lambda)(3 - \lambda) = 0$, giving distinct eigenvalues $\lambda_1 = 2$ and $\lambda_2 = 3$.

For $\lambda_1 = 2$: $(\mathbf{A} - 2\mathbf{I})\mathbf{v} = \mathbf{0}$ gives $\mathbf{v}_1 = (1, 0)^T$.

For $\lambda_2 = 3$: $(\mathbf{A} - 3\mathbf{I})\mathbf{v} = \mathbf{0}$ gives $\mathbf{v}_2 = (1, 1)^T$.

Setting $\mathbf{P} = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$ and $\mathbf{P}^{-1} = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix}$, we verify

$$
\mathbf{P}^{-1}\mathbf{A}\mathbf{P} = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix} = \boldsymbol{\Lambda}
$$

Using this decomposition, $\mathbf{A}^{10} = \mathbf{P}\operatorname{diag}(2^{10}, 3^{10})\mathbf{P}^{-1} = \mathbf{P}\operatorname{diag}(1024, 59049)\mathbf{P}^{-1}$.

## Example -- A Non-Diagonalizable Matrix

The matrix

$$
\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}
$$

has a repeated eigenvalue $\lambda = 2$ with algebraic multiplicity 2, but the eigenspace $\ker(\mathbf{A} - 2\mathbf{I}) = \operatorname{span}\{(1, 0)^T\}$ has dimension 1 (geometric multiplicity 1). Since there is only one linearly independent eigenvector, $\mathbf{A}$ is not diagonalizable. Such matrices require the Jordan canonical form instead.

## Connection to Statistics

Diagonalization is the computational engine behind several core statistical methods:

- **Principal Component Analysis.** The sample covariance matrix $\mathbf{S}$ is symmetric and therefore diagonalizable: $\mathbf{S} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$. The columns of $\mathbf{Q}$ are the principal component directions, and $\boldsymbol{\Lambda}$ contains the variance explained by each component.

- **Quadratic forms.** If $\mathbf{A}$ is a symmetric matrix with eigendecomposition $\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$, then

$$
\mathbf{x}^T\mathbf{A}\mathbf{x} = \mathbf{z}^T\boldsymbol{\Lambda}\mathbf{z} = \sum_{i=1}^n \lambda_i z_i^2
$$

where $\mathbf{z} = \mathbf{Q}^T\mathbf{x}$. This decouples a quadratic form into a weighted sum of squares, which is essential for deriving chi-squared distributions.

- **Matrix inversion.** When $\boldsymbol{\Sigma} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ is positive definite, $\boldsymbol{\Sigma}^{-1} = \mathbf{Q}\boldsymbol{\Lambda}^{-1}\mathbf{Q}^T = \mathbf{Q}\operatorname{diag}(1/\lambda_1, \dots, 1/\lambda_n)\mathbf{Q}^T$, which is both computationally efficient and numerically stable.

## Summary

A matrix is diagonalizable when it has a full set of $n$ linearly independent eigenvectors, which allows the factorization $\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$. This decomposition reduces matrix operations to scalar operations on eigenvalues. Symmetric matrices -- including all covariance matrices -- are always diagonalizable, which makes eigendecomposition a fundamental tool in statistical theory. Matrices that fail to be diagonalizable require the Jordan canonical form, discussed next.
