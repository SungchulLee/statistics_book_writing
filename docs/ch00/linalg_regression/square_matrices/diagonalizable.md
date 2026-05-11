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

## Exercises

**Exercise 1.**
Diagonalize the matrix $\mathbf{A} = \begin{pmatrix} 4 & 1 \\ 0 & 3 \end{pmatrix}$ by finding its eigenvalues, eigenvectors, and the matrices $\mathbf{P}$ and $\boldsymbol{\Lambda}$.

??? success "Solution to Exercise 1"
    The characteristic polynomial is $\det(\mathbf{A} - \lambda\mathbf{I}) = (4-\lambda)(3-\lambda) = 0$, giving eigenvalues $\lambda_1 = 4$ and $\lambda_2 = 3$.

    For $\lambda_1 = 4$: $(\mathbf{A} - 4\mathbf{I})\mathbf{v} = \begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix}\mathbf{v} = \mathbf{0}$, so $\mathbf{v}_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$.

    For $\lambda_2 = 3$: $(\mathbf{A} - 3\mathbf{I})\mathbf{v} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\mathbf{v} = \mathbf{0}$, so $\mathbf{v}_2 = \begin{pmatrix} -1 \\ 1 \end{pmatrix}$.

    Therefore:

    $$
    \mathbf{P} = \begin{pmatrix} 1 & -1 \\ 0 & 1 \end{pmatrix}, \quad \boldsymbol{\Lambda} = \begin{pmatrix} 4 & 0 \\ 0 & 3 \end{pmatrix}, \quad \mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}
    $$

---

**Exercise 2.**
Prove that if $\mathbf{A}$ is diagonalizable with $\mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1}$, then $\mathbf{A}^k = \mathbf{P}\boldsymbol{\Lambda}^k\mathbf{P}^{-1}$ for any positive integer $k$.

??? success "Solution to Exercise 2"
    We proceed by induction. The base case $k = 1$ holds by definition.

    Assume $\mathbf{A}^k = \mathbf{P}\boldsymbol{\Lambda}^k\mathbf{P}^{-1}$. Then:

    $$
    \mathbf{A}^{k+1} = \mathbf{A}^k \cdot \mathbf{A} = \mathbf{P}\boldsymbol{\Lambda}^k\mathbf{P}^{-1} \cdot \mathbf{P}\boldsymbol{\Lambda}\mathbf{P}^{-1} = \mathbf{P}\boldsymbol{\Lambda}^k\boldsymbol{\Lambda}\mathbf{P}^{-1} = \mathbf{P}\boldsymbol{\Lambda}^{k+1}\mathbf{P}^{-1}
    $$

    The key cancellation is $\mathbf{P}^{-1}\mathbf{P} = \mathbf{I}$. Since $\boldsymbol{\Lambda}^k = \operatorname{diag}(\lambda_1^k, \dots, \lambda_n^k)$, computing matrix powers reduces to computing scalar powers of eigenvalues. $\square$

---

**Exercise 3.**
Let $\boldsymbol{\Sigma}$ be a $2 \times 2$ covariance matrix with eigenvalues $\lambda_1 = 5$ and $\lambda_2 = 2$. Without computing $\boldsymbol{\Sigma}$ explicitly, find $\operatorname{tr}(\boldsymbol{\Sigma})$, $\det(\boldsymbol{\Sigma})$, and the eigenvalues of $\boldsymbol{\Sigma}^{-1}$.

??? success "Solution to Exercise 3"
    Since the trace is the sum of eigenvalues:

    $$
    \operatorname{tr}(\boldsymbol{\Sigma}) = \lambda_1 + \lambda_2 = 5 + 2 = 7
    $$

    Since the determinant is the product of eigenvalues:

    $$
    \det(\boldsymbol{\Sigma}) = \lambda_1 \cdot \lambda_2 = 5 \times 2 = 10
    $$

    The eigenvalues of $\boldsymbol{\Sigma}^{-1}$ are the reciprocals of the eigenvalues of $\boldsymbol{\Sigma}$:

    $$
    \lambda_1(\boldsymbol{\Sigma}^{-1}) = \frac{1}{5} = 0.2, \quad \lambda_2(\boldsymbol{\Sigma}^{-1}) = \frac{1}{2} = 0.5
    $$

---

**Exercise 4.**
Give an example of a $2 \times 2$ real matrix that is not diagonalizable. Prove that it cannot be diagonalized by showing it has fewer than two linearly independent eigenvectors.

??? success "Solution to Exercise 4"
    Consider $\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}$. The characteristic polynomial is $(2 - \lambda)^2 = 0$, so $\lambda = 2$ is the only eigenvalue (with algebraic multiplicity 2).

    The eigenspace for $\lambda = 2$ is the null space of:

    $$
    \mathbf{A} - 2\mathbf{I} = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}
    $$

    This has rank 1, so the null space has dimension 1 (geometric multiplicity = 1). The only eigenvector (up to scaling) is $\mathbf{v} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$.

    Since we need 2 linearly independent eigenvectors to form $\mathbf{P}$ but only have 1, the matrix is not diagonalizable. $\square$

---

**Exercise 5.**
Explain why every real symmetric matrix is diagonalizable and why the diagonalizing matrix can be chosen to be orthogonal. Why is this property important for covariance matrices?

??? success "Solution to Exercise 5"
    The Spectral Theorem guarantees that every real symmetric matrix has $n$ real eigenvalues (counting multiplicity) and a full set of $n$ orthonormal eigenvectors. Specifically, eigenvectors corresponding to distinct eigenvalues are orthogonal, and for repeated eigenvalues, the eigenspace can be orthonormalized via Gram-Schmidt. Arranging these eigenvectors as columns of $\mathbf{Q}$ gives an orthogonal matrix ($\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$), so $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$.

    For covariance matrices $\boldsymbol{\Sigma}$, this spectral decomposition is the foundation of Principal Component Analysis (PCA). The eigenvectors give the principal component directions, the eigenvalues give the variance explained by each component, and the orthogonality of $\mathbf{Q}$ means the principal components are uncorrelated. The decomposition also simplifies computation: $\boldsymbol{\Sigma}^{-1} = \mathbf{Q}\boldsymbol{\Lambda}^{-1}\mathbf{Q}^T$ and $\boldsymbol{\Sigma}^{1/2} = \mathbf{Q}\boldsymbol{\Lambda}^{1/2}\mathbf{Q}^T$.
