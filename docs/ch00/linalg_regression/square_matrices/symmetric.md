# Symmetric Matrices

Symmetric matrices — those satisfying $\mathbf{A} = \mathbf{A}^T$ — are the single most important class of matrices in statistics. Every covariance matrix is symmetric. Every hat matrix is symmetric. The Gram matrix $\mathbf{X}^T\mathbf{X}$ that appears in the normal equations is symmetric. The Spectral Theorem guarantees that symmetric matrices have real eigenvalues and an orthonormal eigenbasis, which makes them diagonalizable by an orthogonal transformation. This special structure underlies principal component analysis, the chi-squared distribution of quadratic forms, and the geometry of confidence ellipsoids.

## Definition

!!! info "Definition — Symmetric Matrix"
    A square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is **symmetric** if

    $$
    \mathbf{A} = \mathbf{A}^T
    $$

    Equivalently, $a_{ij} = a_{ji}$ for all $i, j$.

A symmetric matrix is determined by its on-or-above-diagonal entries: only $n(n+1)/2$ of its $n^2$ entries are free.

## The Spectral Theorem

The Spectral Theorem is the single most important result about symmetric matrices.

!!! tip "Theorem — Spectral Theorem (Real Symmetric Matrices)"
    Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be symmetric. Then:

    1. All eigenvalues of $\mathbf{A}$ are **real**.
    2. Eigenvectors corresponding to **distinct** eigenvalues are **orthogonal**.
    3. $\mathbf{A}$ admits the **spectral decomposition**

    $$
    \mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T
    $$

    where $\mathbf{Q}$ is orthogonal ($\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$) with columns the orthonormal eigenvectors, and $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$.

### Proof sketch (eigenvalues are real)

Let $\lambda \in \mathbb{C}$ be an eigenvalue with eigenvector $\mathbf{v} \ne \mathbf{0}$. Then $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$. Taking conjugate transpose with $\mathbf{A}$ real symmetric:

$$
\overline{\mathbf{v}}^T \mathbf{A} = \overline{\lambda}\, \overline{\mathbf{v}}^T
$$

Multiply on the right by $\mathbf{v}$:

$$
\lambda\, \overline{\mathbf{v}}^T \mathbf{v} = \overline{\mathbf{v}}^T \mathbf{A}\mathbf{v} = \overline{\lambda}\, \overline{\mathbf{v}}^T \mathbf{v}
$$

Since $\overline{\mathbf{v}}^T \mathbf{v} = \|\mathbf{v}\|^2 > 0$, we conclude $\lambda = \overline{\lambda}$, so $\lambda \in \mathbb{R}$. $\square$

### Proof sketch (orthogonal eigenvectors)

Let $\mathbf{A}\mathbf{u} = \alpha\mathbf{u}$ and $\mathbf{A}\mathbf{v} = \beta\mathbf{v}$ with $\alpha \ne \beta$. Then

$$
\alpha\, \mathbf{u}^T\mathbf{v} = (\mathbf{A}\mathbf{u})^T \mathbf{v} = \mathbf{u}^T \mathbf{A}^T \mathbf{v} = \mathbf{u}^T \mathbf{A}\mathbf{v} = \beta\, \mathbf{u}^T \mathbf{v}
$$

Hence $(\alpha - \beta)\mathbf{u}^T\mathbf{v} = 0$, and since $\alpha \ne \beta$, $\mathbf{u}^T\mathbf{v} = 0$. $\square$

For repeated eigenvalues, Gram–Schmidt within each eigenspace produces an orthonormal basis. Stitching these bases together gives $\mathbf{Q}$.

## Outer-product form

Writing the spectral decomposition column-by-column yields the **outer-product form**:

$$
\mathbf{A} = \sum_{i=1}^n \lambda_i\, \mathbf{q}_i \mathbf{q}_i^T
$$

Each $\mathbf{q}_i \mathbf{q}_i^T$ is a rank-1 orthogonal projector onto $\mathbf{q}_i$. The symmetric matrix is built from one-dimensional pieces, weighted by eigenvalues — the same idea that, in PCA, presents a covariance matrix as a sum over principal components.

## Properties

### Orthogonal diagonalization

A real matrix is **orthogonally** diagonalizable ($\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ with $\mathbf{Q}$ orthogonal) if and only if it is symmetric. This is strictly stronger than diagonalizability — many non-symmetric matrices are diagonalizable, but only symmetric matrices admit an *orthogonal* diagonalization.

### Inverse and matrix functions

If $\mathbf{A}$ is symmetric and invertible, $\mathbf{A}^{-1}$ is symmetric: $(\mathbf{A}^{-1})^T = (\mathbf{A}^T)^{-1} = \mathbf{A}^{-1}$. Via spectral decomposition,

$$
\mathbf{A}^k = \mathbf{Q}\boldsymbol{\Lambda}^k\mathbf{Q}^T, \quad f(\mathbf{A}) = \mathbf{Q}\operatorname{diag}\!\bigl(f(\lambda_1), \dots, f(\lambda_n)\bigr)\mathbf{Q}^T
$$

In particular, when all $\lambda_i \ge 0$, $\mathbf{A}^{1/2} = \mathbf{Q}\operatorname{diag}(\sqrt{\lambda_1}, \dots, \sqrt{\lambda_n})\mathbf{Q}^T$ is the unique symmetric PSD square root.

### Rayleigh quotients

For symmetric $\mathbf{A}$ with eigenvalues $\lambda_1 \le \cdots \le \lambda_n$, the **Rayleigh quotient** $R(\mathbf{x}) = \mathbf{x}^T \mathbf{A}\mathbf{x} / \mathbf{x}^T\mathbf{x}$ satisfies

$$
\lambda_1 = \min_{\mathbf{x} \ne \mathbf{0}} R(\mathbf{x}), \qquad \lambda_n = \max_{\mathbf{x} \ne \mathbf{0}} R(\mathbf{x})
$$

with extrema attained at the corresponding eigenvectors. This is the variational characterization that drives principal-component derivations.

### Quadratic forms

The change of variable $\mathbf{z} = \mathbf{Q}^T \mathbf{x}$ diagonalizes any quadratic form:

$$
\mathbf{x}^T \mathbf{A}\mathbf{x} = \mathbf{z}^T \boldsymbol{\Lambda}\mathbf{z} = \sum_{i=1}^n \lambda_i z_i^2
$$

A weighted sum of squares of independent rotated coordinates — the bridge to chi-squared distributions of quadratic forms in normal vectors.

## Example

$$
\boldsymbol{\Sigma} = \begin{pmatrix} 5 & 2 \\ 2 & 2 \end{pmatrix}
$$

Characteristic polynomial: $(5 - \lambda)(2 - \lambda) - 4 = \lambda^2 - 7\lambda + 6 = (\lambda - 6)(\lambda - 1)$, so $\lambda_1 = 6$ and $\lambda_2 = 1$.

Normalized eigenvectors: $\mathbf{q}_1 = (2, 1)^T / \sqrt{5}$ and $\mathbf{q}_2 = (-1, 2)^T / \sqrt{5}$.

Spectral decomposition:

$$
\boldsymbol{\Sigma} = \frac{1}{5}\begin{pmatrix} 2 & -1 \\ 1 & 2 \end{pmatrix}\begin{pmatrix} 6 & 0 \\ 0 & 1 \end{pmatrix}\begin{pmatrix} 2 & 1 \\ -1 & 2 \end{pmatrix}
$$

Sanity checks: $\operatorname{tr}(\boldsymbol{\Sigma}) = 7 = 6 + 1$ and $\det(\boldsymbol{\Sigma}) = 6 = 6 \cdot 1$. The total variance is 7; six units of it concentrate along the first principal axis, one along the second.

## Connection to Statistics

### Covariance matrices

For a random vector $\mathbf{X} \in \mathbb{R}^p$, $\boldsymbol{\Sigma} = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$ is symmetric (and PSD). Its spectral decomposition defines principal-component directions and the variance in each direction.

### Normal equations

$\mathbf{X}^T \mathbf{X}$ is symmetric. Its eigenvalues control the conditioning of OLS: when they span many orders of magnitude (multicollinearity), the solution is sensitive to perturbations in $\mathbf{y}$.

### Confidence ellipsoids

For $\boldsymbol{\beta} \sim N(\hat{\boldsymbol{\beta}}, \boldsymbol{\Sigma})$, the level sets $\{\boldsymbol{\beta} : (\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})^T \boldsymbol{\Sigma}^{-1} (\boldsymbol{\beta} - \hat{\boldsymbol{\beta}}) \le c\}$ are ellipsoids whose axes point along eigenvectors of $\boldsymbol{\Sigma}$ and whose lengths scale with $\sqrt{\lambda_i}$.

## Summary

Symmetric matrices have real eigenvalues, orthogonal eigenvectors, and admit orthogonal diagonalization $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$. This structure reduces powers, inverses, and functions of $\mathbf{A}$ to scalar operations on eigenvalues. Covariance matrices, Gram matrices, and projection matrices are all symmetric, making the Spectral Theorem the workhorse for principal component analysis, quadratic forms, and regression theory.

## Exercises

**Exercise 1.**
Find the eigenvalues, eigenvectors, and spectral decomposition of $\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$. Verify $\boldsymbol{\Sigma} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ numerically.

??? success "Solution to Exercise 1"
    Characteristic equation: $(2 - \lambda)^2 - 1 = \lambda^2 - 4\lambda + 3 = (\lambda - 1)(\lambda - 3) = 0$. Eigenvalues $\lambda_1 = 1, \lambda_2 = 3$.

    For $\lambda_1 = 1$: $(\mathbf{A} - \mathbf{I})\mathbf{v} = \mathbf{0}$ gives $\mathbf{v}_1 = (1, -1)^T / \sqrt{2}$.
    For $\lambda_2 = 3$: $(\mathbf{A} - 3\mathbf{I})\mathbf{v} = \mathbf{0}$ gives $\mathbf{v}_2 = (1, 1)^T / \sqrt{2}$.

    $$
    \mathbf{A} = \frac{1}{2}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & 3 \end{pmatrix}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}
    $$

---

**Exercise 2.**
Prove that for a symmetric matrix $\mathbf{A}$, eigenvectors corresponding to distinct eigenvalues are orthogonal.

??? success "Solution to Exercise 2"
    Suppose $\mathbf{A}\mathbf{u} = \alpha\mathbf{u}$ and $\mathbf{A}\mathbf{v} = \beta\mathbf{v}$ with $\alpha \ne \beta$. Compute $\mathbf{u}^T\mathbf{A}\mathbf{v}$ in two ways:

    - As $\mathbf{u}^T(\beta\mathbf{v}) = \beta\, \mathbf{u}^T\mathbf{v}$.
    - As $(\mathbf{A}\mathbf{u})^T \mathbf{v} = (\alpha\mathbf{u})^T\mathbf{v} = \alpha\, \mathbf{u}^T\mathbf{v}$, using $\mathbf{A} = \mathbf{A}^T$.

    Therefore $\alpha\, \mathbf{u}^T\mathbf{v} = \beta\, \mathbf{u}^T\mathbf{v}$, i.e. $(\alpha - \beta)\mathbf{u}^T\mathbf{v} = 0$. With $\alpha \ne \beta$, $\mathbf{u}^T\mathbf{v} = 0$. $\square$

---

**Exercise 3.**
Let $\mathbf{A}$ be symmetric with spectral decomposition $\mathbf{A} = \sum_i \lambda_i \mathbf{q}_i \mathbf{q}_i^T$. Show that $\mathbf{A}^2 = \sum_i \lambda_i^2 \mathbf{q}_i \mathbf{q}_i^T$, and explain why this confirms $\mathbf{A}$ is idempotent iff every eigenvalue is 0 or 1.

??? success "Solution to Exercise 3"
    Using orthonormality $\mathbf{q}_i^T \mathbf{q}_j = \delta_{ij}$:

    $$
    \mathbf{A}^2 = \Bigl(\sum_i \lambda_i \mathbf{q}_i \mathbf{q}_i^T\Bigr)\Bigl(\sum_j \lambda_j \mathbf{q}_j \mathbf{q}_j^T\Bigr) = \sum_{i,j} \lambda_i \lambda_j (\mathbf{q}_i^T \mathbf{q}_j)\mathbf{q}_i \mathbf{q}_j^T = \sum_i \lambda_i^2 \mathbf{q}_i \mathbf{q}_i^T
    $$

    $\mathbf{A}$ is idempotent iff $\mathbf{A}^2 = \mathbf{A}$, i.e. $\lambda_i^2 = \lambda_i$ for every $i$. Equivalently $\lambda_i \in \{0, 1\}$. $\square$

---

**Exercise 4.**
**Rayleigh quotient.** Let $\mathbf{A}$ be symmetric with smallest and largest eigenvalues $\lambda_\min, \lambda_\max$. Prove

$$
\lambda_\min \le \frac{\mathbf{x}^T\mathbf{A}\mathbf{x}}{\mathbf{x}^T\mathbf{x}} \le \lambda_\max
$$

for every nonzero $\mathbf{x} \in \mathbb{R}^n$.

??? success "Solution to Exercise 4"
    Expand $\mathbf{x}$ in the eigenbasis: $\mathbf{x} = \sum_i c_i \mathbf{q}_i$ with $c_i = \mathbf{q}_i^T \mathbf{x}$. Using orthonormality,

    $$
    \mathbf{x}^T\mathbf{A}\mathbf{x} = \sum_i \lambda_i c_i^2, \qquad \mathbf{x}^T\mathbf{x} = \sum_i c_i^2
    $$

    The Rayleigh quotient is therefore a convex combination of eigenvalues with weights $c_i^2 / \sum_j c_j^2$. Any convex combination of numbers lies between their min and max:

    $$
    \lambda_\min = \lambda_\min \sum_i \frac{c_i^2}{\sum_j c_j^2} \le \sum_i \lambda_i \frac{c_i^2}{\sum_j c_j^2} \le \lambda_\max
    $$

    Equality at $\lambda_\min$ occurs when $\mathbf{x}$ is an eigenvector of $\lambda_\min$, and similarly for $\lambda_\max$. $\square$

---

**Exercise 5.**
Show that every symmetric positive semi-definite matrix $\mathbf{A}$ has a unique symmetric PSD square root $\mathbf{A}^{1/2}$ satisfying $\mathbf{A}^{1/2} \mathbf{A}^{1/2} = \mathbf{A}$.

??? success "Solution to Exercise 5"
    **Existence:** From the spectral decomposition $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ with $\lambda_i \ge 0$, define

    $$
    \mathbf{A}^{1/2} := \mathbf{Q}\operatorname{diag}(\sqrt{\lambda_1}, \dots, \sqrt{\lambda_n})\mathbf{Q}^T
    $$

    This is symmetric (product $\mathbf{Q}\mathbf{D}\mathbf{Q}^T$ of symmetric pieces) and PSD ($\sqrt{\lambda_i} \ge 0$). Direct check: $\mathbf{A}^{1/2}\mathbf{A}^{1/2} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T = \mathbf{A}$.

    **Uniqueness:** suppose $\mathbf{B}$ is symmetric PSD with $\mathbf{B}^2 = \mathbf{A}$. Diagonalize $\mathbf{B} = \mathbf{Q}'\mathbf{M}\mathbf{Q}'^T$ with $\mathbf{M} = \operatorname{diag}(\mu_i)$, $\mu_i \ge 0$. Then $\mathbf{B}^2 = \mathbf{Q}'\mathbf{M}^2 \mathbf{Q}'^T = \mathbf{A}$, so $\mathbf{B}$ shares eigenvalues $\mu_i^2 = \lambda_i$ and eigenvectors of $\mathbf{A}$. With $\mu_i \ge 0$ this forces $\mu_i = \sqrt{\lambda_i}$, recovering the formula above. $\square$

    Statistical use: the **Mahalanobis whitening transform** $\mathbf{Z} = \boldsymbol{\Sigma}^{-1/2}(\mathbf{X} - \boldsymbol{\mu})$ produces a vector with identity covariance whenever $\boldsymbol{\Sigma}^{-1/2}$ exists.

---

**Exercise 6.**
Give an example of a non-symmetric matrix that is nevertheless diagonalizable (so symmetry is sufficient but not necessary for diagonalizability), and one example of a symmetric matrix where applying Gram–Schmidt to eigenvectors of a *repeated* eigenvalue produces an explicit orthonormal basis.

??? success "Solution to Exercise 6"
    **Non-symmetric but diagonalizable:**

    $$
    \mathbf{A} = \begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix}
    $$

    Eigenvalues $1, 2$ are distinct, so eigenvectors are linearly independent and $\mathbf{A}$ is diagonalizable. But $\mathbf{A} \ne \mathbf{A}^T$ — symmetry fails.

    **Symmetric with a repeated eigenvalue:**

    $$
    \mathbf{A} = \begin{pmatrix} 2 & 0 & 0 \\ 0 & 1 & 1 \\ 0 & 1 & 1 \end{pmatrix}
    $$

    Eigenvalues: $\lambda_1 = 2$ (eigenvector $(1,0,0)^T$), $\lambda_2 = 2$ (eigenvector $(0,1,1)^T/\sqrt{2}$), and $\lambda_3 = 0$ (eigenvector $(0,1,-1)^T/\sqrt{2}$).

    The eigenvalue $2$ has multiplicity 2 with eigenspace $\operatorname{span}\{(1,0,0)^T, (0,1,1)^T\}$. These two vectors are already orthogonal (no Gram–Schmidt needed); normalizing gives an orthonormal basis. Stacking with the third eigenvector produces the orthogonal $\mathbf{Q}$ promised by the Spectral Theorem.
