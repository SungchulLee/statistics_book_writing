# Positive Definite Matrices

A symmetric matrix is positive definite when the quadratic form $\mathbf{x}^T\mathbf{A}\mathbf{x}$ is strictly positive for every nonzero vector $\mathbf{x}$. This condition is the matrix analogue of a positive real number and guarantees that $\mathbf{A}$ is invertible, has a unique Cholesky factorization, and defines a genuine inner product. In statistics, positive definiteness is the criterion that separates well-posed covariance matrices (invertible, leading to finite-density multivariate normals) from degenerate ones. This section presents the definition, equivalent characterizations, and the Cholesky decomposition.

## Definition

!!! info "Definition -- Positive Definite and Positive Semi-Definite"
    A symmetric matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is:

    - **Positive definite** ($\mathbf{A} \succ 0$) if $\mathbf{x}^T\mathbf{A}\mathbf{x} > 0$ for every $\mathbf{x} \neq \mathbf{0}$.
    - **Positive semi-definite** ($\mathbf{A} \succeq 0$) if $\mathbf{x}^T\mathbf{A}\mathbf{x} \geq 0$ for every $\mathbf{x}$.
    - **Negative definite** ($\mathbf{A} \prec 0$) if $-\mathbf{A} \succ 0$.
    - **Indefinite** if $\mathbf{x}^T\mathbf{A}\mathbf{x}$ takes both positive and negative values.

!!! warning "Symmetry is assumed"
    Some authors define positive definiteness for non-symmetric matrices, but in this book (and in nearly all of statistics), positive definiteness always refers to symmetric matrices.

## Equivalent Characterizations

The following conditions are equivalent for a symmetric matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$.

!!! tip "Theorem -- Equivalent Conditions for Positive Definiteness"
    The following are equivalent:

    1. $\mathbf{A} \succ 0$ (the quadratic form condition).
    2. All eigenvalues of $\mathbf{A}$ are strictly positive: $\lambda_i > 0$ for $i = 1, \dots, n$.
    3. All **leading principal minors** are positive: $\det(\mathbf{A}_k) > 0$ for $k = 1, \dots, n$, where $\mathbf{A}_k$ is the upper-left $k \times k$ submatrix.
    4. $\mathbf{A}$ has a **Cholesky decomposition**: $\mathbf{A} = \mathbf{L}\mathbf{L}^T$ for a unique lower-triangular matrix $\mathbf{L}$ with positive diagonal entries.
    5. There exists an invertible matrix $\mathbf{B}$ such that $\mathbf{A} = \mathbf{B}^T\mathbf{B}$.

### Proof Sketch (Eigenvalue Characterization)

By the Spectral Theorem, $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ with orthogonal $\mathbf{Q}$. Setting $\mathbf{z} = \mathbf{Q}^T\mathbf{x}$ (a bijection since $\mathbf{Q}$ is orthogonal):

$$
\mathbf{x}^T\mathbf{A}\mathbf{x} = \mathbf{z}^T\boldsymbol{\Lambda}\mathbf{z} = \sum_{i=1}^n \lambda_i z_i^2
$$

This is positive for all $\mathbf{z} \neq \mathbf{0}$ if and only if every $\lambda_i > 0$. $\square$

For **positive semi-definiteness**, all conditions above relax to "$\geq 0$" for eigenvalues and leading minors, and condition (5) allows $\mathbf{B}$ to be rank-deficient.

## The Cholesky Decomposition

!!! info "Definition -- Cholesky Decomposition"
    The **Cholesky decomposition** of a positive definite matrix $\mathbf{A}$ is the unique factorization

    $$
    \mathbf{A} = \mathbf{L}\mathbf{L}^T
    $$

    where $\mathbf{L}$ is lower triangular with strictly positive diagonal entries.

The Cholesky decomposition is the matrix analogue of taking the square root of a positive number. It is numerically stable, requires roughly $n^3/3$ operations (half the cost of a general $\mathbf{LU}$ decomposition), and is the preferred method for solving linear systems involving positive definite matrices.

### Example

For the matrix

$$
\mathbf{A} = \begin{pmatrix} 4 & 2 \\ 2 & 5 \end{pmatrix}
$$

**Eigenvalue check:** The eigenvalues satisfy $\lambda^2 - 9\lambda + 16 = 0$, giving $\lambda = (9 \pm \sqrt{17})/2$. Both are positive (approximately 6.56 and 2.44), so $\mathbf{A} \succ 0$.

**Leading minors:** $\det(\mathbf{A}_1) = 4 > 0$ and $\det(\mathbf{A}_2) = 16 > 0$. Both positive, confirming positive definiteness.

**Cholesky decomposition:** Solve for $\mathbf{L}$:

$$
\begin{pmatrix} 4 & 2 \\ 2 & 5 \end{pmatrix} = \begin{pmatrix} l_{11} & 0 \\ l_{21} & l_{22} \end{pmatrix}\begin{pmatrix} l_{11} & l_{21} \\ 0 & l_{22} \end{pmatrix}
$$

From $l_{11}^2 = 4$: $l_{11} = 2$. From $l_{21}l_{11} = 2$: $l_{21} = 1$. From $l_{21}^2 + l_{22}^2 = 5$: $l_{22} = 2$. So

$$
\mathbf{L} = \begin{pmatrix} 2 & 0 \\ 1 & 2 \end{pmatrix}
$$

## Properties

### Positive Definite Matrices Are Invertible

If $\mathbf{A} \succ 0$, all eigenvalues are positive, so $\det(\mathbf{A}) = \prod_i \lambda_i > 0$. Therefore $\mathbf{A}$ is invertible, and $\mathbf{A}^{-1}$ is also positive definite (its eigenvalues are $1/\lambda_i > 0$).

### Sums and Scalar Multiples

If $\mathbf{A} \succ 0$ and $\mathbf{B} \succ 0$, then $\mathbf{A} + \mathbf{B} \succ 0$ and $c\mathbf{A} \succ 0$ for any $c > 0$. The set of positive definite matrices forms an open convex cone.

### Congruence Preserves Positive Definiteness

If $\mathbf{A} \succ 0$ and $\mathbf{B}$ is an $n \times m$ matrix with $\operatorname{rank}(\mathbf{B}) = m$, then $\mathbf{B}^T\mathbf{A}\mathbf{B} \succ 0$ (in $\mathbb{R}^{m \times m}$).

**Proof.** For any $\mathbf{y} \neq \mathbf{0}$ in $\mathbb{R}^m$, set $\mathbf{x} = \mathbf{B}\mathbf{y}$. Since $\mathbf{B}$ has full column rank, $\mathbf{x} \neq \mathbf{0}$, so $\mathbf{y}^T(\mathbf{B}^T\mathbf{A}\mathbf{B})\mathbf{y} = \mathbf{x}^T\mathbf{A}\mathbf{x} > 0$. $\square$

This result explains why $\mathbf{X}^T\mathbf{X}$ is positive definite whenever $\mathbf{X}$ has full column rank: it is the congruence of $\mathbf{I}_n \succ 0$ by $\mathbf{X}$.

### Schur Complement

If a symmetric positive definite matrix is partitioned as

$$
\mathbf{M} = \begin{pmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{B}^T & \mathbf{C} \end{pmatrix} \succ 0
$$

then the **Schur complement** $\mathbf{S} = \mathbf{C} - \mathbf{B}^T\mathbf{A}^{-1}\mathbf{B}$ is also positive definite. The Schur complement appears in the conditional variance of multivariate normal distributions.

## Connection to Statistics

### Covariance Matrices

The covariance matrix $\boldsymbol{\Sigma} = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$ is always positive semi-definite. It is positive definite if and only if no component of $\mathbf{X}$ is an exact linear combination of the others. When $\boldsymbol{\Sigma} \succ 0$, the multivariate normal density is well-defined:

$$
f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2}|\boldsymbol{\Sigma}|^{1/2}}\exp\!\Bigl(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\Bigr)
$$

The positive definiteness of $\boldsymbol{\Sigma}$ ensures that $|\boldsymbol{\Sigma}| > 0$ (the density is finite) and the exponent is always negative (the density decays in all directions).

### OLS Existence

The OLS estimator $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ requires $\mathbf{X}^T\mathbf{X}$ to be invertible. Since $\mathbf{X}^T\mathbf{X}$ is always positive semi-definite, it is invertible (positive definite) precisely when $\mathbf{X}$ has full column rank.

### Mahalanobis Distance

The Mahalanobis distance $d^2 = (\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$ is a quadratic form in $\boldsymbol{\Sigma}^{-1}$, which is positive definite when $\boldsymbol{\Sigma}$ is. This ensures $d^2 \geq 0$ with equality only at $\mathbf{x} = \boldsymbol{\mu}$, making it a genuine distance-like measure.

## Summary

Positive definiteness is the matrix property ensuring that a quadratic form is strictly positive, which translates to invertibility, well-defined densities, and unique least-squares solutions. The key equivalent characterizations -- positive eigenvalues, positive leading minors, and existence of a Cholesky factorization -- provide different computational and theoretical tools. In statistics, positive definiteness of $\boldsymbol{\Sigma}$ underlies the multivariate normal distribution, and positive definiteness of $\mathbf{X}^T\mathbf{X}$ guarantees the existence of OLS estimators.

## Exercises

**Exercise 1.**
Determine whether $\mathbf{A} = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$ is positive definite using the leading-minors criterion.

??? success "Solution to Exercise 1"
    The leading minors are:

    - First leading minor: $a_{11} = 4 > 0$
    - Second leading minor: $\det(\mathbf{A}) = 4 \times 3 - 2 \times 2 = 12 - 4 = 8 > 0$

    Since all leading minors are strictly positive, $\mathbf{A}$ is positive definite. Equivalently, the eigenvalues are $\lambda = \frac{7 \pm \sqrt{49 - 32}}{2} = \frac{7 \pm \sqrt{17}}{2}$, both positive.

---

**Exercise 2.**
Prove that if $\mathbf{A}$ is positive definite, then all diagonal entries $a_{ii} > 0$.

??? success "Solution to Exercise 2"
    Let $\mathbf{e}_i$ be the $i$-th standard basis vector ($1$ in position $i$, $0$ elsewhere). Since $\mathbf{e}_i \neq \mathbf{0}$ and $\mathbf{A}$ is positive definite:

    $$
    \mathbf{e}_i^T \mathbf{A} \mathbf{e}_i > 0
    $$

    But $\mathbf{e}_i^T \mathbf{A} \mathbf{e}_i = a_{ii}$, so $a_{ii} > 0$ for all $i$. Note that the converse is false: positive diagonal entries do not guarantee positive definiteness (e.g., $\begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$ has positive diagonal entries but eigenvalues $3$ and $-1$). $\square$

---

**Exercise 3.**
Find the Cholesky decomposition $\mathbf{A} = \mathbf{L}\mathbf{L}^T$ of $\mathbf{A} = \begin{pmatrix} 4 & 6 \\ 6 & 13 \end{pmatrix}$.

??? success "Solution to Exercise 3"
    We seek a lower-triangular $\mathbf{L} = \begin{pmatrix} l_{11} & 0 \\ l_{21} & l_{22} \end{pmatrix}$ such that $\mathbf{L}\mathbf{L}^T = \mathbf{A}$.

    From $l_{11}^2 = 4$: $l_{11} = 2$.

    From $l_{21} l_{11} = 6$: $l_{21} = 3$.

    From $l_{21}^2 + l_{22}^2 = 13$: $9 + l_{22}^2 = 13$, so $l_{22} = 2$.

    $$
    \mathbf{L} = \begin{pmatrix} 2 & 0 \\ 3 & 2 \end{pmatrix}
    $$

    Verification: $\mathbf{L}\mathbf{L}^T = \begin{pmatrix} 4 & 6 \\ 6 & 13 \end{pmatrix} = \mathbf{A}$.

---

**Exercise 4.**
A covariance matrix $\boldsymbol{\Sigma}$ has eigenvalues $\lambda_1 = 0.01$ and $\lambda_2 = 100$. Is $\boldsymbol{\Sigma}$ positive definite? Discuss the practical implications for computing $\boldsymbol{\Sigma}^{-1}$.

??? success "Solution to Exercise 4"
    Yes, $\boldsymbol{\Sigma}$ is positive definite because both eigenvalues are strictly positive. However, the condition number is $\kappa = \lambda_{\max}/\lambda_{\min} = 100/0.01 = 10{,}000$, which is very large.

    The practical implications are:

    - **Numerical instability:** Floating-point errors in computing $\boldsymbol{\Sigma}^{-1}$ are amplified by the condition number. The eigenvalue $1/\lambda_1 = 100$ of $\boldsymbol{\Sigma}^{-1}$ can be significantly corrupted.
    - **Near-singularity:** The data are nearly collinear in the direction of the smallest eigenvalue, meaning the two variables are nearly perfectly correlated.
    - **Remedies:** Use Cholesky decomposition rather than explicit inversion, apply regularization (ridge regression), or use the pseudo-inverse if appropriate.

---

**Exercise 5.**
Prove that if $\boldsymbol{\Sigma}$ is positive definite, the Mahalanobis distance $d^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$ equals zero if and only if $\mathbf{x} = \boldsymbol{\mu}$.

??? success "Solution to Exercise 5"
    Since $\boldsymbol{\Sigma}$ is positive definite, $\boldsymbol{\Sigma}^{-1}$ is also positive definite (its eigenvalues are $1/\lambda_i > 0$).

    Let $\mathbf{z} = \mathbf{x} - \boldsymbol{\mu}$. Then $d^2 = \mathbf{z}^T\boldsymbol{\Sigma}^{-1}\mathbf{z}$.

    By positive definiteness of $\boldsymbol{\Sigma}^{-1}$: $\mathbf{z}^T\boldsymbol{\Sigma}^{-1}\mathbf{z} \geq 0$ for all $\mathbf{z}$, with equality if and only if $\mathbf{z} = \mathbf{0}$.

    Therefore $d^2 = 0$ if and only if $\mathbf{x} - \boldsymbol{\mu} = \mathbf{0}$, i.e., $\mathbf{x} = \boldsymbol{\mu}$. This confirms that the Mahalanobis distance is a proper distance-like measure (satisfying definiteness). $\square$
