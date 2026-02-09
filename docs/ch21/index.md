# Chapter 21: Mathematical Backup for Linear Regression

This chapter provides the rigorous mathematical foundations underlying the confidence interval theory for linear regression. It is organized into two main sections.

## 21.1 Square Matrices

This section covers the essential linear algebra concepts needed for the statistical theory of linear regression:

- [**Similar Matrices**](square_matrices/similar_matrices.md) — Definition, intuition, and key invariant properties (eigenvalues, trace, determinant, rank).
- [**Diagonal Form of Diagonalizable Matrices**](square_matrices/diagonalizable.md) — Eigenvectors, eigenvalues, and eigendecomposition.
- [**Jordan Canonical Form**](square_matrices/jordan_form.md) — Generalized eigenvectors and Jordan block structure for non-diagonalizable matrices.
- [**Trace and Eigenvalues**](square_matrices/trace_eigenvalues.md) — Trace definition and proof that trace equals the sum of eigenvalues.
- [**Idempotent Matrices**](square_matrices/idempotent.md) — Properties, eigenvalue constraints, and the hat matrix in regression.
- [**Symmetric Matrices**](square_matrices/symmetric.md) — Spectral theorem, orthogonal diagonalization, and geometric interpretation.
- [**Positive Definite Matrices**](square_matrices/positive_definite.md) — Definition, invertibility, and connection to eigenvalues.
- [**Gram Matrices**](square_matrices/gram_matrices.md) — Rank identity, invertibility conditions, and role in least squares.
- [**Projection Matrices**](square_matrices/projection.md) — Definition of general projection matrices.
- [**Orthogonal Projection Matrices**](square_matrices/orthogonal_projection.md) — Characterization theorem, hat matrix, pseudo-inverse, and rank theorems.

## 21.2 Linear Algebra and Statistics

This section connects the linear algebra concepts to statistical distributions and OLS estimator theory:

- [**Chi-Squared Distribution and Quadratic Forms**](linalg_statistics/chi_squared_quadratic.md) — Distribution of quadratic forms involving symmetric idempotent matrices.
- [**Sampling Distributions for Simple OLS Estimators**](linalg_statistics/sampling_dist_simple_ols.md) — Distribution of SSE, expectation of SSE, and the unbiased estimator $s^2$ of $\sigma^2$.
- [**Sampling Distributions for General OLS Estimators**](linalg_statistics/sampling_dist_general_ols.md) — Independence of orthogonal projections and decomposition of Gaussian errors.
