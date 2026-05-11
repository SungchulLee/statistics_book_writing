# Gram Matrices

When a set of vectors is organized as columns of a matrix $\mathbf{X}$, the product $\mathbf{X}^T\mathbf{X}$ collects all pairwise inner products into a single matrix. This **Gram matrix** is always symmetric and positive semi-definite -- it encodes the geometry (angles and lengths) of the column vectors. In regression, the Gram matrix $\mathbf{X}^T\mathbf{X}$ appears in the normal equations, its invertibility determines whether OLS has a unique solution, and its eigenvalues control the numerical stability of the estimator. Understanding Gram matrices connects the algebraic conditions for regression existence to the geometric notion of linear independence among predictors.

## Definition

!!! info "Definition -- Gram Matrix"
    Let $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_p \in \mathbb{R}^n$ be a collection of vectors, and let $\mathbf{X} = (\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_p) \in \mathbb{R}^{n \times p}$ be the matrix with these vectors as columns. The **Gram matrix** is

    $$
    \mathbf{G} = \mathbf{X}^T\mathbf{X} \in \mathbb{R}^{p \times p}
    $$

    Its $(i,j)$ entry is the inner product $[\mathbf{G}]_{ij} = \mathbf{v}_i^T\mathbf{v}_j$.

The diagonal entries $g_{ii} = \mathbf{v}_i^T\mathbf{v}_i = \lVert\mathbf{v}_i\rVert^2$ are the squared lengths of the vectors, and the off-diagonal entries $g_{ij} = \mathbf{v}_i^T\mathbf{v}_j$ measure pairwise alignment.

## Symmetry and Positive Semi-Definiteness

!!! tip "Theorem -- Gram Matrices Are Symmetric Positive Semi-Definite"
    For any matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$, the Gram matrix $\mathbf{G} = \mathbf{X}^T\mathbf{X}$ is:

    1. **Symmetric**: $\mathbf{G}^T = (\mathbf{X}^T\mathbf{X})^T = \mathbf{X}^T\mathbf{X} = \mathbf{G}$
    2. **Positive semi-definite**: $\mathbf{y}^T\mathbf{G}\mathbf{y} \geq 0$ for all $\mathbf{y} \in \mathbb{R}^p$

**Proof of positive semi-definiteness.** For any $\mathbf{y} \in \mathbb{R}^p$:

$$
\mathbf{y}^T\mathbf{G}\mathbf{y} = \mathbf{y}^T\mathbf{X}^T\mathbf{X}\mathbf{y} = (\mathbf{X}\mathbf{y})^T(\mathbf{X}\mathbf{y}) = \lVert\mathbf{X}\mathbf{y}\rVert^2 \geq 0
$$

The quadratic form equals the squared Euclidean norm of $\mathbf{X}\mathbf{y}$, which is always nonneg. $\square$

## When Is the Gram Matrix Positive Definite

!!! tip "Theorem -- Positive Definiteness of the Gram Matrix"
    The Gram matrix $\mathbf{G} = \mathbf{X}^T\mathbf{X}$ is positive definite if and only if $\mathbf{X}$ has full column rank (i.e., $\operatorname{rank}(\mathbf{X}) = p$).

**Proof.** From the computation above, $\mathbf{y}^T\mathbf{G}\mathbf{y} = \lVert\mathbf{X}\mathbf{y}\rVert^2$. This is strictly positive for all $\mathbf{y} \neq \mathbf{0}$ if and only if $\mathbf{X}\mathbf{y} \neq \mathbf{0}$ for all $\mathbf{y} \neq \mathbf{0}$, which is equivalent to $\ker(\mathbf{X}) = \{\mathbf{0}\}$, i.e., $\mathbf{X}$ has full column rank. $\square$

**Consequence for regression.** The OLS estimator $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ exists and is unique if and only if $\mathbf{X}^T\mathbf{X}$ is positive definite, which happens if and only if the predictor columns are linearly independent.

## Eigenvalues of the Gram Matrix

Since $\mathbf{G} = \mathbf{X}^T\mathbf{X}$ is symmetric positive semi-definite, its eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0$ are all nonneg. These eigenvalues are the squares of the **singular values** of $\mathbf{X}$: if $\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$ is the singular value decomposition (SVD), then

$$
\mathbf{X}^T\mathbf{X} = \mathbf{V}\boldsymbol{\Sigma}^T\boldsymbol{\Sigma}\mathbf{V}^T = \mathbf{V}\operatorname{diag}(\sigma_1^2, \dots, \sigma_p^2)\mathbf{V}^T
$$

so $\lambda_i = \sigma_i^2$.

### Condition Number

The **condition number** of $\mathbf{X}^T\mathbf{X}$ is

$$
\kappa(\mathbf{X}^T\mathbf{X}) = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{\sigma_{\max}^2}{\sigma_{\min}^2}
$$

A large condition number indicates that $\mathbf{X}^T\mathbf{X}$ is nearly singular (the columns of $\mathbf{X}$ are nearly linearly dependent), making the OLS solution numerically unstable. This situation is called **multicollinearity**.

## Example

Let the design matrix have two predictor columns (ignoring the intercept for simplicity):

$$
\mathbf{X} = \begin{pmatrix} 1 & 2 \\ 3 & 1 \\ 2 & 4 \end{pmatrix}
$$

The Gram matrix is

$$
\mathbf{X}^T\mathbf{X} = \begin{pmatrix} 1 & 3 & 2 \\ 2 & 1 & 4 \end{pmatrix}\begin{pmatrix} 1 & 2 \\ 3 & 1 \\ 2 & 4 \end{pmatrix} = \begin{pmatrix} 14 & 13 \\ 13 & 21 \end{pmatrix}
$$

**Symmetry check:** The matrix is clearly symmetric ($g_{12} = g_{21} = 13$).

**Positive definiteness check:** $\det(\mathbf{X}^T\mathbf{X}) = 14 \cdot 21 - 13^2 = 294 - 169 = 125 > 0$ and $g_{11} = 14 > 0$, so $\mathbf{X}^T\mathbf{X} \succ 0$. This confirms that the two columns of $\mathbf{X}$ are linearly independent.

**Interpretation:** The entry $g_{11} = 14 = \lVert\mathbf{v}_1\rVert^2$ is the sum of squares of the first predictor. The entry $g_{12} = 13 = \mathbf{v}_1^T\mathbf{v}_2$ measures the alignment between the two predictors. If the predictors were orthogonal, this entry would be zero, and the Gram matrix would be diagonal.

## The Two Gram Matrices

For any $\mathbf{X} \in \mathbb{R}^{n \times p}$, there are actually two Gram matrices:

| Matrix | Size | Entries | Role in regression |
|---|---|---|---|
| $\mathbf{X}^T\mathbf{X}$ | $p \times p$ | Inner products of column (predictor) vectors | Normal equations |
| $\mathbf{X}\mathbf{X}^T$ | $n \times n$ | Inner products of row (observation) vectors | Hat matrix, kernel methods |

Both share the same nonzero eigenvalues (a general property of $\mathbf{A}^T\mathbf{A}$ and $\mathbf{A}\mathbf{A}^T$). When $n \gg p$, working with the $p \times p$ Gram matrix is far more efficient; when $p \gg n$ (high-dimensional settings), the $n \times n$ version is preferred (this is the "kernel trick").

## Gram Matrices and Orthogonality

The Gram matrix encodes the orthogonality structure of the column vectors:

- The columns of $\mathbf{X}$ are **orthogonal** if and only if $\mathbf{X}^T\mathbf{X}$ is diagonal.
- The columns of $\mathbf{X}$ are **orthonormal** if and only if $\mathbf{X}^T\mathbf{X} = \mathbf{I}$.

When the design matrix has orthogonal columns, the normal equations decouple and the OLS estimator simplifies to $\hat{\beta}_j = \mathbf{v}_j^T\mathbf{y}/\lVert\mathbf{v}_j\rVert^2$ for each predictor $j$ independently. This is why orthogonalizing predictors (e.g., via Gram-Schmidt or QR decomposition) simplifies regression computation and interpretation.

## Connection to Statistics

### Normal Equations

The OLS normal equations are $\mathbf{X}^T\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^T\mathbf{y}$. The Gram matrix $\mathbf{X}^T\mathbf{X}$ is the coefficient matrix of this linear system. Its positive definiteness (when $\mathbf{X}$ has full column rank) guarantees a unique solution.

### Variance of OLS Estimator

Under the standard linear model $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ with $\operatorname{Var}(\boldsymbol{\varepsilon}) = \sigma^2\mathbf{I}$:

$$
\operatorname{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}
$$

The eigenvalues of $(\mathbf{X}^T\mathbf{X})^{-1}$ are $1/\lambda_i$, so small eigenvalues of the Gram matrix produce large variances in the estimator. This is the algebraic explanation for why multicollinearity inflates standard errors.

### Sample Covariance Matrix

When $\mathbf{X}$ is the centered data matrix (observations minus column means), the sample covariance matrix is $\mathbf{S} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$, which is a scaled Gram matrix.

## Summary

The Gram matrix $\mathbf{X}^T\mathbf{X}$ collects all pairwise inner products of the columns of $\mathbf{X}$ into a symmetric positive semi-definite matrix. It is positive definite exactly when the columns are linearly independent, which is the condition for OLS to have a unique solution. The eigenvalues of the Gram matrix control the numerical stability of regression through the condition number, and its off-diagonal structure measures the correlation among predictors.

## Exercises

**Exercise 1.**
Let $\mathbf{X} = \begin{pmatrix} 1 & 2 \\ 1 & 3 \\ 1 & 5 \end{pmatrix}$. Compute the Gram matrix $\mathbf{X}^T\mathbf{X}$ and verify that it is symmetric and positive definite.

??? success "Solution to Exercise 1"
    $$
    \mathbf{X}^T\mathbf{X} = \begin{pmatrix} 1 & 1 & 1 \\ 2 & 3 & 5 \end{pmatrix}\begin{pmatrix} 1 & 2 \\ 1 & 3 \\ 1 & 5 \end{pmatrix} = \begin{pmatrix} 3 & 10 \\ 10 & 38 \end{pmatrix}
    $$

    Symmetry is immediate from $(\mathbf{X}^T\mathbf{X})^T = \mathbf{X}^T\mathbf{X}$. For positive definiteness, the leading minors are $3 > 0$ and $\det = 3 \times 38 - 10^2 = 114 - 100 = 14 > 0$, so $\mathbf{X}^T\mathbf{X}$ is positive definite. Equivalently, $\mathbf{X}$ has rank 2 (the two columns are linearly independent), so $\mathbf{X}^T\mathbf{X}$ is positive definite.

---

**Exercise 2.**
Prove that $\mathbf{X}^T\mathbf{X}$ is always positive semi-definite for any real matrix $\mathbf{X}$, and that it is positive definite if and only if $\mathbf{X}$ has full column rank.

??? success "Solution to Exercise 2"
    For any vector $\mathbf{v} \neq \mathbf{0}$:

    $$
    \mathbf{v}^T(\mathbf{X}^T\mathbf{X})\mathbf{v} = (\mathbf{X}\mathbf{v})^T(\mathbf{X}\mathbf{v}) = \lVert \mathbf{X}\mathbf{v} \rVert^2 \geq 0
    $$

    This is zero if and only if $\mathbf{X}\mathbf{v} = \mathbf{0}$, i.e., $\mathbf{v} \in \ker(\mathbf{X})$. If $\mathbf{X}$ has full column rank, $\ker(\mathbf{X}) = \{\mathbf{0}\}$, so $\mathbf{v}^T(\mathbf{X}^T\mathbf{X})\mathbf{v} > 0$ for all $\mathbf{v} \neq \mathbf{0}$, which is positive definiteness. Conversely, if $\mathbf{X}$ does not have full column rank, there exists $\mathbf{v} \neq \mathbf{0}$ with $\mathbf{X}\mathbf{v} = \mathbf{0}$, making the quadratic form zero. $\square$

---

**Exercise 3.**
If the columns of $\mathbf{X}$ are nearly collinear, explain how the condition number of $\mathbf{X}^T\mathbf{X}$ relates to the stability of OLS estimates. What is the condition number in terms of eigenvalues?

??? success "Solution to Exercise 3"
    The condition number of $\mathbf{X}^T\mathbf{X}$ is $\kappa = \lambda_{\max}/\lambda_{\min}$, where $\lambda_{\max}$ and $\lambda_{\min}$ are the largest and smallest eigenvalues.

    When columns are nearly collinear, $\lambda_{\min}$ is close to zero, making $\kappa$ very large. Since $\operatorname{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$ and the eigenvalues of $(\mathbf{X}^T\mathbf{X})^{-1}$ are $1/\lambda_i$, a small $\lambda_{\min}$ produces a large variance $\sigma^2/\lambda_{\min}$ in the direction of the corresponding eigenvector. A large condition number also means small perturbations to $\mathbf{y}$ cause large changes in $\hat{\boldsymbol{\beta}}$, making the estimates numerically unstable.

---

**Exercise 4.**
Show that the sample covariance matrix $\mathbf{S} = \frac{1}{n-1}\mathbf{X}_c^T\mathbf{X}_c$ (where $\mathbf{X}_c$ is the mean-centered data matrix) is positive semi-definite. Under what condition is it positive definite?

??? success "Solution to Exercise 4"
    Since $\mathbf{S} = \frac{1}{n-1}\mathbf{X}_c^T\mathbf{X}_c$ is a positive scalar multiple of a Gram matrix, it inherits positive semi-definiteness:

    $$
    \mathbf{v}^T\mathbf{S}\mathbf{v} = \frac{1}{n-1}\lVert \mathbf{X}_c \mathbf{v} \rVert^2 \geq 0
    $$

    $\mathbf{S}$ is positive definite if and only if $\mathbf{X}_c$ has full column rank, which requires $n - 1 \geq p$ (since centering reduces the rank by at most 1). In practice, this means we need more observations than variables ($n > p$) for the sample covariance matrix to be invertible. When $p > n$, $\mathbf{S}$ is singular and techniques like regularization or dimensionality reduction are needed. $\square$
