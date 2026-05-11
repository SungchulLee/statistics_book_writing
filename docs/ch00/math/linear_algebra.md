# Linear Algebra Notation and Conventions

Linear algebra is the language of multivariate statistics. Almost every quantity in regression, multivariate analysis, dimensionality reduction, and the modern theory of estimation is a vector or matrix expression. This section establishes the notation used throughout the book and reviews the operations and identities that recur most often.

## Definition

### Vectors and matrices

Vectors are column vectors $\mathbf{x} \in \mathbb{R}^n$ written in bold lowercase; matrices are $\mathbf{A} \in \mathbb{R}^{m \times n}$ written in bold uppercase. We use $\mathbf{A}^T$ for transpose and $\mathbf{A}^{-1}$ for inverse (when it exists). The identity matrix is $\mathbf{I}_n$ and the zero matrix is $\mathbf{0}$.

### Core operations

$$
\mathbf{x}^T \mathbf{y} = \sum_{i=1}^n x_i y_i, \qquad \|\mathbf{x}\| = \sqrt{\mathbf{x}^T \mathbf{x}}, \qquad [\mathbf{A}\mathbf{B}]_{ij} = \sum_{\ell} a_{i\ell} b_{\ell j}
$$

The **outer product** $\mathbf{x} \mathbf{y}^T \in \mathbb{R}^{n \times m}$ has rank at most 1. The **trace** $\mathrm{tr}(\mathbf{A}) = \sum_i a_{ii}$. The **determinant** $\det(\mathbf{A})$ measures signed volume scaling and is nonzero iff $\mathbf{A}$ is invertible.

### The design matrix

The **design matrix** $\mathbf{X} \in \mathbb{R}^{n \times p}$ stacks $n$ observations of $p$ predictors row-wise: row $i$ holds the predictors for observation $i$. The column space of $\mathbf{X}$ is the set of all linear combinations of its $p$ columns — the space of fitted values reachable by OLS.

### Rank, null space, four fundamental subspaces

For $\mathbf{A} \in \mathbb{R}^{m \times n}$:

- $\mathrm{Col}(\mathbf{A}) \subseteq \mathbb{R}^m$ (column space)
- $\mathrm{Null}(\mathbf{A}) \subseteq \mathbb{R}^n$ (right null space)
- $\mathrm{Col}(\mathbf{A}^T) \subseteq \mathbb{R}^n$ (row space)
- $\mathrm{Null}(\mathbf{A}^T) \subseteq \mathbb{R}^m$ (left null space)

with $\mathrm{rank}(\mathbf{A}) + \dim \mathrm{Null}(\mathbf{A}) = n$ (rank–nullity theorem) and $\mathrm{Col}(\mathbf{A}^T) \perp \mathrm{Null}(\mathbf{A})$.

## Explanation

### Identities used constantly in this book

- $(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T \mathbf{A}^T$
- $(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1} \mathbf{A}^{-1}$
- $\mathrm{tr}(\mathbf{A}\mathbf{B}) = \mathrm{tr}(\mathbf{B}\mathbf{A})$ (cyclic property)
- $\mathrm{tr}(\mathbf{A}) = \sum_i \lambda_i(\mathbf{A})$, $\det(\mathbf{A}) = \prod_i \lambda_i(\mathbf{A})$ for square $\mathbf{A}$
- $\mathrm{Cov}(\mathbf{A} \mathbf{X}) = \mathbf{A}\, \mathrm{Cov}(\mathbf{X})\, \mathbf{A}^T$ when $\mathbf{X}$ is a random vector
- $\mathbb{E}[\mathbf{X}^T \mathbf{A} \mathbf{X}] = \mathrm{tr}(\mathbf{A}\, \mathrm{Cov}(\mathbf{X})) + \boldsymbol{\mu}^T \mathbf{A} \boldsymbol{\mu}$ (the quadratic-form expectation)

### Eigenvalues and the spectral theorem

For $\mathbf{A} \in \mathbb{R}^{n \times n}$: $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$ defines eigenvalue–eigenvector pairs. For **symmetric** $\mathbf{A}$ (the case that covers every covariance matrix, every $\mathbf{X}^T \mathbf{X}$, every hat matrix), the **spectral theorem** guarantees

$$
\mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T
$$

with $\mathbf{Q}$ orthogonal ($\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$) and $\boldsymbol{\Lambda}$ diagonal of real eigenvalues. This is the engine behind PCA, multivariate normal theory, and the chi-squared distribution of quadratic forms.

### Positive (semi)definiteness

$\mathbf{A}$ is **positive semidefinite** ($\mathbf{A} \succeq 0$) if $\mathbf{x}^T \mathbf{A} \mathbf{x} \ge 0$ for all $\mathbf{x}$, equivalently all eigenvalues $\ge 0$. **Positive definite** ($\mathbf{A} \succ 0$) replaces both inequalities by strict. Covariance matrices are always PSD; they are PD when no variable is a deterministic linear combination of the others. PD is exactly the condition needed for $(\mathbf{X}^T \mathbf{X})^{-1}$ to exist and for OLS to be uniquely defined.

### Projections and the hat matrix

The **OLS hat matrix**

$$
\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T
$$

projects $\mathbf{y} \in \mathbb{R}^n$ orthogonally onto the column space of $\mathbf{X}$. Its defining properties:

- **Symmetric**: $\mathbf{H}^T = \mathbf{H}$.
- **Idempotent**: $\mathbf{H}^2 = \mathbf{H}$.
- **Trace = rank**: $\mathrm{tr}(\mathbf{H}) = p$ (the model's degrees of freedom).
- **Eigenvalues 0 or 1**: $p$ ones (column space) and $n - p$ zeros (residual space).

The complementary projector $\mathbf{M} = \mathbf{I} - \mathbf{H}$ projects onto the residual space, with $\mathrm{tr}(\mathbf{M}) = n - p$ — the residual degrees of freedom that appears in every $t$-test and $F$-test.

### Matrix calculus for OLS

Minimizing $\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\beta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\beta})$ by setting the gradient to zero gives the normal equations $\mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y}$ and thus

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

This single formula and the projection interpretation behind it underlie all of Chapter 13's regression material.

## Examples

```python
import numpy as np

rng = np.random.default_rng(42)
n, p = 50, 3
X = np.column_stack([np.ones(n), rng.standard_normal((n, p - 1))])
beta_true = np.array([2.0, -1.0, 0.5])
y = X @ beta_true + rng.standard_normal(n) * 0.5

# === OLS via normal equations ===
beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
print("True beta:", beta_true)
print("OLS beta: ", beta_hat.round(4))

# === Hat matrix properties ===
H = X @ np.linalg.inv(X.T @ X) @ X.T
print(f"Symmetric:  {np.allclose(H, H.T)}")
print(f"Idempotent: {np.allclose(H @ H, H)}")
print(f"tr(H) = {np.trace(H):.1f}  (should equal p = {p})")

# === Spectral decomposition of X'X (symmetric, PD here) ===
eigvals, eigvecs = np.linalg.eigh(X.T @ X)
print(f"Eigenvalues: {eigvals.round(2)}")
print(f"Condition number: {eigvals.max() / eigvals.min():.2f}")
reconstruction = eigvecs @ np.diag(eigvals) @ eigvecs.T
print(f"Spectral reconstruction matches X'X: {np.allclose(reconstruction, X.T @ X)}")
```

## Exercises

**Exercise 1.**
For the design matrix

$$
\mathbf{X} = \begin{pmatrix} 1 & 2 \\ 1 & 4 \\ 1 & 6 \end{pmatrix}, \qquad \mathbf{y} = \begin{pmatrix} 5 \\ 9 \\ 13 \end{pmatrix}
$$

**(a)** Compute $\mathbf{X}^T \mathbf{X}$ and $\mathbf{X}^T \mathbf{y}$.
**(b)** Solve the normal equations to find $\hat{\boldsymbol{\beta}}$.
**(c)** Compute the fitted values $\hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}}$ and the residuals.

??? success "Solution to Exercise 1"
    (a) $\mathbf{X}^T \mathbf{X} = \begin{pmatrix} 3 & 12 \\ 12 & 56 \end{pmatrix}$, $\mathbf{X}^T \mathbf{y} = \begin{pmatrix} 27 \\ 124 \end{pmatrix}$.

    (b) $\det(\mathbf{X}^T \mathbf{X}) = 168 - 144 = 24$, so

    $$
    (\mathbf{X}^T \mathbf{X})^{-1} = \frac{1}{24}\begin{pmatrix} 56 & -12 \\ -12 & 3 \end{pmatrix}, \quad \hat{\boldsymbol{\beta}} = \frac{1}{24}\begin{pmatrix} 24 \\ 48 \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}
    $$

    (c) $\hat{\mathbf{y}} = (5, 9, 13)^T = \mathbf{y}$, residuals all zero. The three points are collinear, so the fit is exact.

---

**Exercise 2.**
Prove the cyclic property of trace: for $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n \times m}$, $\mathrm{tr}(\mathbf{A}\mathbf{B}) = \mathrm{tr}(\mathbf{B}\mathbf{A})$.

??? success "Solution to Exercise 2"
    Direct calculation:

    $$
    \mathrm{tr}(\mathbf{A}\mathbf{B}) = \sum_{i=1}^m [\mathbf{A}\mathbf{B}]_{ii} = \sum_{i=1}^m \sum_{j=1}^n a_{ij} b_{ji} = \sum_{j=1}^n \sum_{i=1}^m b_{ji} a_{ij} = \sum_{j=1}^n [\mathbf{B}\mathbf{A}]_{jj} = \mathrm{tr}(\mathbf{B}\mathbf{A})
    $$

    The reordering uses only finiteness of the sums. $\square$

---

**Exercise 3.**
Let $\mathbf{X} \in \mathbb{R}^{n \times p}$ have full column rank ($\mathrm{rank}(\mathbf{X}) = p \le n$). Prove that $\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$ is symmetric, idempotent, and has trace $p$.

??? success "Solution to Exercise 3"
    **Symmetric:** $(\mathbf{X}^T \mathbf{X})^{-1}$ is symmetric (inverse of a symmetric matrix), so

    $$
    \mathbf{H}^T = \left(\mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T\right)^T = \mathbf{X}\left((\mathbf{X}^T \mathbf{X})^{-1}\right)^T \mathbf{X}^T = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T = \mathbf{H}
    $$

    **Idempotent:**

    $$
    \mathbf{H}^2 = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1}\underbrace{\mathbf{X}^T \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1}}_{= \mathbf{I}_p}\mathbf{X}^T = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T = \mathbf{H}
    $$

    **Trace:** by the cyclic property,

    $$
    \mathrm{tr}(\mathbf{H}) = \mathrm{tr}\!\left(\mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T\right) = \mathrm{tr}\!\left((\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{X}\right) = \mathrm{tr}(\mathbf{I}_p) = p
    $$

    $\square$

---

**Exercise 4.**
Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be symmetric with spectral decomposition $\mathbf{A} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T$. Prove $\mathrm{tr}(\mathbf{A}) = \sum_i \lambda_i$ and $\det(\mathbf{A}) = \prod_i \lambda_i$.

??? success "Solution to Exercise 4"
    Using the cyclic property,

    $$
    \mathrm{tr}(\mathbf{A}) = \mathrm{tr}(\mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T) = \mathrm{tr}(\boldsymbol{\Lambda} \mathbf{Q}^T \mathbf{Q}) = \mathrm{tr}(\boldsymbol{\Lambda}) = \sum_i \lambda_i
    $$

    using $\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$. For the determinant, $\det$ is multiplicative and $\det(\mathbf{Q}) = \pm 1$ for orthogonal $\mathbf{Q}$:

    $$
    \det(\mathbf{A}) = \det(\mathbf{Q}) \det(\boldsymbol{\Lambda}) \det(\mathbf{Q}^T) = \det(\mathbf{Q})^2 \prod_i \lambda_i = \prod_i \lambda_i
    $$

    $\square$

---

**Exercise 5.**
Let $\mathbf{X}$ be a random vector with mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$. Show that for a symmetric matrix $\mathbf{A}$,

$$
\mathbb{E}[\mathbf{X}^T \mathbf{A} \mathbf{X}] = \mathrm{tr}(\mathbf{A} \boldsymbol{\Sigma}) + \boldsymbol{\mu}^T \mathbf{A} \boldsymbol{\mu}
$$

??? success "Solution to Exercise 5"
    Write $\mathbf{X} = \boldsymbol{\mu} + \mathbf{Z}$ where $\mathbb{E}[\mathbf{Z}] = 0$ and $\mathrm{Cov}(\mathbf{Z}) = \boldsymbol{\Sigma}$. Expanding,

    $$
    \mathbf{X}^T \mathbf{A} \mathbf{X} = \boldsymbol{\mu}^T \mathbf{A} \boldsymbol{\mu} + 2 \boldsymbol{\mu}^T \mathbf{A} \mathbf{Z} + \mathbf{Z}^T \mathbf{A} \mathbf{Z}
    $$

    (using symmetry of $\mathbf{A}$). Taking expectation, the middle term vanishes since $\mathbb{E}[\mathbf{Z}] = 0$. For the last term, $\mathbf{Z}^T \mathbf{A} \mathbf{Z}$ is a scalar so it equals its own trace, and

    $$
    \mathbb{E}[\mathbf{Z}^T \mathbf{A} \mathbf{Z}] = \mathbb{E}[\mathrm{tr}(\mathbf{Z}^T \mathbf{A} \mathbf{Z})] = \mathbb{E}[\mathrm{tr}(\mathbf{A} \mathbf{Z} \mathbf{Z}^T)] = \mathrm{tr}(\mathbf{A}\, \mathbb{E}[\mathbf{Z} \mathbf{Z}^T]) = \mathrm{tr}(\mathbf{A} \boldsymbol{\Sigma})
    $$

    Adding the deterministic piece gives the result. $\square$

---

**Exercise 6.**
If $\mathbf{X}^T \mathbf{X}$ is singular (i.e., $\mathbf{X}$ does not have full column rank), explain in two complementary ways why the OLS estimator $\hat{\boldsymbol{\beta}}$ is not uniquely defined: (a) algebraically and (b) geometrically.

??? success "Solution to Exercise 6"
    **(a) Algebraic:** singularity means $\det(\mathbf{X}^T \mathbf{X}) = 0$, so $(\mathbf{X}^T \mathbf{X})^{-1}$ does not exist. The normal equations $\mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y}$ are consistent (right-hand side lies in the column space of $\mathbf{X}^T \mathbf{X}$) but have infinitely many solutions: if $\boldsymbol{\beta}^*$ is a solution and $\mathbf{v} \in \mathrm{Null}(\mathbf{X})$, then $\boldsymbol{\beta}^* + \mathbf{v}$ also satisfies the equations because $\mathbf{X}\mathbf{v} = \mathbf{0}$.

    **(b) Geometric:** $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ is still the unique orthogonal projection of $\mathbf{y}$ onto $\mathrm{Col}(\mathbf{X})$. However, when columns of $\mathbf{X}$ are linearly dependent, the projection can be written as a linear combination of those columns in infinitely many ways — each gives a valid $\hat{\boldsymbol{\beta}}$. The fitted values are identified; the coefficients are not. Remedies are dropping dependent columns, ridge regression (adds $\lambda \mathbf{I}$ to $\mathbf{X}^T \mathbf{X}$ to restore invertibility), or the pseudoinverse (returns the minimum-norm solution). $\square$
