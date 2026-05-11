# Orthogonal Projection Matrices

An orthogonal projection matrix maps every vector to the closest point in a subspace, where "closest" is measured by Euclidean distance. This is the geometric content of ordinary least squares: the fitted-value vector $\hat{\mathbf{y}}$ is the orthogonal projection of the response vector $\mathbf{y}$ onto the column space of the design matrix $\mathbf{X}$, and the residual vector $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$ is perpendicular to that column space. Orthogonal projections are characterized by being both idempotent and symmetric, and their explicit formula $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is the hat matrix of regression.

## Definition

!!! info "Definition -- Orthogonal Projection Matrix"
    A square matrix $\mathbf{P} \in \mathbb{R}^{n \times n}$ is an **orthogonal projection matrix** if it is both idempotent and symmetric:

    $$
    \mathbf{P}^2 = \mathbf{P} \quad \text{and} \quad \mathbf{P}^T = \mathbf{P}
    $$

The symmetry condition distinguishes orthogonal projections from oblique projections. A projection is orthogonal if and only if the null space is the orthogonal complement of the column space: $\ker(\mathbf{P}) = \operatorname{col}(\mathbf{P})^\perp$.

## The Best-Approximation Property

The defining geometric property of orthogonal projections is that they solve the nearest-point problem.

!!! tip "Theorem -- Best Approximation"
    Let $\mathbf{P}$ be the orthogonal projection onto a subspace $\mathcal{V} \subseteq \mathbb{R}^n$. For any $\mathbf{x} \in \mathbb{R}^n$, the vector $\mathbf{P}\mathbf{x}$ is the unique element of $\mathcal{V}$ that minimizes the distance to $\mathbf{x}$:

    $$
    \mathbf{P}\mathbf{x} = \arg\min_{\mathbf{v} \in \mathcal{V}} \lVert\mathbf{x} - \mathbf{v}\rVert
    $$

**Proof sketch.** Let $\mathbf{v} \in \mathcal{V}$ be arbitrary. Then

$$
\lVert\mathbf{x} - \mathbf{v}\rVert^2 = \lVert(\mathbf{x} - \mathbf{P}\mathbf{x}) + (\mathbf{P}\mathbf{x} - \mathbf{v})\rVert^2
$$

The vector $\mathbf{x} - \mathbf{P}\mathbf{x} \in \mathcal{V}^\perp$ (since $\mathbf{P}$ is an orthogonal projection) and $\mathbf{P}\mathbf{x} - \mathbf{v} \in \mathcal{V}$, so these two vectors are orthogonal. By the Pythagorean theorem:

$$
\lVert\mathbf{x} - \mathbf{v}\rVert^2 = \lVert\mathbf{x} - \mathbf{P}\mathbf{x}\rVert^2 + \lVert\mathbf{P}\mathbf{x} - \mathbf{v}\rVert^2 \geq \lVert\mathbf{x} - \mathbf{P}\mathbf{x}\rVert^2
$$

Equality holds if and only if $\mathbf{v} = \mathbf{P}\mathbf{x}$. $\square$

This is exactly why OLS minimizes the sum of squared residuals: $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ is the closest point to $\mathbf{y}$ in $\operatorname{col}(\mathbf{X})$.

## Formula for Orthogonal Projection onto a Column Space

!!! tip "Theorem -- Projection Formula"
    Let $\mathbf{X} \in \mathbb{R}^{n \times p}$ have full column rank ($\operatorname{rank}(\mathbf{X}) = p$). The orthogonal projection onto $\operatorname{col}(\mathbf{X})$ is

    $$
    \mathbf{P}_{\mathbf{X}} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T
    $$

**Proof.** We verify the two defining properties.

*Idempotent:*

$$
\mathbf{P}_{\mathbf{X}}^2 = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{P}_{\mathbf{X}}
$$

*Symmetric:*

$$
\mathbf{P}_{\mathbf{X}}^T = (\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T)^T = \mathbf{X}((\mathbf{X}^T\mathbf{X})^{-1})^T\mathbf{X}^T = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{P}_{\mathbf{X}}
$$

where the last step uses the fact that $(\mathbf{X}^T\mathbf{X})^{-1}$ is symmetric (the inverse of a symmetric matrix is symmetric). $\square$

## Properties

### Eigenvalues

The eigenvalues of an orthogonal projection are 0 and 1 (inherited from idempotency), with

$$
\operatorname{tr}(\mathbf{P}_{\mathbf{X}}) = \operatorname{rank}(\mathbf{P}_{\mathbf{X}}) = p
$$

### Complementary Projection

The matrix $\mathbf{M} = \mathbf{I} - \mathbf{P}_{\mathbf{X}}$ is the orthogonal projection onto $\operatorname{col}(\mathbf{X})^\perp$:

$$
\mathbf{M}^2 = \mathbf{M}, \quad \mathbf{M}^T = \mathbf{M}, \quad \operatorname{tr}(\mathbf{M}) = n - p
$$

### Orthogonality of Projections

The range of $\mathbf{P}_{\mathbf{X}}$ and the range of $\mathbf{M}$ are orthogonal complements, so

$$
\mathbf{P}_{\mathbf{X}}\mathbf{M} = \mathbf{M}\mathbf{P}_{\mathbf{X}} = \mathbf{0}
$$

### Column Space Characterization

A vector $\mathbf{v}$ is in $\operatorname{col}(\mathbf{X})$ if and only if $\mathbf{P}_{\mathbf{X}}\mathbf{v} = \mathbf{v}$, and $\mathbf{v}$ is orthogonal to $\operatorname{col}(\mathbf{X})$ if and only if $\mathbf{P}_{\mathbf{X}}\mathbf{v} = \mathbf{0}$.

## Projection onto a Single Vector

When the subspace is one-dimensional, $\mathcal{V} = \operatorname{span}\{\mathbf{u}\}$ for a nonzero vector $\mathbf{u}$, the projection formula simplifies to

$$
\mathbf{P}_{\mathbf{u}} = \frac{\mathbf{u}\mathbf{u}^T}{\mathbf{u}^T\mathbf{u}}
$$

This projects any vector $\mathbf{x}$ onto $\mathbf{u}$:

$$
\mathbf{P}_{\mathbf{u}}\mathbf{x} = \frac{\mathbf{u}^T\mathbf{x}}{\mathbf{u}^T\mathbf{u}}\,\mathbf{u}
$$

The scalar $\frac{\mathbf{u}^T\mathbf{x}}{\mathbf{u}^T\mathbf{u}}$ is the coefficient of $\mathbf{x}$ projected onto $\mathbf{u}$.

## Example -- The Centering Matrix

The **centering matrix** is

$$
\mathbf{C} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}_n\mathbf{1}_n^T
$$

where $\mathbf{1}_n = (1, \dots, 1)^T$. This matrix satisfies:

- $\mathbf{C}^2 = \mathbf{C}$ (idempotent)
- $\mathbf{C}^T = \mathbf{C}$ (symmetric)
- $\operatorname{tr}(\mathbf{C}) = n - 1$

So $\mathbf{C}$ is an orthogonal projection of rank $n - 1$. It projects onto the subspace orthogonal to $\mathbf{1}_n$ (the subspace of vectors whose entries sum to zero). For any data vector $\mathbf{x}$:

$$
\mathbf{C}\mathbf{x} = \mathbf{x} - \bar{x}\mathbf{1}_n
$$

where $\bar{x} = \frac{1}{n}\sum_i x_i$ is the sample mean. Centering data is an orthogonal projection.

## Example -- The Hat Matrix

In the linear model $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ with $\mathbf{X} \in \mathbb{R}^{n \times p}$ of full column rank, the **hat matrix** is

$$
\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T
$$

This is the orthogonal projection onto $\operatorname{col}(\mathbf{X})$. The name comes from the fact that $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ -- the hat matrix "puts the hat on $\mathbf{y}$."

Key properties in regression:

- $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ (fitted values)
- $\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}$ (residuals)
- $\hat{\mathbf{y}} \perp \mathbf{e}$ (fitted values orthogonal to residuals)
- $\operatorname{tr}(\mathbf{H}) = p$ (model degrees of freedom)
- $\operatorname{tr}(\mathbf{I} - \mathbf{H}) = n - p$ (residual degrees of freedom)

## The Pythagorean Theorem in Regression

The orthogonality $\hat{\mathbf{y}} \perp \mathbf{e}$ gives the Pythagorean theorem:

$$
\lVert\mathbf{y}\rVert^2 = \lVert\hat{\mathbf{y}}\rVert^2 + \lVert\mathbf{e}\rVert^2
$$

When the model includes an intercept and we measure from the mean, this becomes the ANOVA decomposition:

$$
\text{SST} = \text{SSR} + \text{SSE}
$$

where SST is the total sum of squares, SSR is the regression sum of squares, and SSE is the error sum of squares. The $R^2$ coefficient is then $R^2 = \text{SSR}/\text{SST} = \lVert\hat{\mathbf{y}}\rVert^2 / \lVert\mathbf{y}\rVert^2$, which is the squared cosine of the angle between $\mathbf{y}$ and $\hat{\mathbf{y}}$ -- a geometric measure of how well the model fits.

## Summary

Orthogonal projection matrices are symmetric idempotent matrices that map vectors to the nearest point in a subspace. The formula $\mathbf{P}_{\mathbf{X}} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is the hat matrix of regression, and the complementary projection $\mathbf{I} - \mathbf{P}_{\mathbf{X}}$ produces residuals. The orthogonality of fitted values and residuals yields the Pythagorean decomposition of sums of squares, which is the geometric foundation of ANOVA and $R^2$.

## Exercises

**Exercise 1.**
Let $\mathbf{X} = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{pmatrix}$. Compute the hat matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ and verify that $\operatorname{tr}(\mathbf{H}) = 2$.

??? success "Solution to Exercise 1"
    First compute:

    $$
    \mathbf{X}^T\mathbf{X} = \begin{pmatrix} 3 & 6 \\ 6 & 14 \end{pmatrix}, \quad (\mathbf{X}^T\mathbf{X})^{-1} = \frac{1}{6}\begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix}
    $$

    Then:

    $$
    \mathbf{H} = \mathbf{X} \cdot \frac{1}{6}\begin{pmatrix} 14 & -6 \\ -6 & 3 \end{pmatrix} \cdot \mathbf{X}^T = \frac{1}{6}\begin{pmatrix} 5 & 2 & -1 \\ 2 & 2 & 2 \\ -1 & 2 & 5 \end{pmatrix}
    $$

    The trace is $\operatorname{tr}(\mathbf{H}) = (5 + 2 + 5)/6 = 12/6 = 2$, which equals the number of columns in $\mathbf{X}$ (i.e., $p = 2$). This confirms the general result $\operatorname{tr}(\mathbf{H}) = \operatorname{rank}(\mathbf{H}) = p$.

---

**Exercise 2.**
Prove the Pythagorean decomposition: $\lVert \mathbf{y} \rVert^2 = \lVert \hat{\mathbf{y}} \rVert^2 + \lVert \mathbf{e} \rVert^2$ where $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ and $\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}$.

??? success "Solution to Exercise 2"
    Since $\mathbf{y} = \hat{\mathbf{y}} + \mathbf{e}$:

    $$
    \lVert \mathbf{y} \rVert^2 = (\hat{\mathbf{y}} + \mathbf{e})^T(\hat{\mathbf{y}} + \mathbf{e}) = \lVert \hat{\mathbf{y}} \rVert^2 + 2\hat{\mathbf{y}}^T\mathbf{e} + \lVert \mathbf{e} \rVert^2
    $$

    The cross term vanishes because:

    $$
    \hat{\mathbf{y}}^T\mathbf{e} = (\mathbf{H}\mathbf{y})^T(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y}^T\mathbf{H}^T(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y}^T\mathbf{H}(\mathbf{I} - \mathbf{H})\mathbf{y}
    $$

    Using $\mathbf{H}(\mathbf{I} - \mathbf{H}) = \mathbf{H} - \mathbf{H}^2 = \mathbf{H} - \mathbf{H} = \mathbf{0}$, the cross term is zero, giving $\lVert \mathbf{y} \rVert^2 = \lVert \hat{\mathbf{y}} \rVert^2 + \lVert \mathbf{e} \rVert^2$. $\square$

---

**Exercise 3.**
Show that the hat matrix $\mathbf{H}$ minimizes $\lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert^2$ by proving that $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ is the closest point in $\text{col}(\mathbf{X})$ to $\mathbf{y}$.

??? success "Solution to Exercise 3"
    Let $\mathbf{z} = \mathbf{X}\boldsymbol{\beta}$ be any vector in $\text{col}(\mathbf{X})$. We need to show $\lVert \mathbf{y} - \hat{\mathbf{y}} \rVert \leq \lVert \mathbf{y} - \mathbf{z} \rVert$.

    Write $\mathbf{y} - \mathbf{z} = (\mathbf{y} - \hat{\mathbf{y}}) + (\hat{\mathbf{y}} - \mathbf{z})$. Since $\mathbf{y} - \hat{\mathbf{y}} = \mathbf{e} \in \text{col}(\mathbf{X})^\perp$ and $\hat{\mathbf{y}} - \mathbf{z} \in \text{col}(\mathbf{X})$, these two vectors are orthogonal. By the Pythagorean theorem:

    $$
    \lVert \mathbf{y} - \mathbf{z} \rVert^2 = \lVert \mathbf{y} - \hat{\mathbf{y}} \rVert^2 + \lVert \hat{\mathbf{y}} - \mathbf{z} \rVert^2 \geq \lVert \mathbf{y} - \hat{\mathbf{y}} \rVert^2
    $$

    Equality holds if and only if $\mathbf{z} = \hat{\mathbf{y}}$, confirming that $\hat{\mathbf{y}}$ is the unique closest point. $\square$

---

**Exercise 4.**
Explain the geometric interpretation of $R^2 = \lVert \hat{\mathbf{y}} \rVert^2 / \lVert \mathbf{y} \rVert^2$ in terms of the angle between $\mathbf{y}$ and its projection $\hat{\mathbf{y}}$. What does $R^2 = 1$ mean geometrically?

??? success "Solution to Exercise 4"
    Let $\theta$ be the angle between $\mathbf{y}$ and $\hat{\mathbf{y}}$ in $\mathbb{R}^n$. Since $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ is the projection of $\mathbf{y}$ onto $\text{col}(\mathbf{X})$:

    $$
    \cos\theta = \frac{\hat{\mathbf{y}}^T\mathbf{y}}{\lVert \hat{\mathbf{y}} \rVert \lVert \mathbf{y} \rVert} = \frac{\lVert \hat{\mathbf{y}} \rVert^2}{\lVert \hat{\mathbf{y}} \rVert \lVert \mathbf{y} \rVert} = \frac{\lVert \hat{\mathbf{y}} \rVert}{\lVert \mathbf{y} \rVert}
    $$

    Therefore $R^2 = \cos^2\theta$: it is the squared cosine of the angle between the response vector and its projection onto the model subspace.

    $R^2 = 1$ means $\cos^2\theta = 1$, so $\theta = 0$: the response vector $\mathbf{y}$ lies exactly in $\text{col}(\mathbf{X})$. Geometrically, the data fits the model perfectly with zero residual.
