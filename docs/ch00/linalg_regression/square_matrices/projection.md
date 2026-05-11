# Projection Matrices

A projection "collapses" every vector in $\mathbb{R}^n$ onto a subspace $\mathcal{V}$ along a complementary subspace $\mathcal{W}$. Algebraically, a projection is an idempotent linear transformation: applying it twice produces the same result as applying it once. When the complementary subspace is the orthogonal complement of $\mathcal{V}$, the projection is **orthogonal** (discussed in the next section). This section covers the general (oblique) case, which provides the conceptual framework for understanding projections before adding the orthogonality restriction.

## Definition

!!! info "Definition -- Projection (General)"
    A square matrix $\mathbf{P} \in \mathbb{R}^{n \times n}$ is a **projection matrix** (or **projector**) if

    $$
    \mathbf{P}^2 = \mathbf{P}
    $$

    That is, $\mathbf{P}$ is idempotent.

A projection matrix is associated with two subspaces:

- The **range** (column space) $\mathcal{V} = \operatorname{col}(\mathbf{P})$: the subspace onto which vectors are projected.
- The **null space** $\mathcal{W} = \ker(\mathbf{P})$: the subspace along which the projection collapses.

## Decomposition of the Space

!!! tip "Theorem -- Direct Sum Decomposition"
    If $\mathbf{P}$ is a projection, then $\mathbb{R}^n = \operatorname{col}(\mathbf{P}) \oplus \ker(\mathbf{P})$, and for every $\mathbf{x} \in \mathbb{R}^n$:

    $$
    \mathbf{x} = \mathbf{P}\mathbf{x} + (\mathbf{I} - \mathbf{P})\mathbf{x}
    $$

    where $\mathbf{P}\mathbf{x} \in \operatorname{col}(\mathbf{P})$ and $(\mathbf{I} - \mathbf{P})\mathbf{x} \in \ker(\mathbf{P})$.

**Proof.** The decomposition $\mathbf{x} = \mathbf{P}\mathbf{x} + (\mathbf{I} - \mathbf{P})\mathbf{x}$ is trivially true. We verify the claimed subspace memberships:

- $\mathbf{P}\mathbf{x} \in \operatorname{col}(\mathbf{P})$ by definition.
- $\mathbf{P}\bigl((\mathbf{I} - \mathbf{P})\mathbf{x}\bigr) = (\mathbf{P} - \mathbf{P}^2)\mathbf{x} = (\mathbf{P} - \mathbf{P})\mathbf{x} = \mathbf{0}$, so $(\mathbf{I} - \mathbf{P})\mathbf{x} \in \ker(\mathbf{P})$.

To see the sum is direct, suppose $\mathbf{v} \in \operatorname{col}(\mathbf{P}) \cap \ker(\mathbf{P})$. Then $\mathbf{v} = \mathbf{P}\mathbf{u}$ for some $\mathbf{u}$, and $\mathbf{P}\mathbf{v} = \mathbf{0}$. But $\mathbf{P}\mathbf{v} = \mathbf{P}^2\mathbf{u} = \mathbf{P}\mathbf{u} = \mathbf{v}$, so $\mathbf{v} = \mathbf{0}$. $\square$

## The Complementary Projection

Since $\mathbf{P}$ is idempotent, the complementary matrix $\mathbf{I} - \mathbf{P}$ is also idempotent:

$$
(\mathbf{I} - \mathbf{P})^2 = \mathbf{I} - 2\mathbf{P} + \mathbf{P}^2 = \mathbf{I} - \mathbf{P}
$$

The complementary projection $\mathbf{I} - \mathbf{P}$ projects onto $\ker(\mathbf{P})$ along $\operatorname{col}(\mathbf{P})$. The two projections together decompose any vector into its components in the two complementary subspaces.

**Rank relationship:**

$$
\operatorname{rank}(\mathbf{P}) + \operatorname{rank}(\mathbf{I} - \mathbf{P}) = n
$$

## Eigenvalues and Trace

Since every projection is idempotent, its eigenvalues are restricted to $\{0, 1\}$ and

$$
\operatorname{tr}(\mathbf{P}) = \operatorname{rank}(\mathbf{P}) = \dim(\operatorname{col}(\mathbf{P}))
$$

The trace counts the dimension of the subspace onto which $\mathbf{P}$ projects.

## Oblique vs Orthogonal Projections

A projection $\mathbf{P}$ is **orthogonal** if $\mathbf{P} = \mathbf{P}^T$ (the projection is symmetric). Otherwise, the projection is **oblique**.

| Property | Orthogonal projection | Oblique projection |
|---|---|---|
| $\mathbf{P}^2 = \mathbf{P}$ | Yes | Yes |
| $\mathbf{P}^T = \mathbf{P}$ | Yes | No |
| $\ker(\mathbf{P}) \perp \operatorname{col}(\mathbf{P})$ | Yes | No |
| Minimizes $\lVert\mathbf{x} - \mathbf{P}\mathbf{x}\rVert$ | Yes | No (in general) |

In statistics, nearly all projections that arise naturally are orthogonal (the hat matrix, the residual-maker matrix, the centering matrix). Oblique projections can appear in instrumental-variable estimation and generalized least squares.

## Example -- Oblique Projection

Consider the projection onto $\mathcal{V} = \operatorname{span}\{(1, 0)^T\}$ along $\mathcal{W} = \operatorname{span}\{(1, 1)^T\}$ in $\mathbb{R}^2$.

Any vector $\mathbf{x} = (x_1, x_2)^T$ decomposes as $\mathbf{x} = \alpha(1, 0)^T + \beta(1, 1)^T$ where $\alpha = x_1 - x_2$ and $\beta = x_2$. The projection onto $\mathcal{V}$ along $\mathcal{W}$ keeps only the $\mathcal{V}$-component:

$$
\mathbf{P}\mathbf{x} = \alpha\begin{pmatrix}1 \\ 0\end{pmatrix} = \begin{pmatrix}x_1 - x_2 \\ 0\end{pmatrix}
$$

In matrix form:

$$
\mathbf{P} = \begin{pmatrix} 1 & -1 \\ 0 & 0 \end{pmatrix}
$$

**Verification:** $\mathbf{P}^2 = \begin{pmatrix} 1 & -1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 1 & -1 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 1 & -1 \\ 0 & 0 \end{pmatrix} = \mathbf{P}$. The matrix is idempotent but not symmetric ($\mathbf{P} \neq \mathbf{P}^T$), so this is an oblique projection.

## Example -- Orthogonal Projection in One Dimension

The projection onto $\mathcal{V} = \operatorname{span}\{(1, 0)^T\}$ along $\mathcal{V}^\perp = \operatorname{span}\{(0, 1)^T\}$ is

$$
\mathbf{P} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}
$$

This is both idempotent and symmetric, making it an orthogonal projection. It maps $(x_1, x_2)^T$ to $(x_1, 0)^T$ by dropping the second component.

## Uniqueness

!!! tip "Theorem -- Uniqueness of Projections"
    Given a direct sum decomposition $\mathbb{R}^n = \mathcal{V} \oplus \mathcal{W}$, there is a unique projection $\mathbf{P}$ with $\operatorname{col}(\mathbf{P}) = \mathcal{V}$ and $\ker(\mathbf{P}) = \mathcal{W}$.

This means that specifying the target subspace $\mathcal{V}$ alone does not uniquely determine the projection -- one must also specify the direction of collapse $\mathcal{W}$. When $\mathcal{W} = \mathcal{V}^\perp$, the projection is orthogonal, and in this special case $\mathcal{V}$ alone determines the projection.

## Connection to Statistics

### Fitted Values and Residuals

In linear regression, the decomposition $\mathbf{y} = \mathbf{H}\mathbf{y} + (\mathbf{I} - \mathbf{H})\mathbf{y} = \hat{\mathbf{y}} + \mathbf{e}$ is a projection decomposition. The hat matrix $\mathbf{H}$ projects onto the column space of $\mathbf{X}$, and $\mathbf{I} - \mathbf{H}$ projects onto the orthogonal complement (the residual space).

### ANOVA Decomposition

The total sum of squares decomposition in ANOVA can be written as a sum of quadratic forms involving orthogonal projections onto the model subspace and the error subspace. The orthogonality ensures that the sums of squares are independent (under normality), which is the basis for the F-test.

## Summary

A projection matrix is an idempotent matrix that decomposes $\mathbb{R}^n$ into a direct sum of its column space and null space. Every vector splits into a component in each subspace, and applying the projection a second time does nothing new. The trace equals the dimension of the target subspace. When the projection is additionally symmetric, it becomes an orthogonal projection, which is the type that appears most frequently in regression and ANOVA.

## Exercises

**Exercise 1.**
Verify that $\mathbf{P} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$ is a projection matrix. What subspace does it project onto? What is the complementary projection $\mathbf{I} - \mathbf{P}$?

??? success "Solution to Exercise 1"
    Check idempotency:

    $$
    \mathbf{P}^2 = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix} = \mathbf{P}
    $$

    The column space of $\mathbf{P}$ is $\text{span}\{(1, 0)^T\}$, so $\mathbf{P}$ projects onto the $x_1$-axis. The complementary projection is:

    $$
    \mathbf{I} - \mathbf{P} = \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}
    $$

    which projects onto the $x_2$-axis. Since $\mathbf{P}$ is also symmetric, this is an orthogonal projection.

---

**Exercise 2.**
Prove that if $\mathbf{P}$ is idempotent, then $\operatorname{rank}(\mathbf{P}) = \operatorname{tr}(\mathbf{P})$.

??? success "Solution to Exercise 2"
    Since $\mathbf{P}$ is idempotent, its eigenvalues are either 0 or 1 (if $\mathbf{P}\mathbf{v} = \lambda\mathbf{v}$, then $\mathbf{P}^2\mathbf{v} = \lambda^2\mathbf{v} = \lambda\mathbf{v}$, so $\lambda^2 = \lambda$ and $\lambda \in \{0, 1\}$).

    The rank equals the number of nonzero eigenvalues, which is the number of eigenvalues equal to 1. The trace equals the sum of all eigenvalues, which is also the number of eigenvalues equal to 1 (since the rest are 0).

    Therefore $\operatorname{rank}(\mathbf{P}) = \operatorname{tr}(\mathbf{P})$. $\square$

---

**Exercise 3.**
Give an example of an oblique (non-orthogonal) projection matrix. Verify it is idempotent but not symmetric.

??? success "Solution to Exercise 3"
    Consider $\mathbf{P} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}$.

    Idempotency:

    $$
    \mathbf{P}^2 = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}\begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} = \mathbf{P}
    $$

    But $\mathbf{P}^T = \begin{pmatrix} 1 & 0 \\ 1 & 0 \end{pmatrix} \neq \mathbf{P}$, so $\mathbf{P}$ is not symmetric.

    This matrix projects onto $\text{span}\{(1, 0)^T\}$ along the direction $(1, 0)^T + \ker(\mathbf{P}) = (1, 0)^T + \text{span}\{(-1, 1)^T\}$. The projection direction is oblique (not perpendicular) to the target subspace.

---

**Exercise 4.**
In the regression decomposition $\mathbf{y} = \mathbf{H}\mathbf{y} + (\mathbf{I} - \mathbf{H})\mathbf{y}$, explain why $\mathbf{H}\mathbf{y}$ and $(\mathbf{I} - \mathbf{H})\mathbf{y}$ are orthogonal. What additional property (beyond idempotency) is required?

??? success "Solution to Exercise 4"
    The orthogonality of $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ and $\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}$ requires that $\mathbf{H}$ be symmetric (not just idempotent). With symmetry:

    $$
    \hat{\mathbf{y}}^T\mathbf{e} = \mathbf{y}^T\mathbf{H}^T(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y}^T\mathbf{H}(\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{y}^T(\mathbf{H} - \mathbf{H}^2)\mathbf{y} = \mathbf{0}
    $$

    If $\mathbf{H}$ were idempotent but not symmetric (oblique projection), the decomposition $\mathbf{y} = \mathbf{H}\mathbf{y} + (\mathbf{I} - \mathbf{H})\mathbf{y}$ still holds but the two components are not orthogonal. The OLS hat matrix is both idempotent and symmetric, which is what makes the sums-of-squares decomposition and the Pythagorean theorem work.
