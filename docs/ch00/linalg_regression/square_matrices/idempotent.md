# Idempotent Matrices

An operation that produces the same result when applied twice is called **idempotent**. In matrix algebra, idempotent matrices satisfy $\mathbf{A}^2 = \mathbf{A}$: applying the transformation a second time changes nothing. This property characterizes projections, and projection matrices are everywhere in regression. The hat matrix $\mathbf{H}$ that produces fitted values and the residual-maker matrix $\mathbf{M} = \mathbf{I} - \mathbf{H}$ that produces residuals are both idempotent. Their eigenvalues are restricted to $\{0, 1\}$, which translates directly into degrees-of-freedom counts in ANOVA and regression theory.

## Definition

!!! info "Definition — Idempotent Matrix"
    A square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is **idempotent** if

    $$
    \mathbf{A}^2 = \mathbf{A}
    $$

    Equivalently, $\mathbf{A}(\mathbf{A} - \mathbf{I}) = \mathbf{0}$.

The identity matrix $\mathbf{I}$ and the zero matrix $\mathbf{0}$ are trivially idempotent. The interesting cases arise from projection matrices, which are idempotent but neither $\mathbf{I}$ nor $\mathbf{0}$.

## Eigenvalues of Idempotent Matrices

!!! tip "Theorem — Eigenvalues Are 0 or 1"
    If $\mathbf{A}$ is idempotent and $\lambda$ is an eigenvalue of $\mathbf{A}$, then $\lambda \in \{0, 1\}$.

**Proof.** Let $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ with $\mathbf{v} \ne \mathbf{0}$. Then

$$
\mathbf{A}^2\mathbf{v} = \mathbf{A}(\lambda\mathbf{v}) = \lambda^2 \mathbf{v}
$$

But $\mathbf{A}^2 = \mathbf{A}$ also gives $\mathbf{A}^2\mathbf{v} = \mathbf{A}\mathbf{v} = \lambda\mathbf{v}$. Equating: $(\lambda^2 - \lambda)\mathbf{v} = \mathbf{0}$, so $\lambda(\lambda - 1) = 0$. $\square$

## Trace Equals Rank

!!! tip "Theorem — Trace-Rank Identity for Idempotent Matrices"
    If $\mathbf{A} \in \mathbb{R}^{n \times n}$ is idempotent, then

    $$
    \operatorname{tr}(\mathbf{A}) = \operatorname{rank}(\mathbf{A})
    $$

**Proof.** The trace equals the sum of eigenvalues (with multiplicity). Eigenvalues are 0 or 1, so the trace counts the number of $1$'s — which equals the dimension of the eigenvalue-1 eigenspace. For an idempotent matrix, this eigenspace is exactly the column space (Exercise 1), so its dimension equals the rank. $\square$

This identity has a direct statistical interpretation: the trace of the hat matrix equals the number of estimated parameters, and the trace of the residual-maker matrix equals the residual degrees of freedom.

## Key Properties

### Complement is idempotent

If $\mathbf{A}$ is idempotent, then $\mathbf{I} - \mathbf{A}$ is also idempotent:

$$
(\mathbf{I} - \mathbf{A})^2 = \mathbf{I} - 2\mathbf{A} + \mathbf{A}^2 = \mathbf{I} - 2\mathbf{A} + \mathbf{A} = \mathbf{I} - \mathbf{A}
$$

This is why both the hat matrix $\mathbf{H}$ and the residual-maker $\mathbf{I} - \mathbf{H}$ are projections.

### Rank decomposition

Combining trace-rank with linearity of trace:

$$
\operatorname{rank}(\mathbf{A}) + \operatorname{rank}(\mathbf{I} - \mathbf{A}) = \operatorname{tr}(\mathbf{A}) + \operatorname{tr}(\mathbf{I} - \mathbf{A}) = \operatorname{tr}(\mathbf{I}) = n
$$

### Column space is the set of fixed points

A vector $\mathbf{x}$ lies in the column space of an idempotent $\mathbf{A}$ if and only if $\mathbf{A}\mathbf{x} = \mathbf{x}$. The column space is exactly the eigenvalue-1 eigenspace; the null space is the eigenvalue-0 eigenspace. The two together decompose $\mathbb{R}^n$.

### Diagonalizability

Every idempotent matrix is diagonalizable. Reason: its minimal polynomial divides $\lambda^2 - \lambda = \lambda(\lambda - 1)$, which has distinct roots; a matrix is diagonalizable iff its minimal polynomial has distinct roots.

### Product of commuting idempotents

If $\mathbf{A}$ and $\mathbf{B}$ are idempotent and $\mathbf{A}\mathbf{B} = \mathbf{B}\mathbf{A}$, then $\mathbf{A}\mathbf{B}$ is idempotent. (Without commutativity this can fail.)

## Example

Consider

$$
\mathbf{A} = \frac{1}{3}\begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix} = \frac{1}{3} \mathbf{1}\mathbf{1}^T
$$

**Idempotency:** $\mathbf{A}^2 = \frac{1}{9}\mathbf{1}\mathbf{1}^T\mathbf{1}\mathbf{1}^T = \frac{1}{9}\mathbf{1}(3)\mathbf{1}^T = \frac{1}{3}\mathbf{1}\mathbf{1}^T = \mathbf{A}$.

**Trace and rank:** $\operatorname{tr}(\mathbf{A}) = 1 = \operatorname{rank}(\mathbf{A})$.

**Eigenvalues:** $\lambda_1 = 1$ with eigenvector $(1,1,1)^T$; $\lambda_2 = \lambda_3 = 0$ with eigenspace orthogonal to $(1,1,1)^T$.

**Geometric interpretation:** $\mathbf{A}$ projects every vector onto the span of $\mathbf{1}$ — it replaces each entry of $\mathbf{x}$ with the sample mean. This is the hat matrix of the intercept-only regression model.

## Idempotent Matrices in Regression

### The hat matrix

In the linear model $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ with $\mathbf{X} \in \mathbb{R}^{n \times p}$ of full column rank,

$$
\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T
$$

is symmetric and idempotent. Its trace gives the number of parameters: $\operatorname{tr}(\mathbf{H}) = p$ (Exercise 4).

### The residual-maker matrix

$\mathbf{M} = \mathbf{I} - \mathbf{H}$ is symmetric and idempotent with $\operatorname{tr}(\mathbf{M}) = n - p$, the residual degrees of freedom. Residuals are $\mathbf{e} = \mathbf{M}\mathbf{y}$.

### ANOVA decomposition

The Pythagorean decomposition $\mathbf{y} = \mathbf{H}\mathbf{y} + \mathbf{M}\mathbf{y}$ with $\mathbf{H}\mathbf{M} = \mathbf{0}$ gives

$$
\|\mathbf{y}\|^2 = \|\mathbf{H}\mathbf{y}\|^2 + \|\mathbf{M}\mathbf{y}\|^2
$$

— the sum-of-squares identity whose degrees of freedom are $\operatorname{tr}(\mathbf{H}) = p$ and $\operatorname{tr}(\mathbf{M}) = n - p$, summing to $n$.

## Summary

Idempotent matrices satisfy $\mathbf{A}^2 = \mathbf{A}$, have eigenvalues restricted to $\{0, 1\}$, and obey $\operatorname{tr}(\mathbf{A}) = \operatorname{rank}(\mathbf{A})$. The complement $\mathbf{I} - \mathbf{A}$ is also idempotent. In regression, both the hat matrix and the residual-maker are idempotent, and their traces directly give the degrees of freedom used in F-tests, t-tests, and confidence intervals.

## Exercises

**Exercise 1.**
Let $\mathbf{A}$ be idempotent. Prove that $\mathbf{x} \in \operatorname{Col}(\mathbf{A})$ if and only if $\mathbf{A}\mathbf{x} = \mathbf{x}$.

??? success "Solution to Exercise 1"
    ($\Rightarrow$) If $\mathbf{x} \in \operatorname{Col}(\mathbf{A})$, write $\mathbf{x} = \mathbf{A}\mathbf{y}$. Then $\mathbf{A}\mathbf{x} = \mathbf{A}^2 \mathbf{y} = \mathbf{A}\mathbf{y} = \mathbf{x}$.

    ($\Leftarrow$) If $\mathbf{A}\mathbf{x} = \mathbf{x}$, then $\mathbf{x}$ is expressed as $\mathbf{A}$ times $\mathbf{x}$ itself, so $\mathbf{x} \in \operatorname{Col}(\mathbf{A})$. $\square$

    Consequence: the column space coincides with the eigenvalue-1 eigenspace, and the null space coincides with the eigenvalue-0 eigenspace. The two are complementary subspaces of $\mathbb{R}^n$.

---

**Exercise 2.**
Prove that if $\mathbf{A}$ is idempotent, then so is $\mathbf{I} - \mathbf{A}$. What is $\operatorname{rank}(\mathbf{I} - \mathbf{A})$ in terms of $\operatorname{rank}(\mathbf{A})$?

??? success "Solution to Exercise 2"
    Direct computation:

    $$
    (\mathbf{I} - \mathbf{A})^2 = \mathbf{I} - 2\mathbf{A} + \mathbf{A}^2 = \mathbf{I} - 2\mathbf{A} + \mathbf{A} = \mathbf{I} - \mathbf{A}
    $$

    By the trace-rank identity, $\operatorname{rank}(\mathbf{I} - \mathbf{A}) = \operatorname{tr}(\mathbf{I} - \mathbf{A}) = n - \operatorname{tr}(\mathbf{A}) = n - \operatorname{rank}(\mathbf{A})$. $\square$

---

**Exercise 3.**
Prove that every idempotent matrix is diagonalizable. Give one example of a non-symmetric idempotent matrix.

??? success "Solution to Exercise 3"
    The minimal polynomial of an idempotent $\mathbf{A}$ divides $\lambda^2 - \lambda = \lambda(\lambda - 1)$, which factors into distinct linear factors. A matrix is diagonalizable iff its minimal polynomial splits into distinct linear factors. Hence $\mathbf{A}$ is diagonalizable.

    Non-symmetric example:

    $$
    \mathbf{A} = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}, \quad \mathbf{A}^2 = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} = \mathbf{A}
    $$

    Eigenvalues: $1$ with eigenvector $(1, 0)^T$ and $0$ with eigenvector $(-1, 1)^T$. The matrix projects onto the $x$-axis along the line $y = -x$ — an **oblique** (non-orthogonal) projection.

---

**Exercise 4.**
For the hat matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ with $\mathbf{X} \in \mathbb{R}^{n \times p}$ of full column rank, prove that $\mathbf{H}$ is symmetric and idempotent, and that $\operatorname{tr}(\mathbf{H}) = p$.

??? success "Solution to Exercise 4"
    **Symmetric:** $\mathbf{X}^T\mathbf{X}$ is symmetric, hence so is its inverse. Therefore

    $$
    \mathbf{H}^T = \mathbf{X}\bigl[(\mathbf{X}^T\mathbf{X})^{-1}\bigr]^T \mathbf{X}^T = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{H}
    $$

    **Idempotent:**

    $$
    \mathbf{H}^2 = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\underbrace{\mathbf{X}^T\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}}_{\mathbf{I}_p}\mathbf{X}^T = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{H}
    $$

    **Trace:** by the cyclic property of trace,

    $$
    \operatorname{tr}(\mathbf{H}) = \operatorname{tr}\!\bigl((\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X}\bigr) = \operatorname{tr}(\mathbf{I}_p) = p
    $$

    $\square$

---

**Exercise 5.**
Show that $\mathbf{H}\mathbf{X} = \mathbf{X}$ and $\mathbf{M}\mathbf{X} = \mathbf{0}$ for $\mathbf{M} = \mathbf{I} - \mathbf{H}$. Interpret each statement geometrically.

??? success "Solution to Exercise 5"
    Compute directly:

    $$
    \mathbf{H}\mathbf{X} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X} = \mathbf{X}, \qquad \mathbf{M}\mathbf{X} = (\mathbf{I} - \mathbf{H})\mathbf{X} = \mathbf{X} - \mathbf{X} = \mathbf{0}
    $$

    **Geometric meaning of $\mathbf{H}\mathbf{X} = \mathbf{X}$:** each column of $\mathbf{X}$ already lies in the column space of $\mathbf{X}$, so projecting it onto that space leaves it unchanged. $\mathbf{H}$ acts as the identity on $\operatorname{Col}(\mathbf{X})$.

    **Geometric meaning of $\mathbf{M}\mathbf{X} = \mathbf{0}$:** residuals are orthogonal to the column space of $\mathbf{X}$. This is precisely the normal-equation condition $\mathbf{X}^T \mathbf{e} = \mathbf{0}$ that defines OLS: the fitted values capture all of the linear signal in $\mathbf{X}$, leaving nothing in the residuals that can be explained by any column of $\mathbf{X}$.

---

**Exercise 6.**
Let $\mathbf{A}, \mathbf{B}$ be symmetric idempotent matrices in $\mathbb{R}^{n \times n}$ with $\mathbf{A}\mathbf{B} = \mathbf{0}$. Prove that $\mathbf{A} + \mathbf{B}$ is also symmetric idempotent, and that $\operatorname{rank}(\mathbf{A} + \mathbf{B}) = \operatorname{rank}(\mathbf{A}) + \operatorname{rank}(\mathbf{B})$. (This is the foundation of Cochran's theorem, which decomposes $\chi^2$ statistics in ANOVA.)

??? success "Solution to Exercise 6"
    **Symmetry:** $(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T = \mathbf{A} + \mathbf{B}$.

    **Idempotency:** Note $\mathbf{A}\mathbf{B} = \mathbf{0}$ implies $\mathbf{B}\mathbf{A} = (\mathbf{A}\mathbf{B})^T = \mathbf{0}$ (using symmetry). Then

    $$
    (\mathbf{A} + \mathbf{B})^2 = \mathbf{A}^2 + \mathbf{A}\mathbf{B} + \mathbf{B}\mathbf{A} + \mathbf{B}^2 = \mathbf{A} + \mathbf{0} + \mathbf{0} + \mathbf{B} = \mathbf{A} + \mathbf{B}
    $$

    **Rank:** by the trace-rank identity, $\operatorname{rank}(\mathbf{A} + \mathbf{B}) = \operatorname{tr}(\mathbf{A} + \mathbf{B}) = \operatorname{tr}(\mathbf{A}) + \operatorname{tr}(\mathbf{B}) = \operatorname{rank}(\mathbf{A}) + \operatorname{rank}(\mathbf{B})$. $\square$

    Statistical use: in the ANOVA decomposition $\mathbf{y} = \mathbf{P}_1\mathbf{y} + \mathbf{P}_2\mathbf{y} + \cdots$ with mutually orthogonal projection matrices $\mathbf{P}_i$, this exercise guarantees that the ranks (and therefore the $\chi^2$ degrees of freedom) add up to $n$.
