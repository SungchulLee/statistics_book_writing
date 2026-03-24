# Idempotent Matrices

An operation that produces the same result when applied twice is called **idempotent**. In matrix algebra, idempotent matrices satisfy $\mathbf{A}^2 = \mathbf{A}$: applying the transformation a second time does nothing new. This property characterizes projections, and projection matrices are everywhere in regression. The hat matrix $\mathbf{H}$ that produces fitted values and the residual-maker matrix $\mathbf{M} = \mathbf{I} - \mathbf{H}$ that produces residuals are both idempotent. The eigenvalue structure of idempotent matrices (only 0s and 1s) directly yields degrees-of-freedom counts in ANOVA and regression theory.

## Definition

!!! info "Definition -- Idempotent Matrix"
    A square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is **idempotent** if

    $$
    \mathbf{A}^2 = \mathbf{A}
    $$

    Equivalently, $\mathbf{A}(\mathbf{A} - \mathbf{I}) = \mathbf{0}$.

The identity matrix $\mathbf{I}$ and the zero matrix $\mathbf{0}$ are trivially idempotent. The interesting cases arise from projection matrices, which are idempotent but neither $\mathbf{I}$ nor $\mathbf{0}$.

## Eigenvalues of Idempotent Matrices

!!! tip "Theorem -- Eigenvalues Are 0 or 1"
    If $\mathbf{A}$ is idempotent and $\lambda$ is an eigenvalue of $\mathbf{A}$, then $\lambda \in \{0, 1\}$.

**Proof.** Let $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ with $\mathbf{v} \neq \mathbf{0}$. Then

$$
\mathbf{A}^2\mathbf{v} = \mathbf{A}(\mathbf{A}\mathbf{v}) = \mathbf{A}(\lambda\mathbf{v}) = \lambda\mathbf{A}\mathbf{v} = \lambda^2\mathbf{v}
$$

Since $\mathbf{A}^2 = \mathbf{A}$, we also have $\mathbf{A}^2\mathbf{v} = \mathbf{A}\mathbf{v} = \lambda\mathbf{v}$. Equating the two expressions gives $\lambda^2\mathbf{v} = \lambda\mathbf{v}$, so $(\lambda^2 - \lambda)\mathbf{v} = \mathbf{0}$. Since $\mathbf{v} \neq \mathbf{0}$, we must have $\lambda^2 - \lambda = 0$, which gives $\lambda(\lambda - 1) = 0$. $\square$

## Trace Equals Rank

!!! tip "Theorem -- Trace-Rank Identity for Idempotent Matrices"
    If $\mathbf{A} \in \mathbb{R}^{n \times n}$ is idempotent, then

    $$
    \operatorname{tr}(\mathbf{A}) = \operatorname{rank}(\mathbf{A})
    $$

**Proof.** Since the eigenvalues of $\mathbf{A}$ are either 0 or 1, and $\operatorname{tr}(\mathbf{A}) = \sum_{i=1}^n \lambda_i$, the trace counts the number of eigenvalues equal to 1. The rank of a matrix equals the number of nonzero eigenvalues (counted with algebraic multiplicity). For an idempotent matrix, the nonzero eigenvalues are exactly the eigenvalues equal to 1. Therefore $\operatorname{tr}(\mathbf{A}) = \operatorname{rank}(\mathbf{A})$. $\square$

This result has a direct statistical interpretation: the trace of the hat matrix equals the number of parameters, and the trace of the residual-maker matrix equals the residual degrees of freedom.

## Key Properties

Several useful properties follow directly from the definition.

### Complement Is Idempotent

If $\mathbf{A}$ is idempotent, then $\mathbf{I} - \mathbf{A}$ is also idempotent:

$$
(\mathbf{I} - \mathbf{A})^2 = \mathbf{I} - 2\mathbf{A} + \mathbf{A}^2 = \mathbf{I} - 2\mathbf{A} + \mathbf{A} = \mathbf{I} - \mathbf{A}
$$

This is the algebraic reason that both the hat matrix $\mathbf{H}$ and the residual-maker $\mathbf{I} - \mathbf{H}$ are projections.

### Rank Complement

Since $\operatorname{tr}(\mathbf{I} - \mathbf{A}) = n - \operatorname{tr}(\mathbf{A})$:

$$
\operatorname{rank}(\mathbf{I} - \mathbf{A}) = n - \operatorname{rank}(\mathbf{A})
$$

### Product of Commuting Idempotent Matrices

If $\mathbf{A}$ and $\mathbf{B}$ are both idempotent and $\mathbf{A}\mathbf{B} = \mathbf{B}\mathbf{A}$, then $\mathbf{A}\mathbf{B}$ is idempotent:

$$
(\mathbf{A}\mathbf{B})^2 = \mathbf{A}\mathbf{B}\mathbf{A}\mathbf{B} = \mathbf{A}\mathbf{A}\mathbf{B}\mathbf{B} = \mathbf{A}^2\mathbf{B}^2 = \mathbf{A}\mathbf{B}
$$

### Column Space Is the Set of Fixed Points

A vector $\mathbf{x}$ is in the column space of an idempotent matrix $\mathbf{A}$ if and only if $\mathbf{A}\mathbf{x} = \mathbf{x}$. That is, the column space of $\mathbf{A}$ is exactly the set of vectors that are unchanged by the transformation.

**Proof.** If $\mathbf{x} = \mathbf{A}\mathbf{y}$ for some $\mathbf{y}$, then $\mathbf{A}\mathbf{x} = \mathbf{A}^2\mathbf{y} = \mathbf{A}\mathbf{y} = \mathbf{x}$. Conversely, if $\mathbf{A}\mathbf{x} = \mathbf{x}$, then $\mathbf{x}$ is in the column space of $\mathbf{A}$ (take $\mathbf{y} = \mathbf{x}$). $\square$

## Example

Consider the $3 \times 3$ matrix

$$
\mathbf{A} = \frac{1}{3}\begin{pmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{pmatrix}
$$

**Idempotency check:** Each row of $\mathbf{A}$ is $\frac{1}{3}(1, 1, 1)$, and multiplying $\mathbf{A}$ by any column of $\mathbf{A}$ yields $\frac{1}{3} \cdot \frac{1}{3}(1 + 1 + 1) = \frac{1}{3}$, so $\mathbf{A}^2 = \mathbf{A}$.

**Trace and rank:** $\operatorname{tr}(\mathbf{A}) = \frac{1}{3} + \frac{1}{3} + \frac{1}{3} = 1 = \operatorname{rank}(\mathbf{A})$.

**Eigenvalues:** The eigenvalues are $\lambda_1 = 1$ (with eigenvector $(1, 1, 1)^T$) and $\lambda_2 = \lambda_3 = 0$.

**Geometric interpretation:** This matrix projects every vector onto the span of $\mathbf{1}_3 = (1, 1, 1)^T$, mapping any vector to the vector whose entries are all equal to the mean of the original entries. In regression, this is the hat matrix for the intercept-only model.

## Idempotent Matrices in Regression

### The Hat Matrix

In the linear model $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ with design matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ of full column rank:

$$
\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T
$$

The hat matrix satisfies $\mathbf{H}^2 = \mathbf{H}$ (idempotent) and $\mathbf{H}^T = \mathbf{H}$ (symmetric). Its trace gives the number of parameters: $\operatorname{tr}(\mathbf{H}) = p$.

### The Residual-Maker Matrix

The residual vector is $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = (\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{M}\mathbf{y}$. The matrix $\mathbf{M} = \mathbf{I} - \mathbf{H}$ is idempotent with $\operatorname{tr}(\mathbf{M}) = n - p$, which is the residual degrees of freedom.

### Degrees of Freedom via Trace

The ANOVA decomposition

$$
\mathbf{y}^T\mathbf{y} = \hat{\mathbf{y}}^T\hat{\mathbf{y}} + \mathbf{e}^T\mathbf{e}
$$

(when the model includes an intercept and we center $\mathbf{y}$) splits the total sum of squares into model and residual sums of squares. The degrees of freedom associated with each term are $\operatorname{tr}(\mathbf{H}) = p$ and $\operatorname{tr}(\mathbf{M}) = n - p$, summing to $n$.

## Summary

Idempotent matrices satisfy $\mathbf{A}^2 = \mathbf{A}$, have eigenvalues restricted to 0 and 1, and obey $\operatorname{tr}(\mathbf{A}) = \operatorname{rank}(\mathbf{A})$. The complement $\mathbf{I} - \mathbf{A}$ is also idempotent with complementary rank. In regression, the hat matrix and residual-maker matrix are both idempotent, and their traces directly give the degrees of freedom used in F-tests, t-tests, and confidence intervals.
