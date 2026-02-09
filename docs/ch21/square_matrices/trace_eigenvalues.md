# Trace and Eigenvalues

## Definition: Trace

The trace of an $n \times n$ matrix $\mathbf{A}$ is the sum of its diagonal elements:

$$
\text{tr}(\mathbf{A}) = \sum_{i=1}^n \mathbf{a}_{ii}
$$

## Theorem: Trace as the Sum of Eigenvalues

!!! info "Theorem"
    For any square matrix $\mathbf{A}$ of size $n \times n$, the trace of $\mathbf{A}$ is equal to the sum of its eigenvalues (counted with algebraic multiplicity):

    $$
    \text{tr}(\mathbf{A}) = \sum_{i=1}^n \lambda_i
    $$

    where $\lambda_1, \lambda_2, \dots, \lambda_n$ are the eigenvalues of $\mathbf{A}$.

**Key Points:**

- The eigenvalues are counted with their algebraic multiplicity.
- The result holds for all square matrices, whether they are diagonalizable or not.
- The trace is invariant under similarity transformations: for any invertible matrix $\mathbf{P}$, if $\mathbf{A}' = \mathbf{P}^{-1} \mathbf{A} \mathbf{P}$, then $\text{tr}(\mathbf{A}) = \text{tr}(\mathbf{A}')$.

The last property follows from the fact that for two $n\times n$ matrices $\mathbf{A}$ and $\mathbf{B}$, $\text{tr}(\mathbf{A}\mathbf{B})=\text{tr}(\mathbf{B}\mathbf{A})$.

??? note "Proof"
    For a square matrix $\mathbf{A}$, we can write $\mathbf{A} = \mathbf{P} \mathbf{J} \mathbf{P}^{-1}$ where $\mathbf{J}$ is a block-diagonal matrix (Jordan form) with eigenvalues along the diagonal and possibly some entries of 1 on the superdiagonal.

    Since for two $n\times n$ matrices $\mathbf{C}$ and $\mathbf{D}$ we have $\text{tr}(\mathbf{C}\mathbf{D})=\text{tr}(\mathbf{D}\mathbf{C})$, the trace of $\mathbf{A}$ equals the trace of $\mathbf{J}$, which equals the sum of the eigenvalues, because the superdiagonal entries do not affect the trace:

    $$
    \text{tr}(\mathbf{A})
    = \text{tr}(\mathbf{P} \mathbf{J} \mathbf{P}^{-1})
    = \text{tr}( \mathbf{J} \mathbf{P}^{-1} \mathbf{P})
    = \text{tr}( \mathbf{J} )
    $$
