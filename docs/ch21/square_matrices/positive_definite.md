# Positive Definite Matrices

## Definition: Positive Definite Matrix

A matrix $\mathbf{A}$ is **positive definite** if it is **symmetric** and for any non-zero vector $\mathbf{v} \in \mathbb{R}^n$, the quadratic form $\mathbf{v}^T \mathbf{A} \mathbf{v}$ is **strictly positive**:

$$
\mathbf{v}^T \mathbf{A} \mathbf{v} > 0 \quad \text{for all } \mathbf{v} \neq \mathbf{0}
$$

## Theorem: Invertibility of Positive Definite Matrix

!!! info "Theorem"
    A positive definite matrix $\mathbf{A}$ is always invertible.

??? note "Proof"
    Since $\mathbf{A}$ is symmetric, $\mathbf{A}$ can be diagonalized with real eigenvalues, meaning there exists an orthogonal matrix $\mathbf{P}$ and a diagonal matrix $\Lambda$ such that:

    $$
    \mathbf{A} = \mathbf{P} \Lambda \mathbf{P}^T
    $$

    Since $\mathbf{A}$ is positive definite, all eigenvalues are strictly positive. So, $\Lambda$ is invertible. The two matrices $\mathbf{P}$ and $\mathbf{P}^T$ are invertible since $\mathbf{P}^T\mathbf{P}=\mathbf{I}$. Therefore, $\mathbf{A} = \mathbf{P} \Lambda \mathbf{P}^T$ is invertible.
