# Symmetric Matrices

## Definition: Symmetric Matrix

A **symmetric** matrix $\mathbf{A}$ is a square matrix such that

$$
\mathbf{A}^T = \mathbf{A}
$$

## Spectral Theorem for Symmetric Matrices

!!! info "Theorem (Spectral Theorem)"
    If $\mathbf{A}$ is a **symmetric matrix** (i.e., $\mathbf{A} = \mathbf{A}^T$), then $\mathbf{A}$ is **diagonalizable**. Moreover, there exists an **orthogonal matrix** $\mathbf{P}$ (with $\mathbf{P}^T = \mathbf{P}^{-1}$) and a **diagonal matrix** $\Lambda$ such that:

    $$
    \mathbf{A} = \mathbf{P} \Lambda \mathbf{P}^T
    $$

    where $\mathbf{P}$ has **orthonormal eigenvectors** of $\mathbf{A}$ as columns, and $\Lambda$ contains the **real eigenvalues** of $\mathbf{A}$.

## Properties Guaranteed by the Spectral Theorem

**Real Eigenvalues.** All eigenvalues of a symmetric matrix are **real**. The diagonal elements of $\Lambda$ are real numbers.

**Orthogonal Eigenvectors.** The eigenvectors of a symmetric matrix corresponding to distinct eigenvalues are **orthogonal**. Even in the case of repeated eigenvalues, it is possible to find a set of **orthonormal** eigenvectors.

**Diagonalization with Orthogonal Matrix $\mathbf{P}$.** The symmetric matrix $\mathbf{A}$ can be diagonalized as $\mathbf{A} = \mathbf{P} \Lambda \mathbf{P}^T$, where $\mathbf{P}$ is an orthogonal matrix with $\mathbf{P}^T = \mathbf{P}^{-1}$, and the columns $\mathbf{v}_i$ of $\mathbf{P} = [\mathbf{v}_1 \, \mathbf{v}_2 \, \dots \, \mathbf{v}_n]$ are orthonormal eigenvectors of $\mathbf{A}$. Indeed,

$$
\mathbf{A}[\mathbf{v}_1 \, \mathbf{v}_2 \, \dots \, \mathbf{v}_n]
=\mathbf{A}\mathbf{P}
= \mathbf{P} \Lambda \mathbf{P}^{T}\mathbf{P}
= \mathbf{P} \Lambda
=[\lambda_1\mathbf{v}_1 \, \lambda_2\mathbf{v}_2 \, \dots \, \lambda_n\mathbf{v}_n]
$$

**Positive Definite and Semidefinite Matrices.** The Spectral Theorem also plays an important role in understanding positive definite and positive semidefinite matrices. If all the eigenvalues of a symmetric matrix $\mathbf{A}$ are positive, then $\mathbf{A}$ is **positive definite**. If all eigenvalues are non-negative, the matrix is **positive semidefinite**.

## Geometric Interpretation

The Spectral Theorem implies that a symmetric matrix represents a linear transformation that can be described in terms of **stretching or shrinking** along certain **orthogonal directions**. The directions are given by the eigenvectors, and the amount of stretching or shrinking is given by the corresponding eigenvalues.

In practical terms, for any symmetric matrix, it is possible to find a set of orthonormal eigenvectors that form a basis of the vector space. This makes symmetric matrices particularly easy to work with, as they can be fully understood in terms of these eigenvectors and eigenvalues.
