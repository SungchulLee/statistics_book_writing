# Similar Matrices

A single linear transformation can be described by many different matrices, one for each choice of basis. Two matrices that represent the same transformation are called **similar**. Recognizing similarity lets us replace a complicated matrix with a simpler one -- such as a diagonal or Jordan form -- without changing the transformation's intrinsic properties. In regression and multivariate statistics, similarity underlies every change-of-basis argument: diagonalizing a covariance matrix, rotating into principal-component coordinates, or simplifying a quadratic form.

## Definition

!!! info "Definition -- Similar Matrices"
    Two square matrices $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{n \times n}$ are **similar** if there exists an invertible matrix $\mathbf{P} \in \mathbb{R}^{n \times n}$ such that

    $$
    \mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}
    $$

    Equivalently, $\mathbf{A} = \mathbf{P}\mathbf{B}\mathbf{P}^{-1}$. The matrix $\mathbf{P}$ is called a **change-of-basis matrix**.

The geometric intuition is straightforward. If $\mathbf{A}$ represents a linear map $T: \mathbb{R}^n \to \mathbb{R}^n$ with respect to the standard basis, and $\mathbf{P}$ collects the vectors of a new basis as its columns, then $\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$ is the matrix of the same map $T$ expressed in the new basis.

## Similarity Is an Equivalence Relation

Similarity partitions the set of $n \times n$ matrices into equivalence classes.

**Reflexive.** Every matrix is similar to itself via $\mathbf{P} = \mathbf{I}$:

$$
\mathbf{A} = \mathbf{I}^{-1}\mathbf{A}\mathbf{I}
$$

**Symmetric.** If $\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$, then $\mathbf{A} = \mathbf{Q}^{-1}\mathbf{B}\mathbf{Q}$ where $\mathbf{Q} = \mathbf{P}^{-1}$.

**Transitive.** If $\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$ and $\mathbf{C} = \mathbf{Q}^{-1}\mathbf{B}\mathbf{Q}$, then

$$
\mathbf{C} = \mathbf{Q}^{-1}\mathbf{P}^{-1}\mathbf{A}\mathbf{P}\mathbf{Q} = (\mathbf{P}\mathbf{Q})^{-1}\mathbf{A}(\mathbf{P}\mathbf{Q})
$$

so $\mathbf{C}$ is similar to $\mathbf{A}$ via $\mathbf{R} = \mathbf{P}\mathbf{Q}$.

## Invariants Under Similarity

The central reason similarity matters is that many important matrix quantities are **invariants** -- they take the same value for every matrix in a similarity class.

!!! tip "Theorem -- Similarity Invariants"
    If $\mathbf{B} = \mathbf{P}^{-1}\mathbf{A}\mathbf{P}$, then $\mathbf{A}$ and $\mathbf{B}$ share the following properties:

    1. **Eigenvalues** (including algebraic multiplicities)
    2. **Characteristic polynomial**: $\det(\mathbf{B} - \lambda\mathbf{I}) = \det(\mathbf{A} - \lambda\mathbf{I})$
    3. **Trace**: $\operatorname{tr}(\mathbf{B}) = \operatorname{tr}(\mathbf{A})$
    4. **Determinant**: $\det(\mathbf{B}) = \det(\mathbf{A})$
    5. **Rank**: $\operatorname{rank}(\mathbf{B}) = \operatorname{rank}(\mathbf{A})$
    6. **Minimal polynomial**

### Proof Sketch (Characteristic Polynomial)

The characteristic polynomial of $\mathbf{B}$ is

$$
\det(\mathbf{B} - \lambda\mathbf{I}) = \det(\mathbf{P}^{-1}\mathbf{A}\mathbf{P} - \lambda\mathbf{P}^{-1}\mathbf{I}\mathbf{P})
$$

Factor $\mathbf{P}^{-1}$ on the left and $\mathbf{P}$ on the right:

$$
= \det\!\bigl(\mathbf{P}^{-1}(\mathbf{A} - \lambda\mathbf{I})\mathbf{P}\bigr) = \det(\mathbf{P}^{-1})\,\det(\mathbf{A} - \lambda\mathbf{I})\,\det(\mathbf{P})
$$

Since $\det(\mathbf{P}^{-1})\det(\mathbf{P}) = 1$, the characteristic polynomials are identical. Because eigenvalues are the roots of the characteristic polynomial, the eigenvalues coincide. The trace equals the sum of the eigenvalues (with multiplicity) and the determinant equals their product, so both are invariant as well. $\square$

### Proof Sketch (Rank)

For any invertible $\mathbf{P}$, the map $\mathbf{x} \mapsto \mathbf{P}\mathbf{x}$ is a bijection on $\mathbb{R}^n$. Therefore $\dim(\operatorname{col}(\mathbf{B})) = \dim(\operatorname{col}(\mathbf{P}^{-1}\mathbf{A}\mathbf{P})) = \dim(\operatorname{col}(\mathbf{A}))$. $\square$

## Properties That Are Not Invariants

Not every matrix property is preserved under similarity. In particular:

- **Symmetry** is not invariant: a symmetric matrix can be similar to a non-symmetric matrix (the change-of-basis matrix $\mathbf{P}$ need not be orthogonal).
- **Positive definiteness** is not invariant under general similarity, although the eigenvalue characterization of positive definiteness is invariant.
- **Individual matrix entries** obviously change.

When the change-of-basis matrix $\mathbf{P}$ is restricted to be orthogonal ($\mathbf{P}^T = \mathbf{P}^{-1}$), the resulting **orthogonal similarity** $\mathbf{B} = \mathbf{P}^T\mathbf{A}\mathbf{P}$ does preserve symmetry, which is why the spectral theorem produces an orthogonal diagonalization.

## Example

Consider the matrix

$$
\mathbf{A} = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}
$$

The eigenvalues are found from $\det(\mathbf{A} - \lambda\mathbf{I}) = (4 - \lambda)(3 - \lambda) - 2 = \lambda^2 - 7\lambda + 10 = (\lambda - 5)(\lambda - 2) = 0$, giving $\lambda_1 = 5$ and $\lambda_2 = 2$.

Eigenvectors: for $\lambda_1 = 5$, solve $(\mathbf{A} - 5\mathbf{I})\mathbf{v} = \mathbf{0}$, yielding $\mathbf{v}_1 = (1, 1)^T$. For $\lambda_2 = 2$, solve $(\mathbf{A} - 2\mathbf{I})\mathbf{v} = \mathbf{0}$, yielding $\mathbf{v}_2 = (1, -2)^T$.

Form $\mathbf{P} = \begin{pmatrix} 1 & 1 \\ 1 & -2 \end{pmatrix}$. Then

$$
\mathbf{P}^{-1}\mathbf{A}\mathbf{P} = \begin{pmatrix} 5 & 0 \\ 0 & 2 \end{pmatrix} = \boldsymbol{\Lambda}
$$

The diagonal matrix $\boldsymbol{\Lambda}$ is similar to $\mathbf{A}$, and we can verify that both matrices share $\operatorname{tr} = 7$, $\det = 10$, and eigenvalues $\{5, 2\}$.

## Connection to Statistics

Similar matrices appear throughout multivariate statistics:

- **Spectral decomposition of covariance matrices.** If $\boldsymbol{\Sigma} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$, then $\boldsymbol{\Sigma}$ is similar to $\boldsymbol{\Lambda}$ (via the orthogonal matrix $\mathbf{Q}$). Working in the eigenbasis simplifies computation: $\operatorname{tr}(\boldsymbol{\Sigma}) = \sum_i \lambda_i$ gives the total variance, and $\det(\boldsymbol{\Sigma}) = \prod_i \lambda_i$ measures the generalized variance.

- **Simplifying quadratic forms.** The Mahalanobis distance $(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})$ can be analyzed by changing to the eigenbasis, where $\boldsymbol{\Sigma}^{-1}$ becomes diagonal. This is the basis for deriving the chi-squared distribution of quadratic forms in normal random vectors.

- **Invariance of the hat matrix trace.** In regression, $\operatorname{tr}(\mathbf{H}) = p$ regardless of how the predictor variables are coded or scaled, because re-parameterization corresponds to a similarity transformation of $\mathbf{X}^T\mathbf{X}$.

## Summary

Two matrices are similar when they represent the same linear transformation in different bases. Similar matrices share all intrinsic properties of the transformation -- eigenvalues, trace, determinant, rank, and characteristic polynomial -- while differing only in the choice of coordinate system. Diagonalization (the next topic) is the most important special case: finding a basis in which the matrix becomes diagonal, making computation and interpretation straightforward.
