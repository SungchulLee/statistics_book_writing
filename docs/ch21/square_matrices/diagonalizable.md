# Diagonal Form of Diagonalizable Matrices

## Eigenvectors and Eigenvalues

An **eigenvector** $\mathbf{v}$ and corresponding **eigenvalue** $\lambda$ satisfy:

$$
\mathbf{A} \mathbf{v} = \lambda \mathbf{v},
$$

where $\mathbf{A}$ is a square matrix of size $n \times n$, $\mathbf{v}$ is a non-zero column vector of size $n$, and $\lambda$ is a scalar.

### Steps to Compute Eigenvectors

**Step 1: Find the eigenvalues ($\lambda$).** Solve the **characteristic equation** $\det(\mathbf{A} - \lambda \mathbf{I}) = 0$, where $\mathbf{I}$ is the identity matrix. The roots of this equation are the eigenvalues of $\mathbf{A}$.

**Step 2: Find the eigenvectors for each $\lambda$.** Solve $(\mathbf{A} - \lambda \mathbf{I}) \mathbf{v} = \mathbf{0}$. This is a homogeneous system, and the eigenvectors $\mathbf{v}$ form the null space (kernel) of $\mathbf{A} - \lambda \mathbf{I}$.

## Definition: Diagonalizable Matrix

$\mathbf{A}$ is **diagonalizable** if it is similar to a diagonal matrix $\Lambda$, which can be represented as

$$
\Lambda = \mathbf{P}^{-1} \mathbf{A} \mathbf{P}
\quad\Rightarrow\quad
\mathbf{A} = \mathbf{P} \Lambda \mathbf{P}^{-1}
$$

where:

- $\Lambda$ is a diagonal matrix that contains the **eigenvalues** of $\mathbf{A}$ on its diagonal.
- $\mathbf{P}$ is an invertible matrix whose columns are the **eigenvectors** of $\mathbf{A}$.

## Intuition Behind Diagonalizable Matrices

**Matrix $\Lambda$.** The diagonal entries of $\Lambda$ are the eigenvalues of $\mathbf{A}$. Specifically, if $\lambda_1, \lambda_2, \dots, \lambda_n$ are the eigenvalues of $\mathbf{A}$, then:

$$
\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)
$$

**Matrix $\mathbf{P}$.** The columns of $\mathbf{P}$ are the eigenvectors of $\mathbf{A}$ corresponding to the eigenvalues in $\Lambda$. If $\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n$ are eigenvectors of $\mathbf{A}$, then:

$$
\mathbf{P} = [\mathbf{v}_1 \, \mathbf{v}_2 \, \dots \, \mathbf{v}_n]
$$

Indeed,

$$
\mathbf{A}[\mathbf{v}_1 \, \mathbf{v}_2 \, \dots \, \mathbf{v}_n]
=\mathbf{A}\mathbf{P}
= \mathbf{P} \Lambda \mathbf{P}^{-1}\mathbf{P}
= \mathbf{P} \Lambda
=[\lambda_1\mathbf{v}_1 \, \lambda_2\mathbf{v}_2 \, \dots \, \lambda_n\mathbf{v}_n]
$$

**Similarity Transformation.** The similarity transformation $\Lambda=\mathbf{P}^{-1}\mathbf{A}\mathbf{P}$ expresses $\mathbf{A}$ in a basis defined by its eigenvectors, which simplifies $\mathbf{A}$ to a diagonal form, making certain operations, like raising $\mathbf{A}$ to a power, much simpler.
