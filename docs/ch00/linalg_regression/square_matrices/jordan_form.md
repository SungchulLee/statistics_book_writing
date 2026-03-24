# Jordan Canonical Form

Not every matrix can be diagonalized. When a matrix lacks a full set of linearly independent eigenvectors, the best similarity reduction available is the **Jordan canonical form** (also called Jordan normal form). The Jordan form replaces the diagonal matrix $\boldsymbol{\Lambda}$ with a nearly diagonal matrix -- one that has eigenvalues on the main diagonal and ones (or zeros) on the superdiagonal. For statisticians, the Jordan form rarely appears in routine data analysis (because covariance matrices are symmetric and always diagonalizable), but it provides the theoretical foundation for understanding why certain matrix structures behave differently and completes the classification of square matrices up to similarity.

## Jordan Blocks

!!! info "Definition -- Jordan Block"
    A **Jordan block** of size $k$ associated with eigenvalue $\lambda$ is the $k \times k$ upper triangular matrix

    $$
    \mathbf{J}_k(\lambda) = \begin{pmatrix} \lambda & 1 & 0 & \cdots & 0 \\ 0 & \lambda & 1 & \cdots & 0 \\ \vdots & & \ddots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda & 1 \\ 0 & 0 & \cdots & 0 & \lambda \end{pmatrix}
    $$

    The diagonal entries are all $\lambda$, the superdiagonal entries are all 1, and every other entry is 0.

A $1 \times 1$ Jordan block $\mathbf{J}_1(\lambda) = (\lambda)$ is simply a scalar. When every Jordan block is $1 \times 1$, the Jordan form is diagonal and the matrix is diagonalizable.

## The Jordan Canonical Form Theorem

!!! tip "Theorem -- Jordan Canonical Form"
    Every square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ (or $\mathbb{C}^{n \times n}$) is similar to a **Jordan matrix**

    $$
    \mathbf{J} = \begin{pmatrix} \mathbf{J}_{k_1}(\lambda_1) & & \\ & \ddots & \\ & & \mathbf{J}_{k_r}(\lambda_r) \end{pmatrix}
    $$

    where $k_1 + k_2 + \cdots + k_r = n$. That is, there exists an invertible $\mathbf{P}$ such that $\mathbf{A} = \mathbf{P}\mathbf{J}\mathbf{P}^{-1}$. The Jordan form is unique up to the ordering of the blocks.

The eigenvalues $\lambda_1, \dots, \lambda_r$ need not be distinct: the same eigenvalue can appear in multiple blocks of different sizes.

## Relationship to Diagonalizability

The Jordan form provides a clean characterization of diagonalizability:

- $\mathbf{A}$ is diagonalizable if and only if every Jordan block is $1 \times 1$.
- Equivalently, $\mathbf{A}$ is diagonalizable if and only if for each eigenvalue $\lambda$, the geometric multiplicity (dimension of $\ker(\mathbf{A} - \lambda\mathbf{I})$) equals the algebraic multiplicity (multiplicity of $\lambda$ as a root of the characteristic polynomial).
- When the geometric multiplicity is strictly less than the algebraic multiplicity, at least one Jordan block for that eigenvalue has size greater than 1.

## Powers of Jordan Blocks

Computing powers of a Jordan block reveals why non-diagonalizable matrices behave differently. For a Jordan block $\mathbf{J}_k(\lambda)$:

$$
[\mathbf{J}_k(\lambda)^m]_{ij} = \begin{cases} \binom{m}{j-i}\lambda^{m-(j-i)} & \text{if } j \geq i \text{ and } j - i \leq m \\ 0 & \text{otherwise} \end{cases}
$$

In particular, when $|\lambda| < 1$, the powers $\mathbf{J}_k(\lambda)^m \to \mathbf{0}$ as $m \to \infty$, but the rate of convergence depends on the block size $k$. When $|\lambda| = 1$ and $k > 1$, the powers grow polynomially (the binomial coefficients $\binom{m}{j-i}$ grow), unlike the diagonal case where they remain bounded.

## Example

Consider the non-diagonalizable matrix from the previous section:

$$
\mathbf{A} = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}
$$

This matrix has eigenvalue $\lambda = 2$ with algebraic multiplicity 2 and geometric multiplicity 1. The matrix is already in Jordan form: it is a single $2 \times 2$ Jordan block $\mathbf{J}_2(2)$.

Its powers are

$$
\mathbf{A}^m = \begin{pmatrix} 2^m & m \cdot 2^{m-1} \\ 0 & 2^m \end{pmatrix}
$$

Notice the polynomial factor $m$ multiplying $2^{m-1}$ in the $(1,2)$ entry. A diagonal matrix with eigenvalue 2 would have $\mathbf{D}^m = \operatorname{diag}(2^m, 2^m)$ with no polynomial growth -- just pure exponential behavior.

## A Larger Example

Consider

$$
\mathbf{A} = \begin{pmatrix} 3 & 1 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 5 \end{pmatrix}
$$

The eigenvalues are $\lambda_1 = 3$ (algebraic multiplicity 2) and $\lambda_2 = 5$ (algebraic multiplicity 1). The eigenspace for $\lambda_1 = 3$ is $\ker(\mathbf{A} - 3\mathbf{I}) = \operatorname{span}\{(1, 0, 0)^T\}$, which has geometric multiplicity 1 (less than the algebraic multiplicity 2). The Jordan form is

$$
\mathbf{J} = \begin{pmatrix} 3 & 1 & 0 \\ 0 & 3 & 0 \\ 0 & 0 & 5 \end{pmatrix} = \begin{pmatrix} \mathbf{J}_2(3) & \\ & \mathbf{J}_1(5) \end{pmatrix}
$$

In this case, $\mathbf{A}$ is already in Jordan form. The Jordan form has one $2 \times 2$ block for eigenvalue 3 and one $1 \times 1$ block for eigenvalue 5.

## Generalized Eigenvectors

The columns of the change-of-basis matrix $\mathbf{P}$ in $\mathbf{A} = \mathbf{P}\mathbf{J}\mathbf{P}^{-1}$ are called **generalized eigenvectors**. For a Jordan block $\mathbf{J}_k(\lambda)$, the corresponding generalized eigenvectors $\mathbf{v}_1, \dots, \mathbf{v}_k$ satisfy:

$$
(\mathbf{A} - \lambda\mathbf{I})\mathbf{v}_1 = \mathbf{0}, \quad (\mathbf{A} - \lambda\mathbf{I})\mathbf{v}_2 = \mathbf{v}_1, \quad \dots, \quad (\mathbf{A} - \lambda\mathbf{I})\mathbf{v}_k = \mathbf{v}_{k-1}
$$

The first vector $\mathbf{v}_1$ is an ordinary eigenvector; the remaining vectors $\mathbf{v}_2, \dots, \mathbf{v}_k$ are generalized eigenvectors that form a **Jordan chain**.

## Connection to Statistics

While the Jordan form itself rarely appears in applied statistics (because the matrices encountered -- covariance matrices, hat matrices, residual-maker matrices -- are all symmetric), the theory has indirect importance:

- **Completeness of the similarity theory.** The Jordan form guarantees that every square matrix is similar to an essentially unique "canonical" matrix. This justifies the general claim that matrix properties such as trace, determinant, and eigenvalues completely characterize the similarity class.

- **Stability analysis.** In time-series models such as VAR (vector autoregressive) processes, the eigenvalues of the companion matrix determine stationarity. The Jordan form clarifies what happens at the boundary: unit-root eigenvalues with Jordan blocks larger than $1 \times 1$ produce polynomial trends rather than constant levels.

- **Matrix functions.** The formula $f(\mathbf{A}) = \mathbf{P}f(\mathbf{J})\mathbf{P}^{-1}$ extends the notion of applying a scalar function to a matrix, even when $\mathbf{A}$ is not diagonalizable. Computing $f(\mathbf{J}_k(\lambda))$ requires the first $k-1$ derivatives of $f$ evaluated at $\lambda$.

## Summary

The Jordan canonical form is the most general similarity reduction for a square matrix: every matrix is similar to a block-diagonal matrix of Jordan blocks. When every block is $1 \times 1$, the matrix is diagonalizable; otherwise, the superdiagonal ones in larger blocks capture the "deficiency" in the eigenvector count. For the symmetric matrices that dominate statistics (covariance matrices, projection matrices), the Jordan form always reduces to a diagonal, but the Jordan theory completes the theoretical picture and is needed for non-symmetric settings such as time-series companion matrices.
