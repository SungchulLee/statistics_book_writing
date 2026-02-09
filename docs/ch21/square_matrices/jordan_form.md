# Jordan Canonical Form of Square Matrices

The **Jordan Canonical Form** is a representation of a square matrix $\mathbf{A}$ that simplifies its structure by expressing it in terms of its eigenvalues and generalized eigenvectors. It is particularly useful for understanding the properties of matrices that are not diagonalizable.

## Generalized Eigenvectors

If $\mathbf{A}$ is defective (does not have enough linearly independent eigenvectors to form a basis), we use **generalized eigenvectors** to complete the eigenbasis.

A **generalized eigenvector** $\mathbf{v}_g$ of rank $k$ satisfies:

$$
(\mathbf{A} - \lambda \mathbf{I})^k \mathbf{v}_g = \mathbf{0},
$$

where $k > 1$ is the smallest integer such that $\mathbf{v}_g$ lies in the null space of $(\mathbf{A} - \lambda \mathbf{I})^k$ but not $(\mathbf{A} - \lambda \mathbf{I})^{k-1}$.

### Steps to Compute Generalized Eigenvectors

**Step 1: Identify defective eigenvalues.** Compute the **geometric multiplicity** of each eigenvalue $\lambda$ (dimension of the null space of $\mathbf{A} - \lambda \mathbf{I}$). If the geometric multiplicity is less than the **algebraic multiplicity** (the multiplicity of $\lambda$ as a root of the characteristic polynomial), then $\lambda$ is defective.

**Step 2: Find generalized eigenvectors.** Compute higher powers $(\mathbf{A} - \lambda \mathbf{I})^k$ and determine vectors in the null space of $(\mathbf{A} - \lambda \mathbf{I})^k$ but not $(\mathbf{A} - \lambda \mathbf{I})^{k-1}$.

### Example: Find Eigenvectors and Generalized Eigenvectors

Let $\mathbf{A} = \begin{bmatrix} 6 & 2 \\ 0 & 6 \end{bmatrix}$.

**1. Find Eigenvalues.** Solve $\det(\mathbf{A} - \lambda \mathbf{I}) = 0$:

$$
\det \begin{bmatrix} 6-\lambda & 2 \\ 0 & 6-\lambda \end{bmatrix} = (6-\lambda)^2 = 0
$$

Thus, $\lambda = 6$ (algebraic multiplicity 2).

**2. Eigenvectors.** Solve $(\mathbf{A} - 6\mathbf{I}) \mathbf{v} = 0$:

$$
(\mathbf{A} - 6\mathbf{I}) = \begin{bmatrix} 0 & 2 \\ 0 & 0 \end{bmatrix}, \quad
\begin{bmatrix} 0 & 2 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

This gives $y = 0$, so $\mathbf{v} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ is an eigenvector.

**3. Generalized Eigenvector.** Since $\lambda = 6$ is defective (geometric multiplicity 1 < algebraic multiplicity 2), find $\mathbf{v}_g$ such that $(\mathbf{A} - 6\mathbf{I})^2 \mathbf{v}_g = 0$ and $(\mathbf{A} - 6\mathbf{I}) \mathbf{v}_g \neq 0$. From:

$$
(\mathbf{A} - 6\mathbf{I}) \mathbf{v}_g = \begin{bmatrix} 0 & 2 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 2y \\ 0 \end{bmatrix}
$$

Choose $y = 1$, $x = 0$. Then $\mathbf{v}_g = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$.

**Final Answer:** Eigenvector: $\mathbf{v} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$, Generalized eigenvector: $\mathbf{v}_g = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$.

## Definition: Jordan Canonical Form

Every square matrix $\mathbf{A}$ of size $n \times n$ can be written as

$$
\mathbf{A} = \mathbf{P} \mathbf{J} \mathbf{P}^{-1},
$$

where $\mathbf{J}$ is the **Jordan Canonical Form** of $\mathbf{A}$ (a block-diagonal matrix) and $\mathbf{P}$ is an invertible matrix whose columns are the eigenvectors and generalized eigenvectors of $\mathbf{A}$.

## Structure of the Jordan Form

The matrix $\mathbf{J}$ is a block-diagonal matrix with **Jordan blocks** on its diagonal. Each Jordan block corresponds to one eigenvalue of $\mathbf{A}$. A Jordan block has the following structure:

$$
\mathbf{J}_k(\lambda) =
\begin{bmatrix}
\lambda & 1 & 0 & \cdots & 0 \\
0 & \lambda & 1 & \cdots & 0 \\
0 & 0 & \lambda & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & 1 \\
0 & 0 & 0 & \cdots & \lambda
\end{bmatrix}
$$

where $\lambda$ is the eigenvalue, the diagonal consists of $\lambda$'s, the superdiagonal consists of 1's, and all other entries are 0.

If $\mathbf{A}$ is diagonalizable, each Jordan block has size 1, and $\mathbf{J}$ is simply a diagonal matrix.

## Key Properties of Jordan Canonical Form

**Eigenvalues on the Diagonal.** The diagonal entries of $\mathbf{J}$ are the eigenvalues of $\mathbf{A}$.

**Block Size and Multiplicity.** The size of a Jordan block corresponds to the size of a generalized eigenspace for an eigenvalue. The number of Jordan blocks for $\lambda$ equals the geometric multiplicity of $\lambda$ (the number of linearly independent eigenvectors).

## Example: Jordan Canonical Form

Consider the matrix:

$$
\mathbf{A} = \begin{bmatrix}
5 & 4 & 2 \\
0 & 5 & 2 \\
0 & 0 & 5
\end{bmatrix}
$$

The eigenvalue of $\mathbf{A}$ is $\lambda = 5$ with algebraic multiplicity 3 but only one linearly independent eigenvector. The Jordan canonical form of $\mathbf{A}$ is:

$$
\mathbf{J} = \begin{bmatrix}
5 & 1 & 0 \\
0 & 5 & 1 \\
0 & 0 & 5
\end{bmatrix}
$$

Here, $\mathbf{J}$ has a single Jordan block for $\lambda = 5$ with size 3.

## Example: More Complex Jordan Canonical Form

$$
\mathbf{A} =
\begin{bmatrix}
3 & 3 & 2 & 1 & 1 & 0 & 0 & 0 \\
0 & 3 & 1 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 2 & 1 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 2 & 1 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 2 & 1 & 0 & 1 \\
1 & 0 & 0 & 0 & 0 & 2 & 1 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & -1 & 1 \\
0 & 0 & 1 & 1 & 0 & 0 & 0 & -1
\end{bmatrix}
$$

### Step 1: Eigenvalues and Algebraic Multiplicities

The **characteristic polynomial** is:

$$
\det(\mathbf{A} - \lambda \mathbf{I}) = (\lambda - 3)^2 (\lambda - 2)^4 (\lambda + 1)^2 = 0
$$

From this polynomial: eigenvalue $\lambda = 3$ with algebraic multiplicity $m_a = 2$, eigenvalue $\lambda = 2$ with $m_a = 4$, and eigenvalue $\lambda = -1$ with $m_a = 2$.

### Step 2: Geometric Multiplicities

The **geometric multiplicity** $m_g$ of an eigenvalue is the dimension of the null space of $(\mathbf{A} - \lambda \mathbf{I})$. Solving $(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0$:

**For $\lambda = 3$:** $m_g = 1$, meaning there is only **one Jordan block** for $\lambda = 3$. Since $m_a = 2$, the corresponding Jordan block is $\begin{bmatrix} 3 & 1 \\ 0 & 3 \end{bmatrix}$.

**For $\lambda = 2$:** $m_g = 2$, meaning there are **two Jordan blocks** for $\lambda = 2$. Since $m_a = 4$, there are two possibilities: either two $2\times2$ Jordan blocks, or one $3\times3$ block and one $1\times1$ block. The analysis below shows the second possibility is correct.

**For $\lambda = -1$:** $m_g = 1$, meaning there is only **one Jordan block** for $\lambda = -1$. Since $m_a = 2$, the corresponding Jordan block is $\begin{bmatrix} -1 & 1 \\ 0 & -1 \end{bmatrix}$.

### Step 3: Jordan Block Sizes

We determine the sizes by computing the dimensions of the **generalized eigenspaces** for powers of $(\mathbf{A} - \lambda \mathbf{I})$.

**For $\lambda = 2$:**

| $k$ | $\dim(\ker((\mathbf{A} - 2\mathbf{I})^k)) - \dim(\ker((\mathbf{A} - 2\mathbf{I})^{k-1}))$ | Interpretation |
|-----|---|---|
| 1 | $2 - 0 = 2$ | **2 Jordan blocks** of size $\geq 1$ |
| 2 | $3 - 2 = 1$ | **1 Jordan block** of size $\geq 2$ |
| 3 | $4 - 3 = 1$ | **1 Jordan block** of size $\geq 3$ |
| 4 | $4 - 4 = 0$ | **0 Jordan blocks** of size $\geq 4$ |

Combining these results: **one Jordan block of size 3** and **one Jordan block of size 1** for $\lambda = 2$.

### Step 4: Assemble the Jordan Canonical Form $\mathbf{J}$

The Jordan canonical form is:

$$
\mathbf{J} =
\begin{bmatrix}
3 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 3 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 2 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 2 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 2 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 2 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & -1 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
\end{bmatrix}
$$

The block structure is:

- **Block 1** ($\lambda = 3$): $2 \times 2$ block with eigenvalue $3$ on diagonal and 1 on superdiagonal.
- **Block 2** ($\lambda = 2$): $3 \times 3$ block with eigenvalue $2$ on diagonal and 1's on superdiagonal.
- **Block 3** ($\lambda = 2$): $1 \times 1$ block with single eigenvalue $2$.
- **Block 4** ($\lambda = -1$): $2 \times 2$ block with eigenvalue $-1$ on diagonal and 1 on superdiagonal.

The block sizes are determined by the algebraic and geometric multiplicities of the eigenvalues and the structure of the generalized eigenspaces.

## When is the Jordan Form Useful?

**Non-Diagonalizable Matrices.** For matrices that are not diagonalizable, the Jordan form provides the next best structure.

**Solving Matrix Equations.** Simplifies solving differential equations or evaluating functions of matrices like $e^A$.

**Understanding Matrix Properties.** It clarifies the relationship between eigenvalues, eigenvectors, and generalized eigenvectors.
