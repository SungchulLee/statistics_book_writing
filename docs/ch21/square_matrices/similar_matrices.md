# Similar Matrices

## Definition: Similar Matrices

Two $n \times n$ matrices $A$ and $B$ are **similar** if there exists an invertible $n \times n$ matrix $P$ such that

$$
B = P^{-1} A P
$$

## Intuition Behind Similar Matrices

When two $n \times n$ matrices $A$ and $B$ are similar, they represent the **same linear transformation**, but in **different bases**. The similarity transformation $B = P^{-1} A P$ is essentially a way to "relabel" or "change the perspective" of the transformation described by $A$, using a new basis defined by the columns of $P$.

**Linear Transformations and Matrices.** A matrix $A$ represents a linear transformation $T: \mathbb{R}^n \to \mathbb{R}^n$ with respect to a specific basis. If you change the basis, the same transformation $T$ is described by a different matrix $B$.

**Role of $P$.** The columns of the matrix $P$ define the new basis. $P^{-1}$ maps vectors from the new basis back to the original basis, while $P$ transforms vectors from the original basis to the new one:

$$
\begin{array}{ccc}
\text{coordinate under new basis}
&\stackrel{P}{\rightarrow}&
\text{coordinate under old basis}\\
\text{coordinate under new basis}
&\stackrel{P^{-1}}{\leftarrow}&
\text{coordinate under old basis}\\
\end{array}
$$

**The Equation $B = P^{-1} A P$.** The matrix $A$ operates on vectors in the original basis. $P$ translates the vector to the new basis, $A$ acts on this transformed vector, and $P^{-1}$ translates the result back to the new basis. The resulting transformation is described by $B$:

$$
\begin{array}{ccccccc}
\text{new basis}
&\stackrel{P}{\rightarrow}&
\text{old basis}
&\stackrel{A}{\rightarrow}&
\text{old basis}
&\stackrel{P^{-1}}{\rightarrow}&
\text{new basis}
\end{array}
$$

**Key Insight.** While the matrices $A$ and $B$ may look different, they describe the same linear operation in different "coordinate systems."

## Key Properties of Similar Matrices

!!! info "Theorem"
    Similar matrices share the following properties: same eigenvalues (including algebraic multiplicity), same trace, same determinant, and same rank.

### Same Eigenvalues

Similar matrices have the same eigenvalues (including algebraic multiplicity). This is because similarity transformations do not change the characteristic polynomial. Indeed,

$$
\det(\mathbf{B}-\lambda \mathbf{I}) = \det(\mathbf{P}^{-1} \mathbf{A} \mathbf{P} -\lambda I)
= \det(\mathbf{P}^{-1} (\mathbf{A}-\lambda \mathbf{I}) \mathbf{P} )
= \det(\mathbf{P}^{-1}) \det(\mathbf{A}-\lambda I) \det(\mathbf{P})
= \det(\mathbf{A}-\lambda \mathbf{I})
$$

### Same Trace

Similar matrices have the same trace. Note that $\text{tr}(\mathbf{C}\mathbf{D})=\text{tr}(\mathbf{D}\mathbf{C})$ for any $n\times n$ matrices $\mathbf{C}$ and $\mathbf{D}$. Suppose $\mathbf{A}$ and $\mathbf{B}$ are similar, that is, there is an invertible matrix $\mathbf{P}$ such that $\mathbf{B} = \mathbf{P}^{-1} \mathbf{A} \mathbf{P}$. Then,

$$
\text{tr}(\mathbf{B}) = \text{tr}(\mathbf{P}^{-1} \mathbf{A} \mathbf{P}) = \text{tr}( \mathbf{A} \mathbf{P} \mathbf{P}^{-1}) = \text{tr}(\mathbf{A})
$$

### Same Determinant

Similar matrices have the same determinant:

$$
\det(\mathbf{B}) = \det(\mathbf{P}^{-1} \mathbf{A} \mathbf{P})
= \det(\mathbf{P}^{-1}) \det(\mathbf{A}) \det(\mathbf{P})
= \det(\mathbf{P})^{-1} \det(\mathbf{A}) \det(\mathbf{P})
= \det(\mathbf{A})
$$

### Same Rank

Similar matrices have the same rank: $\text{rank}(\mathbf{A}) = \text{rank}(\mathbf{B})$.

??? note "Proof"
    The rank of a linear transformation, or the matrix representing it, is defined as the dimension of the image space of the transformation. This means that the rank of a linear transformation is independent of the specific matrix chosen to represent it, as the matrix depends on the choice of basis. Since similar matrices correspond to representations of the same linear transformation in different bases, their ranks must be identical.
