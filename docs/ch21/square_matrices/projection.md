# Projection Matrices

## Definition: Projection Matrix

A **projection matrix** is a square matrix $\mathbf{P}$ that maps vectors onto a subspace in such a way that applying the projection twice yields the same result as applying it once. In other words, $\mathbf{P}$ is **idempotent**:

$$
\mathbf{P}^2 = \mathbf{P}
$$

If $\mathbf{P}$ projects vectors onto a subspace $W$ of a vector space $V$, then for any vector $\mathbf{v}$ in $V$, $\mathbf{P} \mathbf{v}$ lies in $W$.

A projection matrix can be used to represent projections onto different types of subspaces, such as lines or planes. In general, a projection need not be orthogonal.
