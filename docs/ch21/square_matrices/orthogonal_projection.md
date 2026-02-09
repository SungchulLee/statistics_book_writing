# Orthogonal Projection Matrices

## Definition: Orthogonal Projection Matrix

Let $S$ be a subspace of $\mathbb{R}^n$. We can choose an orthonormal basis $\{\mathbf{e}_i\}$ for $\mathbb{R}^n$, where the first $m$ vectors form an orthonormal basis for $S$. Using this basis, we define the orthogonal projection $L$ onto the subspace $S$ as:

$$
\mathbf{v} = \sum_{i=1}^n \langle \mathbf{v}, \mathbf{e}_i \rangle \mathbf{e}_i
\quad\rightarrow\quad
L(\mathbf{v}) = \sum_{i=1}^m \langle \mathbf{v}, \mathbf{e}_i \rangle \mathbf{e}_i
$$

An orthogonal projection matrix $\mathbf{P}$ is the matrix representation of this orthogonal projection $L$. It has the following properties:

**Idempotency.** Like any projection matrix: $\mathbf{P}^2 = \mathbf{P}$.

**Symmetry.** The matrix is symmetric: $\mathbf{P} = \mathbf{P}^T$. With respect to the orthonormal basis $\{\mathbf{e}_i\}$, the matrix representation is a diagonal matrix, which is symmetric. In general, the matrix representation of an orthogonal projection is similar to a diagonal matrix, which ensures it is symmetric.

## Example of Orthogonal Projection

Consider projecting a vector $\mathbf{v}$ onto a line spanned by a unit vector $\mathbf{u}$. The orthogonal projection matrix $\mathbf{P}$ is given by:

$$
\mathbf{P} = \mathbf{u} \mathbf{u}^T
$$

This matrix projects any vector $\mathbf{v}$ onto the line defined by $\mathbf{u}$ in an orthogonal manner.

## Theorem: Characterization of Orthogonal Projection Matrices

!!! info "Theorem"
    $\mathbf{P}$ is an orthogonal projection matrix if and only if it is idempotent and symmetric. In this case, the rank of $\mathbf{P}$ is the dimension of the projection subspace.

??? note "Proof"
    If $\mathbf{P}$ is an orthogonal projection matrix, then it is of course idempotent and symmetric. Conversely, suppose $\mathbf{P}$ is idempotent and symmetric.

    Since $\mathbf{P}$ is idempotent, $\mathbf{P}^2 = \mathbf{P}$, which implies that for any vector $\mathbf{y} \in \mathbb{R}^n$, applying $\mathbf{P}$ twice has the same effect as applying it once: $\mathbf{P}(\mathbf{P}\mathbf{y}) = \mathbf{P}\mathbf{y}$.

    For any vector $\mathbf{y}$, we decompose it as $\mathbf{y} = \mathbf{P}\mathbf{y} + \mathbf{r}$ where $\mathbf{r} = \mathbf{y} - \mathbf{P}\mathbf{y}$. We see that $\mathbf{P}\mathbf{y}$ is in the range space of $\mathbf{P}$ and does not change under $\mathbf{P}$. We claim that $\mathbf{r}$ is orthogonal to the range space of $\mathbf{P}$. We need to show that the **residual** $\mathbf{r}$ is orthogonal to the **projected vector** $\mathbf{P}\mathbf{y}$:

    $$
    \begin{array}{lll}
    \mathbf{r}^T (\mathbf{P}\mathbf{y})
    &=& (\mathbf{y} - \mathbf{P}\mathbf{y})^T (\mathbf{P}\mathbf{y})\\
    &=& \mathbf{y}^T \mathbf{P}\mathbf{y} - (\mathbf{P}\mathbf{y})^T (\mathbf{P}\mathbf{y})\\
    &=& \mathbf{y}^T \mathbf{P}\mathbf{y} - \mathbf{y}^T\mathbf{P}^T \mathbf{P}\mathbf{y}\\
    &=& \mathbf{y}^T \mathbf{P}\mathbf{y} - \mathbf{y}^T\mathbf{P} \mathbf{P}\mathbf{y}\\
    &=& \mathbf{y}^T \mathbf{P}\mathbf{y} - \mathbf{y}^T \mathbf{P}\mathbf{y}\\
    &=& 0
    \end{array}
    $$

    This implies that $\mathbf{r} = \mathbf{y} - \mathbf{P}\mathbf{y}$ is orthogonal to $\mathbf{P}\mathbf{y}$.

    **Rank of $\mathbf{P}$.** The rank of a projection matrix $\mathbf{P}$ is the dimension of the subspace onto which it projects. The column space of $\mathbf{P}$ is exactly the subspace $S$ onto which it projects. Therefore, the rank of $\mathbf{P}$ is $k$, which is the dimension of $S$.

## Orthogonal Projection Matrix in Linear Algebra

Suppose we have a subspace $S$ of $\mathbb{R}^n$ and we want to project a vector $\mathbf{y} \in \mathbb{R}^n$ onto $S$. If the columns of a matrix $\mathbf{X}$ form a basis for the subspace $S$, the orthogonal projection matrix $\mathbf{P}$ that projects any vector onto $S$ can be expressed as:

$$
\mathbf{P} = \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T
$$

where $\mathbf{X}$ is an $n \times k$ matrix whose columns are the basis vectors of $S$, and $(\mathbf{X}^T \mathbf{X})^{-1}$ is the inverse of the Gram matrix.

$$
\begin{array}{lll}
\mathbf{X}\beta\approx\mathbf{y}
&\Rightarrow&\mathbf{X}^T\mathbf{X}\beta=\mathbf{X}^T\mathbf{y}\\
&\Rightarrow&\hat{\beta}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}\\
&\Rightarrow&\mathbf{X}\hat{\beta}=\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}=\mathbf{P}\mathbf{y}\\
&\Rightarrow&\mathbf{P} = \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T
\end{array}
$$

## Orthogonal Projection Matrix in Linear Regression (Hat Matrix)

In linear regression, the **hat matrix** $\mathbf{H}$ is an example of a projection matrix. It projects the observed data $\mathbf{y}$ onto the column space of the design matrix $\mathbf{X}$:

$$
\hat{\mathbf{y}} = \mathbf{H} \mathbf{y}, \quad \text{where} \quad \mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T
$$

The hat matrix $\mathbf{H}$ is **symmetric** and **idempotent**. It "puts a hat" on $\mathbf{y}$ to give the fitted values $\hat{\mathbf{y}}$, hence the name "hat matrix." It projects $\mathbf{y}$ onto the **column space** of $\mathbf{X}$, meaning that $\hat{\mathbf{y}}$ lies within the span of the predictor variables.

If $\mathbf{X}$ does not have full column rank (i.e., the columns of $\mathbf{X}$ are not independent), then $(\mathbf{X}^T \mathbf{X})^{-1}$ is replaced by the Moore-Penrose pseudo inverse $(\mathbf{X}^T \mathbf{X})^{+}$.

## Moore-Penrose Pseudo-Inverse

The pseudo-inverse is typically computed using the **singular value decomposition (SVD)** of $\mathbf{X}$. For $\mathbf{X}$ of size $m \times n$:

$$
\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
$$

where $\mathbf{U}$ is an $m \times m$ orthogonal matrix, $\mathbf{\Sigma}$ is an $m \times n$ diagonal matrix of singular values, and $\mathbf{V}$ is an $n \times n$ orthogonal matrix.

The pseudo-inverse of $\mathbf{X}$ is:

$$
\mathbf{X}^+ = \mathbf{V} \mathbf{\Sigma}^+ \mathbf{U}^T
$$

where $\mathbf{\Sigma}^+$ is obtained by taking the reciprocal of the non-zero singular values in $\mathbf{\Sigma}$ and transposing the resulting matrix.

The pseudo-inverse provides a stable solution even when $\mathbf{X}$ is rank-deficient, handles ill-conditioned or singular data matrices, ensures numerical stability with multicollinearity, and extends linear regression to underdetermined or overdetermined systems.

## Theorem: Rank of $\mathbf{H}$

!!! info "Theorem"
    $$\text{rank}(\mathbf{H}) = \text{rank}(\mathbf{X})$$

??? note "Proof"
    Suppose $\mathbf{X}$ has the full column rank $k$. Then, $\mathbf{H}$ is the orthogonal projection matrix projecting onto the column space of $\mathbf{X}$. So, the rank of $\mathbf{H}$ is also $k$. Suppose $\mathbf{X}$ has column rank $c<k$. Then, $\mathbf{H}$ is still the orthogonal projection matrix projecting onto the column space of $\mathbf{X}$. So, the rank of $\mathbf{H}$ is also $c$.

## Theorem: Rank of $\mathbf{I}-\mathbf{H}$

!!! info "Theorem"
    $$\text{rank}(\mathbf{I}-\mathbf{H}) = n-\text{rank}(\mathbf{H}) = n-\text{rank}(\mathbf{X})$$

??? note "Proof"
    $\mathbf{I}-\mathbf{H}$ is itself an orthogonal projection, which complements the given orthogonal projection $\mathbf{H}$.
