# Idempotent Matrices

## Definition: Idempotent Matrix

An **idempotent** matrix $\mathbf{A}$ is a square matrix that, when multiplied by itself, yields the same matrix:

$$
\mathbf{A}^2 = \mathbf{A}
$$

## Properties of Idempotent Matrices

**Eigenvalues.** The eigenvalues of an idempotent matrix can only be $0$ or $1$. If $\lambda$ is an eigenvalue with corresponding eigenvector $\mathbf{v}$, then $\mathbf{A} \mathbf{v} = \lambda \mathbf{v}$. Since $\mathbf{A}$ is idempotent, $\mathbf{A}^2 \mathbf{v} = \mathbf{A} \mathbf{v} = \lambda \mathbf{v}$, which implies:

$$
\lambda^2 \mathbf{v} = \lambda \mathbf{v}
$$

Therefore, $\lambda^2 = \lambda$, giving $\lambda = 0$ or $\lambda = 1$.

**Trace.** The trace of an idempotent matrix (the sum of its diagonal elements) is equal to the rank of the matrix, which is the number of non-zero eigenvalues (all of which must be 1).

## Idempotent Matrices in Statistics

In regression analysis, the **hat matrix** $\mathbf{H}$ is a well-known example of an idempotent matrix. The hat matrix is defined as

$$
\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T
$$

where $\mathbf{X}$ is the design matrix of the regression model. The hat matrix is idempotent, meaning $\mathbf{H}^2 = \mathbf{H}$. It is called the "hat" matrix because it "puts a hat on" $\mathbf{y}$ to produce the fitted values $\hat{\mathbf{y}}$:

$$
\hat{\mathbf{y}} = \mathbf{H} \mathbf{y}
$$

Indeed,

$$
\begin{array}{lll}
\mathbf{X}\theta \approx \mathbf{y}
&\Rightarrow&\mathbf{X}^T\mathbf{X}\theta = \mathbf{X}^T\mathbf{y}\\
&\Rightarrow&\hat{\theta} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}\\
&\Rightarrow&\hat{\mathbf{y}}=\mathbf{X}\hat{\theta} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}= \mathbf{H} \mathbf{y}\\
\end{array}
$$

Idempotent matrices like $\mathbf{H}$ play a key role in statistical theory, particularly in the derivation of properties of estimators and residuals in linear regression models.
