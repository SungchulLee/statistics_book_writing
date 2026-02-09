# Chi-Squared Distribution and Quadratic Forms

## Distribution of a Quadratic Form of a Normal Random Vector

!!! info "Theorem"
    If $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2 \mathbf{I})$ is a normal random vector of length $n$, and $\mathbf{A}$ is a symmetric, idempotent matrix of rank $r$, then the quadratic form $\mathbf{Q} = \boldsymbol{\varepsilon}^T \mathbf{A} \boldsymbol{\varepsilon}$ follows a scaled chi-squared distribution with $r$ degrees of freedom:

    $$
    \mathbf{Q} \sim \sigma^2 \chi^2_r
    $$

??? note "Proof"
    Since $\mathbf{A}$ is idempotent and symmetric, it is a projection matrix. Since $\mathbf{A}$ is of rank $r$, it is actually a projection matrix onto a subspace of dimension $r$.

    When a normally distributed random vector $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2 \mathbf{I})$ is involved in a quadratic form with such a matrix, the resulting quadratic form represents the sum of the squares of projections onto $r$ orthogonal directions.

    Therefore, the quadratic form $\mathbf{Q} = \boldsymbol{\varepsilon}^T \mathbf{A} \boldsymbol{\varepsilon}$ follows a chi-squared distribution with degrees of freedom equal to the rank of the idempotent matrix (i.e., $r$), scaled by $\sigma^2$.
