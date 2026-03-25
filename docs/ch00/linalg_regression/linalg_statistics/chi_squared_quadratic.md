# Chi-Squared Distribution and Quadratic Forms

Many test statistics in classical statistics -- the goodness-of-fit statistic, the variance ratio, the sum of squared residuals -- are quadratic forms in normal random vectors. Understanding when such a quadratic form has a chi-squared distribution is essential for deriving the exact distributions of these test statistics. The key result connects the **idempotent matrix** structure (from the previous sections) to the chi-squared distribution: if $\mathbf{A}$ is a symmetric idempotent matrix of rank $r$ and $\mathbf{z}$ is a standard normal vector, then $\mathbf{z}^T\mathbf{A}\mathbf{z} \sim \chi^2_r$.

## Review of the Chi-Squared Distribution

!!! info "Definition -- Chi-Squared Distribution"
    If $Z_1, Z_2, \dots, Z_k$ are independent standard normal random variables ($Z_i \sim N(0,1)$), then

    $$
    Q = \sum_{i=1}^k Z_i^2 \sim \chi^2_k
    $$

    The distribution has $k$ **degrees of freedom**. In vector notation, if $\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_k)$, then $\mathbf{z}^T\mathbf{z} \sim \chi^2_k$.

Key properties:

- $E[Q] = k$ and $\operatorname{Var}(Q) = 2k$
- If $Q_1 \sim \chi^2_{k_1}$ and $Q_2 \sim \chi^2_{k_2}$ are independent, then $Q_1 + Q_2 \sim \chi^2_{k_1 + k_2}$
- The chi-squared distribution is a special case of the Gamma distribution: $\chi^2_k = \text{Gamma}(k/2, 2)$

## Quadratic Forms in Normal Vectors

A **quadratic form** in a random vector $\mathbf{z} \in \mathbb{R}^n$ is an expression of the form $\mathbf{z}^T\mathbf{A}\mathbf{z}$, where $\mathbf{A} \in \mathbb{R}^{n \times n}$ is a symmetric matrix. Without loss of generality, $\mathbf{A}$ can be assumed symmetric because $\mathbf{z}^T\mathbf{A}\mathbf{z} = \mathbf{z}^T\bigl(\frac{\mathbf{A} + \mathbf{A}^T}{2}\bigr)\mathbf{z}$.

When $\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$, the distribution of $\mathbf{z}^T\mathbf{A}\mathbf{z}$ depends on the eigenvalues of $\mathbf{A}$.

### Diagonalization Approach

By the Spectral Theorem, $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ where $\mathbf{Q}$ is orthogonal and $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$. Setting $\mathbf{w} = \mathbf{Q}^T\mathbf{z}$:

$$
\mathbf{z}^T\mathbf{A}\mathbf{z} = \mathbf{w}^T\boldsymbol{\Lambda}\mathbf{w} = \sum_{i=1}^n \lambda_i W_i^2
$$

Since $\mathbf{Q}$ is orthogonal and $\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$, the rotated vector $\mathbf{w} = \mathbf{Q}^T\mathbf{z}$ also has distribution $N(\mathbf{0}, \mathbf{I}_n)$ (the standard normal is invariant under orthogonal transformations). Therefore $W_1, \dots, W_n$ are independent $N(0,1)$ random variables, and the quadratic form is a weighted sum of independent $\chi^2_1$ variables.

## The Fundamental Chi-Squared Theorem

!!! tip "Theorem -- Quadratic Form with Idempotent Matrix"
    Let $\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$ and let $\mathbf{A} \in \mathbb{R}^{n \times n}$ be a symmetric idempotent matrix of rank $r$. Then

    $$
    \mathbf{z}^T\mathbf{A}\mathbf{z} \sim \chi^2_r
    $$

**Proof.** Since $\mathbf{A}$ is symmetric and idempotent, its eigenvalues are all 0 or 1 (from the idempotent matrix theory). Since $\operatorname{rank}(\mathbf{A}) = r$, exactly $r$ eigenvalues equal 1 and $n - r$ eigenvalues equal 0.

Using the diagonalization $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$ with $\mathbf{w} = \mathbf{Q}^T\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$:

$$
\mathbf{z}^T\mathbf{A}\mathbf{z} = \sum_{i=1}^n \lambda_i W_i^2 = \sum_{i:\,\lambda_i = 1} W_i^2
$$

This is a sum of $r$ independent $\chi^2_1$ variables, so $\mathbf{z}^T\mathbf{A}\mathbf{z} \sim \chi^2_r$. $\square$

## General Necessary and Sufficient Condition

The idempotent condition is not just sufficient -- it is also necessary for obtaining a chi-squared distribution (up to scaling).

!!! tip "Theorem -- Cochran's Condition"
    Let $\mathbf{z} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$ and $\mathbf{A}$ be a symmetric $n \times n$ matrix. Then $\mathbf{z}^T\mathbf{A}\mathbf{z}/\sigma^2 \sim \chi^2_r$ if and only if $\mathbf{A}$ is idempotent with $\operatorname{rank}(\mathbf{A}) = r$.

**Proof sketch (necessity).** If $\mathbf{z}^T\mathbf{A}\mathbf{z}/\sigma^2 \sim \chi^2_r$, the moment-generating function must match that of $\chi^2_r$. The MGF of $\mathbf{z}^T\mathbf{A}\mathbf{z}/\sigma^2$ is $\prod_{i=1}^n(1 - 2\lambda_i t)^{-1/2}$, while the MGF of $\chi^2_r$ is $(1 - 2t)^{-r/2}$. Equating these requires exactly $r$ eigenvalues equal to 1 and the rest equal to 0, which means $\mathbf{A}$ is idempotent of rank $r$. $\square$

## Independence of Quadratic Forms

!!! tip "Theorem -- Craig's Theorem"
    Let $\mathbf{z} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$ and let $\mathbf{A}$ and $\mathbf{B}$ be symmetric $n \times n$ matrices. Then the quadratic forms $\mathbf{z}^T\mathbf{A}\mathbf{z}$ and $\mathbf{z}^T\mathbf{B}\mathbf{z}$ are **independent** if and only if

    $$
    \mathbf{A}\mathbf{B} = \mathbf{0}
    $$

**Proof sketch.** The joint MGF of $(\mathbf{z}^T\mathbf{A}\mathbf{z}, \mathbf{z}^T\mathbf{B}\mathbf{z})$ factors into the product of the marginal MGFs if and only if $\mathbf{A}\mathbf{B} = \mathbf{0}$. This can be seen by simultaneously diagonalizing $\mathbf{A}$ and $\mathbf{B}$: the condition $\mathbf{A}\mathbf{B} = \mathbf{0}$ ensures that the two quadratic forms involve disjoint subsets of the independent $W_i^2$ terms. $\square$

## Cochran's Theorem

Cochran's theorem combines the chi-squared and independence results into a single powerful statement about decompositions of sums of squares.

!!! tip "Theorem -- Cochran's Theorem"
    Let $\mathbf{z} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$ and suppose

    $$
    \mathbf{z}^T\mathbf{z} = \mathbf{z}^T\mathbf{A}_1\mathbf{z} + \mathbf{z}^T\mathbf{A}_2\mathbf{z} + \cdots + \mathbf{z}^T\mathbf{A}_k\mathbf{z}
    $$

    where $\mathbf{A}_1, \dots, \mathbf{A}_k$ are symmetric positive semi-definite matrices with $\operatorname{rank}(\mathbf{A}_i) = r_i$ and $r_1 + r_2 + \cdots + r_k = n$. Then the quadratic forms $\mathbf{z}^T\mathbf{A}_1\mathbf{z}/\sigma^2, \dots, \mathbf{z}^T\mathbf{A}_k\mathbf{z}/\sigma^2$ are mutually independent with $\mathbf{z}^T\mathbf{A}_i\mathbf{z}/\sigma^2 \sim \chi^2_{r_i}$.

Cochran's theorem is the theoretical engine behind the ANOVA F-test: it guarantees that the regression sum of squares and the residual sum of squares are independent chi-squared random variables (after dividing by $\sigma^2$), which is needed to form the F-statistic.

## Example -- Residual Sum of Squares

In the linear model $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$:

The residual vector is $\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}$ where $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$. The residual sum of squares is

$$
\text{SSE} = \mathbf{e}^T\mathbf{e} = \mathbf{y}^T(\mathbf{I} - \mathbf{H})\mathbf{y}
$$

Since $(\mathbf{I} - \mathbf{H})\mathbf{X}\boldsymbol{\beta} = \mathbf{0}$, we can write $\text{SSE} = \boldsymbol{\varepsilon}^T(\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}$. The matrix $\mathbf{M} = \mathbf{I} - \mathbf{H}$ is symmetric and idempotent with $\operatorname{rank}(\mathbf{M}) = n - p$. Setting $\mathbf{z} = \boldsymbol{\varepsilon}/\sigma$:

$$
\frac{\text{SSE}}{\sigma^2} = \mathbf{z}^T\mathbf{M}\mathbf{z} \sim \chi^2_{n-p}
$$

This explains why we divide SSE by $n - p$ to get an unbiased estimate of $\sigma^2$:

$$
s^2 = \frac{\text{SSE}}{n - p}, \qquad E[s^2] = \sigma^2
$$

## Non-Central Chi-Squared Distribution

When $\mathbf{z} \sim N(\boldsymbol{\mu}, \mathbf{I}_n)$ with $\boldsymbol{\mu} \neq \mathbf{0}$ and $\mathbf{A}$ is symmetric idempotent of rank $r$, the quadratic form $\mathbf{z}^T\mathbf{A}\mathbf{z}$ has a **non-central chi-squared distribution**:

$$
\mathbf{z}^T\mathbf{A}\mathbf{z} \sim \chi^2_r(\delta)
$$

where the non-centrality parameter is $\delta = \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu}$. The non-central chi-squared distribution appears in power calculations for F-tests and in the distribution of the regression sum of squares under the alternative hypothesis.

## Summary

The chi-squared distribution of quadratic forms rests on the idempotent structure of projection matrices. When $\mathbf{z}$ is standard normal and $\mathbf{A}$ is symmetric idempotent of rank $r$, the quadratic form $\mathbf{z}^T\mathbf{A}\mathbf{z}$ is chi-squared with $r$ degrees of freedom. Craig's theorem provides the independence condition ($\mathbf{A}\mathbf{B} = \mathbf{0}$), and Cochran's theorem unifies these results for decompositions of the total sum of squares. These results provide the theoretical foundation for F-tests, t-tests, and ANOVA in the linear regression framework.
