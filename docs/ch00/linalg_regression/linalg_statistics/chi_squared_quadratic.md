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

## Exercises

**Exercise 1.**
Let $\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I}_n)$ and let $\mathbf{P}$ be a symmetric idempotent matrix with $\operatorname{rank}(\mathbf{P}) = r$. State the distribution of $\mathbf{Z}^T \mathbf{P} \mathbf{Z}$ and apply it: in simple linear regression with $n = 20$ observations, find the distribution of $\mathbf{e}^T \mathbf{e}/\sigma^2$ where $\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}$.

??? success "Solution to Exercise 1"
    By the fundamental chi-squared theorem, $\mathbf{Z}^T \mathbf{P} \mathbf{Z} \sim \chi^2_r$.

    For simple linear regression, $\mathbf{M} = \mathbf{I} - \mathbf{H}$ is symmetric idempotent with rank $n - 2 = 18$. Under $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I})$, the residual $\mathbf{e} = \mathbf{M}\mathbf{y} = \mathbf{M}\boldsymbol{\varepsilon}$ (using $\mathbf{M}\mathbf{X} = \mathbf{0}$). Therefore

    $$
    \frac{\mathbf{e}^T\mathbf{e}}{\sigma^2} = \frac{\boldsymbol{\varepsilon}^T\mathbf{M}\boldsymbol{\varepsilon}}{\sigma^2} = \mathbf{z}^T\mathbf{M}\mathbf{z} \sim \chi^2_{18}
    $$

    where $\mathbf{z} = \boldsymbol{\varepsilon}/\sigma \sim N(\mathbf{0}, \mathbf{I}_n)$.

---

**Exercise 2.**
Show that the MGF of $\mathbf{z}^T\mathbf{A}\mathbf{z}$ for $\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$ and symmetric $\mathbf{A}$ with eigenvalues $\lambda_1, \ldots, \lambda_n$ is

$$
M(t) = \prod_{i=1}^n (1 - 2\lambda_i t)^{-1/2}
$$

for $t$ in a neighborhood of zero. Use this to verify that the MGF reduces to $(1 - 2t)^{-r/2}$ when $\mathbf{A}$ is idempotent of rank $r$.

??? success "Solution to Exercise 2"
    Diagonalize $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$. Set $\mathbf{w} = \mathbf{Q}^T\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$. Then

    $$
    \mathbf{z}^T\mathbf{A}\mathbf{z} = \sum_{i=1}^n \lambda_i W_i^2
    $$

    The $W_i$ are independent, so the MGF factors:

    $$
    M(t) = \prod_{i=1}^n \mathbb{E}[e^{t \lambda_i W_i^2}] = \prod_{i=1}^n (1 - 2\lambda_i t)^{-1/2}
    $$

    using the $\chi^2_1$ MGF $\mathbb{E}[e^{tW^2}] = (1 - 2t)^{-1/2}$.

    If $\mathbf{A}$ is idempotent with $r$ eigenvalues equal to 1 and the rest 0, only the $r$ unit eigenvalues contribute, giving $M(t) = (1 - 2t)^{-r/2}$ — the MGF of $\chi^2_r$. $\square$

---

**Exercise 3.**
Let $\mathbf{y} \sim N(\mathbf{X}\boldsymbol{\beta}, \sigma^2\mathbf{I}_n)$. Show that under $H_0: \boldsymbol{\beta} = \mathbf{0}$, the regression sum of squares $\mathrm{SSR}/\sigma^2 = \mathbf{y}^T\mathbf{H}\mathbf{y}/\sigma^2 \sim \chi^2_p$ and is independent of $\mathrm{SSE}/\sigma^2 \sim \chi^2_{n-p}$.

??? success "Solution to Exercise 3"
    Under $H_0$, $\mathbf{y} = \boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$. Let $\mathbf{z} = \mathbf{y}/\sigma$. Then $\mathrm{SSR}/\sigma^2 = \mathbf{z}^T\mathbf{H}\mathbf{z}$ where $\mathbf{H}$ is symmetric idempotent of rank $p$, giving $\chi^2_p$.

    Similarly $\mathrm{SSE}/\sigma^2 = \mathbf{z}^T\mathbf{M}\mathbf{z}$ with $\mathbf{M}$ symmetric idempotent of rank $n - p$, giving $\chi^2_{n-p}$.

    Independence follows from Craig's theorem: $\mathbf{H}\mathbf{M} = \mathbf{H}(\mathbf{I} - \mathbf{H}) = \mathbf{H} - \mathbf{H}^2 = \mathbf{0}$. $\square$

    This is exactly the setup for the overall F-test: under $H_0$, $F = (\mathrm{SSR}/p)/(\mathrm{SSE}/(n-p)) \sim F_{p, n-p}$.

---

**Exercise 4.**
A common ANOVA decomposition splits $\mathbf{y}^T\mathbf{y}$ into two quadratic forms. Take $\mathbf{A}_1 = \mathbf{H}$, $\mathbf{A}_2 = \mathbf{I} - \mathbf{H}$. Verify that Cochran's theorem applies, and conclude the chi-squared distributions and independence for each piece.

??? success "Solution to Exercise 4"
    The decomposition is $\mathbf{y}^T\mathbf{y} = \mathbf{y}^T\mathbf{H}\mathbf{y} + \mathbf{y}^T(\mathbf{I} - \mathbf{H})\mathbf{y}$.

    Cochran's hypotheses check:

    - $\mathbf{A}_1 + \mathbf{A}_2 = \mathbf{H} + (\mathbf{I} - \mathbf{H}) = \mathbf{I}$. ✓
    - Each $\mathbf{A}_i$ is symmetric positive semi-definite. ✓
    - $\operatorname{rank}(\mathbf{A}_1) + \operatorname{rank}(\mathbf{A}_2) = p + (n - p) = n$. ✓

    Conclusion (under $\mathbf{y} \sim N(\mathbf{0}, \sigma^2\mathbf{I})$): $\mathbf{y}^T\mathbf{H}\mathbf{y}/\sigma^2 \sim \chi^2_p$ and $\mathbf{y}^T(\mathbf{I} - \mathbf{H})\mathbf{y}/\sigma^2 \sim \chi^2_{n-p}$, with the two independent. $\square$

---

**Exercise 5.**
Compute the **expectation** of the quadratic form $\mathbf{z}^T\mathbf{A}\mathbf{z}$ when $\mathbf{z} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $\mathbf{A}$ is symmetric. Then specialize to $\boldsymbol{\Sigma} = \sigma^2\mathbf{I}$ to show $\mathbb{E}[\mathbf{z}^T\mathbf{A}\mathbf{z}] = \sigma^2 \operatorname{tr}(\mathbf{A}) + \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu}$.

??? success "Solution to Exercise 5"
    Write $\mathbf{z} = \boldsymbol{\mu} + \mathbf{w}$ with $\mathbf{w} \sim N(\mathbf{0}, \boldsymbol{\Sigma})$. Expanding,

    $$
    \mathbf{z}^T\mathbf{A}\mathbf{z} = \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu} + 2\boldsymbol{\mu}^T\mathbf{A}\mathbf{w} + \mathbf{w}^T\mathbf{A}\mathbf{w}
    $$

    (using symmetry of $\mathbf{A}$). The middle term has mean zero. For the last,

    $$
    \mathbb{E}[\mathbf{w}^T\mathbf{A}\mathbf{w}] = \mathbb{E}[\operatorname{tr}(\mathbf{A}\mathbf{w}\mathbf{w}^T)] = \operatorname{tr}(\mathbf{A}\,\mathbb{E}[\mathbf{w}\mathbf{w}^T]) = \operatorname{tr}(\mathbf{A}\boldsymbol{\Sigma})
    $$

    Hence $\mathbb{E}[\mathbf{z}^T\mathbf{A}\mathbf{z}] = \operatorname{tr}(\mathbf{A}\boldsymbol{\Sigma}) + \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu}$, and with $\boldsymbol{\Sigma} = \sigma^2\mathbf{I}$ this becomes $\sigma^2 \operatorname{tr}(\mathbf{A}) + \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu}$. $\square$

    Statistical use: under $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I})$, $\mathbb{E}[\mathrm{SSE}] = \sigma^2 \operatorname{tr}(\mathbf{M}) = \sigma^2(n - p)$ since $\mathbf{M}\mathbf{X}\boldsymbol{\beta} = \mathbf{0}$. Dividing by $n - p$ produces the unbiased $\hat{\sigma}^2$.

---

**Exercise 6.**
Under the alternative $\mathbf{z} \sim N(\boldsymbol{\mu}, \mathbf{I}_n)$ with $\boldsymbol{\mu} \ne \mathbf{0}$ and $\mathbf{A}$ symmetric idempotent of rank $r$, the quadratic form $\mathbf{z}^T\mathbf{A}\mathbf{z}$ has a **non-central** chi-squared distribution $\chi^2_r(\delta)$. Identify the non-centrality parameter $\delta$ and explain its role in power calculations for the F-test.

??? success "Solution to Exercise 6"
    Decompose $\mathbf{z} = \boldsymbol{\mu} + \mathbf{w}$ with $\mathbf{w} \sim N(\mathbf{0}, \mathbf{I}_n)$. Diagonalize $\mathbf{A} = \mathbf{Q}\operatorname{diag}(\mathbf{1}_r, \mathbf{0}_{n-r})\mathbf{Q}^T$. With $\tilde{\boldsymbol{\mu}} = \mathbf{Q}^T \boldsymbol{\mu}$ and $\tilde{\mathbf{w}} = \mathbf{Q}^T \mathbf{w}$,

    $$
    \mathbf{z}^T\mathbf{A}\mathbf{z} = \sum_{i=1}^r (\tilde{\mu}_i + \tilde{W}_i)^2
    $$

    This is exactly the definition of a non-central $\chi^2_r$ with non-centrality

    $$
    \delta = \sum_{i=1}^r \tilde{\mu}_i^2 = \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu}
    $$

    **Role in F-test power:** under $H_1: \boldsymbol{\beta} \ne \mathbf{0}$, the F-numerator's $\chi^2$ becomes non-central with $\delta = \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}/\sigma^2$. Larger $\delta$ (further from the null) shifts the F-statistic distribution toward larger values, increasing rejection probability — i.e., higher power. This is the formula plugged into power calculators for sample-size planning.

