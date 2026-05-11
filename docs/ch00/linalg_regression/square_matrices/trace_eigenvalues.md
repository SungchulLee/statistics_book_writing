# Trace and Eigenvalues

The trace of a matrix -- the sum of its diagonal entries -- is one of the simplest matrix quantities to compute. Yet it carries deep information: for any square matrix, the trace equals the sum of the eigenvalues. This connection appears constantly in statistics. The trace of a covariance matrix gives the total variance, the trace of the hat matrix gives the number of estimated parameters, and the expected value of a quadratic form $\mathbf{z}^T\mathbf{A}\mathbf{z}$ can be expressed in terms of $\operatorname{tr}(\mathbf{A})$. This section develops the trace-eigenvalue relationship and its key properties.

## Definition and Basic Properties

!!! info "Definition -- Trace"
    The **trace** of a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is the sum of its diagonal entries:

    $$
    \operatorname{tr}(\mathbf{A}) = \sum_{i=1}^n a_{ii}
    $$

### Linearity

The trace is a linear function on the space of $n \times n$ matrices:

$$
\operatorname{tr}(\alpha\mathbf{A} + \beta\mathbf{B}) = \alpha\operatorname{tr}(\mathbf{A}) + \beta\operatorname{tr}(\mathbf{B})
$$

for any scalars $\alpha, \beta$ and matrices $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{n \times n}$.

### Cyclic Property

The most frequently used trace identity in statistics is the **cyclic property**.

!!! tip "Theorem -- Cyclic Property of Trace"
    For matrices $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n \times m}$:

    $$
    \operatorname{tr}(\mathbf{A}\mathbf{B}) = \operatorname{tr}(\mathbf{B}\mathbf{A})
    $$

    More generally, for matrices $\mathbf{A}_1, \dots, \mathbf{A}_k$ with compatible dimensions for the product:

    $$
    \operatorname{tr}(\mathbf{A}_1\mathbf{A}_2\cdots\mathbf{A}_k) = \operatorname{tr}(\mathbf{A}_k\mathbf{A}_1\cdots\mathbf{A}_{k-1})
    $$

**Proof.** For the two-matrix case with $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n \times m}$:

$$
\operatorname{tr}(\mathbf{A}\mathbf{B}) = \sum_{i=1}^m [\mathbf{A}\mathbf{B}]_{ii} = \sum_{i=1}^m \sum_{j=1}^n a_{ij}b_{ji} = \sum_{j=1}^n \sum_{i=1}^m b_{ji}a_{ij} = \sum_{j=1}^n [\mathbf{B}\mathbf{A}]_{jj} = \operatorname{tr}(\mathbf{B}\mathbf{A})
$$

The general case follows by induction, grouping the last matrix with the first. $\square$

!!! warning "Caution: cyclic, not arbitrary permutations"
    The cyclic property allows cyclic permutations of the factors: $\operatorname{tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \operatorname{tr}(\mathbf{C}\mathbf{A}\mathbf{B}) = \operatorname{tr}(\mathbf{B}\mathbf{C}\mathbf{A})$. It does **not** allow arbitrary reorderings: in general, $\operatorname{tr}(\mathbf{A}\mathbf{B}\mathbf{C}) \neq \operatorname{tr}(\mathbf{A}\mathbf{C}\mathbf{B})$.

### Transpose

Since transposing a matrix does not change its diagonal entries:

$$
\operatorname{tr}(\mathbf{A}^T) = \operatorname{tr}(\mathbf{A})
$$

### Trace of an Outer Product

For vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$, the outer product $\mathbf{a}\mathbf{b}^T$ is an $n \times n$ matrix with trace

$$
\operatorname{tr}(\mathbf{a}\mathbf{b}^T) = \mathbf{b}^T\mathbf{a} = \sum_{i=1}^n a_i b_i
$$

This follows from the cyclic property: $\operatorname{tr}(\mathbf{a}\mathbf{b}^T) = \operatorname{tr}(\mathbf{b}^T\mathbf{a}) = \mathbf{b}^T\mathbf{a}$, where the last equality holds because $\mathbf{b}^T\mathbf{a}$ is a $1 \times 1$ matrix (a scalar).

## Trace Equals Sum of Eigenvalues

!!! tip "Theorem -- Trace-Eigenvalue Identity"
    Let $\mathbf{A} \in \mathbb{R}^{n \times n}$ (or $\mathbb{C}^{n \times n}$) have eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_n$ (counted with algebraic multiplicity, possibly complex). Then

    $$
    \operatorname{tr}(\mathbf{A}) = \sum_{i=1}^n \lambda_i
    $$

**Proof via the characteristic polynomial.** The characteristic polynomial of $\mathbf{A}$ is

$$
p(\lambda) = \det(\lambda\mathbf{I} - \mathbf{A}) = \lambda^n - (\operatorname{tr}\mathbf{A})\lambda^{n-1} + \cdots + (-1)^n\det(\mathbf{A})
$$

The coefficient of $\lambda^{n-1}$ can be computed in two ways. From the cofactor expansion of $\det(\lambda\mathbf{I} - \mathbf{A})$, the only way to get a product of $n-1$ diagonal terms is to choose all but one diagonal entry $(\lambda - a_{ii})$, which gives a coefficient of $-(a_{11} + \cdots + a_{nn}) = -\operatorname{tr}(\mathbf{A})$.

On the other hand, factoring the characteristic polynomial over its roots gives

$$
p(\lambda) = (\lambda - \lambda_1)(\lambda - \lambda_2)\cdots(\lambda - \lambda_n)
$$

Expanding, the coefficient of $\lambda^{n-1}$ is $-(\lambda_1 + \lambda_2 + \cdots + \lambda_n)$.

Equating the two expressions: $\operatorname{tr}(\mathbf{A}) = \lambda_1 + \lambda_2 + \cdots + \lambda_n$. $\square$

**Alternative proof via similarity.** Since trace is a similarity invariant ($\operatorname{tr}(\mathbf{P}^{-1}\mathbf{A}\mathbf{P}) = \operatorname{tr}(\mathbf{A}\mathbf{P}\mathbf{P}^{-1}) = \operatorname{tr}(\mathbf{A})$ by the cyclic property), the trace of $\mathbf{A}$ equals the trace of its Jordan form. The diagonal entries of the Jordan form are precisely the eigenvalues, so $\operatorname{tr}(\mathbf{A}) = \sum_i \lambda_i$.

## Determinant Equals Product of Eigenvalues

A companion result links the determinant to eigenvalues.

!!! tip "Theorem -- Determinant-Eigenvalue Identity"
    For the same matrix $\mathbf{A}$ with eigenvalues $\lambda_1, \dots, \lambda_n$:

    $$
    \det(\mathbf{A}) = \prod_{i=1}^n \lambda_i
    $$

**Proof.** Set $\lambda = 0$ in the characteristic polynomial: $\det(-\mathbf{A}) = (-1)^n\det(\mathbf{A}) = (-\lambda_1)(-\lambda_2)\cdots(-\lambda_n) = (-1)^n\prod_i\lambda_i$. $\square$

## Example

Consider the matrix

$$
\mathbf{A} = \begin{pmatrix} 4 & 2 \\ 1 & 3 \end{pmatrix}
$$

**Trace from the definition:** $\operatorname{tr}(\mathbf{A}) = 4 + 3 = 7$.

**Eigenvalues:** The characteristic polynomial is $\lambda^2 - 7\lambda + 10 = (\lambda - 5)(\lambda - 2)$, giving $\lambda_1 = 5$ and $\lambda_2 = 2$.

**Verification:** $\lambda_1 + \lambda_2 = 5 + 2 = 7 = \operatorname{tr}(\mathbf{A})$ and $\lambda_1 \cdot \lambda_2 = 10 = \det(\mathbf{A})$.

## Applications in Statistics

### Total Variance

For a random vector $\mathbf{X} \in \mathbb{R}^p$ with covariance matrix $\boldsymbol{\Sigma}$, the **total variance** is

$$
\operatorname{tr}(\boldsymbol{\Sigma}) = \sum_{i=1}^p \sigma_{ii} = \sum_{i=1}^p \lambda_i
$$

where $\sigma_{ii} = \operatorname{Var}(X_i)$ and $\lambda_1, \dots, \lambda_p$ are the eigenvalues (the variance in each principal-component direction). The trace-eigenvalue identity shows that the total variance is the same whether computed from the marginal variances or from the principal-component variances.

### Hat Matrix and Effective Parameters

In linear regression with $\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}$ where $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$, the hat matrix $\mathbf{H}$ is idempotent ($\mathbf{H}^2 = \mathbf{H}$), so its eigenvalues are either 0 or 1. The trace gives

$$
\operatorname{tr}(\mathbf{H}) = \text{(number of eigenvalues equal to 1)} = \operatorname{rank}(\mathbf{X}) = p
$$

where $p$ is the number of estimated parameters. Similarly, $\operatorname{tr}(\mathbf{I} - \mathbf{H}) = n - p$ counts the residual degrees of freedom.

### Expected Value of Quadratic Forms

If $\mathbf{z} \sim (\boldsymbol{\mu}, \mathbf{I}_n)$ (mean $\boldsymbol{\mu}$, identity covariance), then for any symmetric matrix $\mathbf{A}$:

$$
E[\mathbf{z}^T\mathbf{A}\mathbf{z}] = \operatorname{tr}(\mathbf{A}) + \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu}
$$

When $\boldsymbol{\mu} = \mathbf{0}$, this simplifies to $E[\mathbf{z}^T\mathbf{A}\mathbf{z}] = \operatorname{tr}(\mathbf{A})$. This identity is fundamental in ANOVA, where sums of squares are quadratic forms in the data vector.

## Summary

The trace is a linear, similarity-invariant functional that equals the sum of eigenvalues. Its cyclic property $\operatorname{tr}(\mathbf{A}\mathbf{B}) = \operatorname{tr}(\mathbf{B}\mathbf{A})$ is the workhorse identity for manipulating matrix expressions in statistical proofs. In statistics, the trace connects the diagonal entries of a covariance matrix (marginal variances) to its eigenvalues (principal-component variances), counts the effective number of parameters in a projection, and computes expected values of quadratic forms.

## Exercises

**Exercise 1.**
Let $\mathbf{A} = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$. Compute $\operatorname{tr}(\mathbf{A})$ and verify it equals the sum of the eigenvalues.

??? success "Solution to Exercise 1"
    The trace is $\operatorname{tr}(\mathbf{A}) = 3 + 3 = 6$.

    The eigenvalues satisfy $\det(\mathbf{A} - \lambda\mathbf{I}) = (3-\lambda)^2 - 1 = \lambda^2 - 6\lambda + 8 = 0$, giving $\lambda_1 = 4$ and $\lambda_2 = 2$.

    Sum of eigenvalues: $4 + 2 = 6 = \operatorname{tr}(\mathbf{A})$. Also, $\det(\mathbf{A}) = 9 - 1 = 8 = 4 \times 2 = \lambda_1 \lambda_2$.

---

**Exercise 2.**
Using the cyclic property, prove that $\operatorname{tr}(\mathbf{A}\mathbf{B}) = \operatorname{tr}(\mathbf{B}\mathbf{A})$ for any matrices $\mathbf{A}$ ($m \times n$) and $\mathbf{B}$ ($n \times m$).

??? success "Solution to Exercise 2"
    The $(i,i)$ entry of $\mathbf{A}\mathbf{B}$ is $\sum_{k=1}^n a_{ik} b_{ki}$, so:

    $$
    \operatorname{tr}(\mathbf{A}\mathbf{B}) = \sum_{i=1}^m \sum_{k=1}^n a_{ik} b_{ki}
    $$

    The $(k,k)$ entry of $\mathbf{B}\mathbf{A}$ is $\sum_{i=1}^m b_{ki} a_{ik}$, so:

    $$
    \operatorname{tr}(\mathbf{B}\mathbf{A}) = \sum_{k=1}^n \sum_{i=1}^m b_{ki} a_{ik}
    $$

    Both double sums are over the same terms $a_{ik} b_{ki}$ for $i = 1, \dots, m$ and $k = 1, \dots, n$. By commutativity of addition:

    $$
    \operatorname{tr}(\mathbf{A}\mathbf{B}) = \operatorname{tr}(\mathbf{B}\mathbf{A})
    $$

    Note: $\mathbf{A}\mathbf{B}$ is $m \times m$ and $\mathbf{B}\mathbf{A}$ is $n \times n$; they may have different sizes but always have equal traces. $\square$

---

**Exercise 3.**
Show that for the hat matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$, the trace equals the number of predictors $p$ (including the intercept) using the cyclic property.

??? success "Solution to Exercise 3"
    Apply the cyclic property by grouping factors:

    $$
    \operatorname{tr}(\mathbf{H}) = \operatorname{tr}\bigl(\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\bigr) = \operatorname{tr}\bigl(\mathbf{X}^T\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\bigr) = \operatorname{tr}(\mathbf{I}_p) = p
    $$

    The cyclic rearrangement moves $\mathbf{X}^T$ from the right to the left, producing the $p \times p$ identity matrix. This result holds regardless of $n$ and the specific entries of $\mathbf{X}$.

---

**Exercise 4.**
Let $\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$ and let $\mathbf{A}$ be a symmetric idempotent matrix of rank $r$. Using the identity $E[\mathbf{z}^T\mathbf{A}\mathbf{z}] = \operatorname{tr}(\mathbf{A})$, show that $E[\mathbf{z}^T\mathbf{A}\mathbf{z}] = r$.

??? success "Solution to Exercise 4"
    Since $\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$, we have $\boldsymbol{\mu} = \mathbf{0}$, and the identity gives:

    $$
    E[\mathbf{z}^T\mathbf{A}\mathbf{z}] = \operatorname{tr}(\mathbf{A})
    $$

    Since $\mathbf{A}$ is symmetric idempotent, its eigenvalues are all 0 or 1, and the number of eigenvalues equal to 1 is the rank $r$. Therefore:

    $$
    \operatorname{tr}(\mathbf{A}) = \sum_{i=1}^n \lambda_i = r \cdot 1 + (n-r) \cdot 0 = r
    $$

    This result explains why $\text{SSE}/\sigma^2 \sim \chi^2_{n-p}$: the residual-maker matrix $\mathbf{M} = \mathbf{I} - \mathbf{H}$ has rank $n - p$, so $E[\text{SSE}/\sigma^2] = \operatorname{tr}(\mathbf{M}) = n - p$, matching the mean of the $\chi^2_{n-p}$ distribution.
