# Sampling Distributions (General Ordinary Least Squares)

The simple regression results from the previous section extend to the general multiple regression model using matrix algebra. In the general case, the OLS estimator $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ is a multivariate normal vector, the residual sum of squares divided by $\sigma^2$ is chi-squared, and the two are independent. These three facts -- normality, chi-squared, and independence -- combine to produce the t-statistics for individual coefficients and the F-statistic for testing groups of coefficients. The proofs rely on the projection-matrix and quadratic-form theory developed earlier in this chapter.

## The General Linear Model

The model in matrix form is

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

where:

- $\mathbf{y} \in \mathbb{R}^n$ is the response vector.
- $\mathbf{X} \in \mathbb{R}^{n \times p}$ is the fixed design matrix with $\operatorname{rank}(\mathbf{X}) = p$ (full column rank, $p \leq n$).
- $\boldsymbol{\beta} \in \mathbb{R}^p$ is the unknown parameter vector.
- $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$: the errors are independent and identically distributed normal with mean 0 and common variance $\sigma^2$.

From these assumptions, $\mathbf{y} \sim N(\mathbf{X}\boldsymbol{\beta}, \sigma^2\mathbf{I}_n)$.

## The OLS Estimator

The OLS estimator minimizes $\lVert\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\rVert^2$:

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

This is a linear function of $\mathbf{y}$: the matrix $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ maps $\mathbf{y}$ to $\hat{\boldsymbol{\beta}}$.

## Sampling Distribution of the OLS Estimator

!!! tip "Theorem -- Distribution of the OLS Estimator"
    Under the general linear model with normal errors:

    $$
    \hat{\boldsymbol{\beta}} \sim N\!\left(\boldsymbol{\beta},\; \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}\right)
    $$

**Proof.**

*Linearity:* $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ is a linear transformation of $\mathbf{y}$.

*Mean:*

$$
E[\hat{\boldsymbol{\beta}}] = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T E[\mathbf{y}] = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \boldsymbol{\beta}
$$

So $\hat{\boldsymbol{\beta}}$ is **unbiased**.

*Covariance matrix:*

$$
\operatorname{Var}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \operatorname{Var}(\mathbf{y})\, \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}
$$

$$
= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T (\sigma^2\mathbf{I}_n)\, \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1} = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}
$$

*Normality:* A linear transformation of a multivariate normal vector is multivariate normal. $\square$

The variance of the $j$-th coefficient is $\operatorname{Var}(\hat{\beta}_j) = \sigma^2[(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}$.

## Fitted Values and Residuals

The fitted values and residuals are

$$
\hat{\mathbf{y}} = \mathbf{H}\mathbf{y}, \qquad \mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y} = \mathbf{M}\mathbf{y}
$$

where $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is the hat matrix and $\mathbf{M} = \mathbf{I} - \mathbf{H}$ is the residual-maker matrix.

Their distributions:

$$
\hat{\mathbf{y}} \sim N(\mathbf{X}\boldsymbol{\beta},\; \sigma^2\mathbf{H})
$$

$$
\mathbf{e} \sim N(\mathbf{0},\; \sigma^2\mathbf{M})
$$

Note that $\mathbf{e}$ has mean zero regardless of $\boldsymbol{\beta}$ (because $\mathbf{M}\mathbf{X} = \mathbf{0}$), but its covariance matrix $\sigma^2\mathbf{M}$ is singular (rank $n - p$), so the residuals are not independent of each other.

## Distribution of the Residual Sum of Squares

!!! tip "Theorem -- Chi-Squared Distribution of SSE"
    The residual sum of squares

    $$
    \text{SSE} = \mathbf{e}^T\mathbf{e} = \mathbf{y}^T\mathbf{M}\mathbf{y}
    $$

    satisfies

    $$
    \frac{\text{SSE}}{\sigma^2} \sim \chi^2_{n-p}
    $$

**Proof.** Write $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$. Since $\mathbf{M}\mathbf{X} = \mathbf{0}$:

$$
\text{SSE} = \boldsymbol{\varepsilon}^T\mathbf{M}\boldsymbol{\varepsilon}
$$

Let $\mathbf{z} = \boldsymbol{\varepsilon}/\sigma \sim N(\mathbf{0}, \mathbf{I}_n)$. Then $\text{SSE}/\sigma^2 = \mathbf{z}^T\mathbf{M}\mathbf{z}$. Since $\mathbf{M}$ is symmetric idempotent with $\operatorname{rank}(\mathbf{M}) = n - p$, the fundamental chi-squared theorem gives $\mathbf{z}^T\mathbf{M}\mathbf{z} \sim \chi^2_{n-p}$. $\square$

The unbiased estimator of $\sigma^2$ is therefore

$$
s^2 = \frac{\text{SSE}}{n - p}
$$

## Independence of Estimator and SSE

!!! tip "Theorem -- Independence"
    $\hat{\boldsymbol{\beta}}$ and $\text{SSE}$ are independent.

**Proof.** Since $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ is a linear function of $\mathbf{y}$ and $\text{SSE} = \mathbf{y}^T\mathbf{M}\mathbf{y}$ is a quadratic form, independence follows from the fact that the linear part and the quadratic part involve orthogonal projections. Formally:

$$
\operatorname{Cov}(\hat{\boldsymbol{\beta}}, \mathbf{e}) = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\operatorname{Var}(\mathbf{y})\mathbf{M} = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{M} = \mathbf{0}
$$

because $\mathbf{X}^T\mathbf{M} = \mathbf{X}^T(\mathbf{I} - \mathbf{H}) = \mathbf{X}^T - \mathbf{X}^T = \mathbf{0}$. Under joint normality, zero covariance implies independence. $\square$

## The t-Statistic for Individual Coefficients

Combining the normal distribution of $\hat{\beta}_j$, the chi-squared distribution of $\text{SSE}/\sigma^2$, and their independence yields a t-distribution.

!!! tip "Theorem -- t-Distribution for Coefficient Tests"
    For each $j = 1, \dots, p$:

    $$
    T_j = \frac{\hat{\beta}_j - \beta_j}{\text{SE}(\hat{\beta}_j)} \sim t_{n-p}
    $$

    where $\text{SE}(\hat{\beta}_j) = s\sqrt{[(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}}$.

**Proof sketch.** The standardized estimator $(\hat{\beta}_j - \beta_j)/(\sigma\sqrt{[(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}}) \sim N(0,1)$. Replacing $\sigma$ with $s = \sqrt{\text{SSE}/(n-p)}$ introduces a ratio with $\sqrt{\chi^2_{n-p}/(n-p)}$ in the denominator. Independence of the numerator and denominator (from the independence theorem above) gives a $t_{n-p}$ distribution. $\square$

## The F-Statistic for Testing Multiple Coefficients

To test whether a subset of coefficients are simultaneously zero, consider testing $H_0: \mathbf{C}\boldsymbol{\beta} = \mathbf{0}$ where $\mathbf{C}$ is a $q \times p$ matrix of rank $q$.

!!! tip "Theorem -- F-Distribution for Linear Hypotheses"
    Under $H_0: \mathbf{C}\boldsymbol{\beta} = \mathbf{0}$:

    $$
    F = \frac{(\mathbf{C}\hat{\boldsymbol{\beta}})^T[\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T]^{-1}(\mathbf{C}\hat{\boldsymbol{\beta}})/q}{s^2} \sim F_{q,\,n-p}
    $$

**Proof sketch.** Under $H_0$, $\mathbf{C}\hat{\boldsymbol{\beta}} \sim N(\mathbf{0}, \sigma^2\mathbf{C}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{C}^T)$. The quadratic form in the numerator, divided by $\sigma^2$, is $\chi^2_q$ (by the chi-squared theorem for quadratic forms in normal vectors). This is independent of $s^2$ (which is based on $\text{SSE}/(n-p)$). The ratio of two independent chi-squared variables divided by their respective degrees of freedom is $F_{q,\,n-p}$. $\square$

### Special Case: Overall F-Test

Testing $H_0: \beta_1 = \beta_2 = \cdots = \beta_{p-1} = 0$ (all slopes zero, assuming the first column of $\mathbf{X}$ is the intercept) gives the overall F-statistic:

$$
F = \frac{\text{SSR}/(p - 1)}{\text{SSE}/(n - p)} = \frac{\text{MSR}}{\text{MSE}}
$$

where $\text{SSR} = \hat{\mathbf{y}}^T\hat{\mathbf{y}} - n\bar{Y}^2$ is the regression sum of squares. Under $H_0$, $F \sim F_{p-1,\,n-p}$.

## Gauss-Markov Theorem

Even without assuming normality, the OLS estimator has an optimality property.

!!! tip "Theorem -- Gauss-Markov"
    Under the assumptions $E[\boldsymbol{\varepsilon}] = \mathbf{0}$ and $\operatorname{Var}(\boldsymbol{\varepsilon}) = \sigma^2\mathbf{I}_n$ (no normality required), the OLS estimator $\hat{\boldsymbol{\beta}}$ is the **Best Linear Unbiased Estimator (BLUE)**: among all linear unbiased estimators of $\boldsymbol{\beta}$, OLS has the smallest variance (in the matrix sense).

**Proof sketch.** Let $\tilde{\boldsymbol{\beta}} = \mathbf{A}\mathbf{y}$ be any linear unbiased estimator. Unbiasedness requires $\mathbf{A}\mathbf{X} = \mathbf{I}_p$. Write $\mathbf{A} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T + \mathbf{D}$ where $\mathbf{D}\mathbf{X} = \mathbf{0}$. Then

$$
\operatorname{Var}(\tilde{\boldsymbol{\beta}}) = \sigma^2\mathbf{A}\mathbf{A}^T = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1} + \sigma^2\mathbf{D}\mathbf{D}^T
$$

Since $\mathbf{D}\mathbf{D}^T \succeq 0$, we have $\operatorname{Var}(\tilde{\boldsymbol{\beta}}) \succeq \operatorname{Var}(\hat{\boldsymbol{\beta}})$ in the positive semi-definite ordering. $\square$

## Summary of Key Distributions

| Quantity | Distribution | Degrees of freedom |
|---|---|---|
| $\hat{\boldsymbol{\beta}}$ | $N(\boldsymbol{\beta}, \sigma^2(\mathbf{X}^T\mathbf{X})^{-1})$ | -- |
| $\text{SSE}/\sigma^2$ | $\chi^2_{n-p}$ | $n - p$ |
| $T_j = (\hat{\beta}_j - \beta_j)/\text{SE}(\hat{\beta}_j)$ | $t_{n-p}$ | $n - p$ |
| $F = \text{MSR}/\text{MSE}$ (under $H_0$) | $F_{q,\,n-p}$ | $q$ and $n-p$ |

All of these distributions rely on three ingredients: (1) $\hat{\boldsymbol{\beta}}$ is a linear function of the normal vector $\mathbf{y}$, (2) $\mathbf{M}$ is symmetric idempotent, and (3) $\hat{\boldsymbol{\beta}}$ and $\text{SSE}$ are independent because $\mathbf{X}^T\mathbf{M} = \mathbf{0}$.

## Summary

The matrix formulation of OLS compresses the simple-regression sampling theory into a unified framework. The estimator $\hat{\boldsymbol{\beta}}$ is multivariate normal with covariance $\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$, the residual sum of squares $\text{SSE}/\sigma^2$ is chi-squared with $n - p$ degrees of freedom, and the two are independent. These three facts, which follow from the orthogonal projection structure of OLS and the normality of the errors, generate all the standard t-tests for individual coefficients and F-tests for groups of coefficients. The Gauss-Markov theorem further shows that OLS is optimal among linear unbiased estimators even without the normality assumption.
