# Sampling Distributions for Simple OLS Estimators

## Distribution of SSE

The least squares estimators of the regression coefficients $\beta_0$ and $\beta_1$ are $\hat{\beta}_0$ and $\hat{\beta}_1$, which minimize the sum of squared errors (SSE). The fitted values are given by:

$$
\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i,
$$

and the residuals are:

$$
e_i = y_i - \hat{y}_i
$$

The sum of squared residuals, also known as the sum of squared errors (SSE), is:

$$
SSE = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n e_i^2
$$

!!! info "Theorem"
    $$SSE \sim \sigma^2 \chi^2_{n-2}$$

??? note "Proof"
    **Step 1: Linear Regression and the Error Terms.** Consider the linear regression model:

    $$
    y_i = \beta_0 + \beta_1 x_i + \varepsilon_i
    $$

    where $\varepsilon_i \sim N(0, \sigma^2)$ are independent and identically distributed. In matrix form:

    $$
    \mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}
    $$

    where $\mathbf{y}$ is an $n \times 1$ vector, $\mathbf{X}$ is an $n \times 2$ design matrix (containing 1's for the intercept and $x_i$ values for the slope), $\boldsymbol{\beta}$ is a $2 \times 1$ parameter vector, and $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2 \mathbf{I})$.

    **Step 2: Residuals and SSE.** The fitted values are $\hat{\mathbf{y}} = \mathbf{X} \hat{\boldsymbol{\beta}}$, the residuals are $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}$, and:

    $$
    SSE = \sum_{i=1}^n e_i^2 = \mathbf{e}^T \mathbf{e}
    $$

    **Step 3: Expressing SSE in Terms of Errors.** The residuals can be expressed as $\mathbf{e} = (\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}$, where $\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1}\mathbf{X}^T$ is the hat matrix. The matrix $\mathbf{I} - \mathbf{H}$ is idempotent, symmetric, and has rank $n - 2$.

    **Step 4: Distribution of SSE.**

    $$
    SSE = \mathbf{e}^T \mathbf{e} = \boldsymbol{\varepsilon}^T (\mathbf{I} - \mathbf{H}) \boldsymbol{\varepsilon}
    $$

    Since $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2 \mathbf{I})$, the quadratic form $\boldsymbol{\varepsilon}^T (\mathbf{I} - \mathbf{H}) \boldsymbol{\varepsilon}$ follows a scaled chi-squared distribution. The matrix $(\mathbf{I} - \mathbf{H})$ is idempotent and symmetric with rank $n - 2$, so the quadratic form follows a chi-squared distribution with $n - 2$ degrees of freedom, scaled by $\sigma^2$:

    $$
    SSE = \boldsymbol{\varepsilon}^T (\mathbf{I} - \mathbf{H}) \boldsymbol{\varepsilon} \sim \sigma^2 \chi^2_{n-2}
    $$

This result is essential in deriving properties of estimators in regression analysis, such as estimating $\sigma^2$ by dividing $SSE$ by its degrees of freedom $(n-2)$ to obtain an unbiased estimator for $\sigma^2$.

## Expectation of SSE

!!! info "Theorem"
    Since $SSE\sim\sigma^2\chi^2_{n-2}$,

    $$E(SSE)=(n-2)\sigma^2$$

## Estimator of $\sigma^2$

!!! info "Theorem"
    $s^2$ is an unbiased estimator of $\sigma^2$, where

    $$s^2 = \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{n - 2} = \frac{SSE}{n-2}$$

??? note "Proof"
    $$
    Es^2 = \frac{E(SSE)}{n-2}= \frac{(n-2)\sigma^2}{n-2}=\sigma^2
    $$
