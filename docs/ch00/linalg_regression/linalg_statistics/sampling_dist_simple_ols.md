# Sampling Distributions (Simple Ordinary Least Squares)

Before developing the general multiple-regression theory, it is instructive to derive the sampling distributions of the estimators in **simple linear regression** -- the model with a single predictor. Working through the simple case builds intuition for how normality of errors propagates to normality of estimators, why the degrees of freedom are $n - 2$, and how the t-statistic arises. The simple-case formulas also make the role of each algebraic quantity (sum of squares, cross-products) transparent in a way that matrix notation can obscure.

## The Simple Linear Regression Model

The model is

$$
Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \quad i = 1, \dots, n
$$

where:

- $x_1, \dots, x_n$ are fixed (non-random) predictor values, not all equal.
- $\varepsilon_1, \dots, \varepsilon_n$ are independent $N(0, \sigma^2)$ random variables.
- $\beta_0$ (intercept) and $\beta_1$ (slope) are unknown parameters.
- $\sigma^2$ is the unknown error variance.

Since $\varepsilon_i \sim N(0, \sigma^2)$ and the $x_i$ are fixed, each response is $Y_i \sim N(\beta_0 + \beta_1 x_i, \sigma^2)$, independently.

## OLS Estimators

The OLS estimators minimize $\sum_{i=1}^n (Y_i - \beta_0 - \beta_1 x_i)^2$. The closed-form solutions are:

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(Y_i - \bar{Y})}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}
$$

$$
\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{x}
$$

where $\bar{x} = \frac{1}{n}\sum_i x_i$, $\bar{Y} = \frac{1}{n}\sum_i Y_i$, $S_{xx} = \sum_i(x_i - \bar{x})^2$, and $S_{xy} = \sum_i(x_i - \bar{x})(Y_i - \bar{Y})$.

## Linearity of the Estimators

A key observation is that both estimators are **linear functions of the responses** $Y_1, \dots, Y_n$.

For the slope, since $S_{xy} = \sum_i(x_i - \bar{x})Y_i$ (because $\sum_i(x_i - \bar{x})\bar{Y} = 0$):

$$
\hat{\beta}_1 = \sum_{i=1}^n c_i Y_i, \quad \text{where } c_i = \frac{x_i - \bar{x}}{S_{xx}}
$$

The weights $c_i$ satisfy two useful identities:

$$
\sum_{i=1}^n c_i = 0, \qquad \sum_{i=1}^n c_i^2 = \frac{1}{S_{xx}}
$$

Since $\hat{\beta}_1$ is a linear combination of independent normal random variables, it is itself normal.

## Sampling Distribution of the Slope

!!! tip "Theorem -- Distribution of the Slope Estimator"
    Under the simple linear regression model with normal errors:

    $$
    \hat{\beta}_1 \sim N\!\left(\beta_1,\; \frac{\sigma^2}{S_{xx}}\right)
    $$

**Proof.**

*Mean:*

$$
E[\hat{\beta}_1] = \sum_i c_i E[Y_i] = \sum_i c_i(\beta_0 + \beta_1 x_i) = \beta_0\sum_i c_i + \beta_1\sum_i c_i x_i
$$

Since $\sum_i c_i = 0$ and $\sum_i c_i x_i = \sum_i \frac{(x_i - \bar{x})x_i}{S_{xx}} = \frac{S_{xx}}{S_{xx}} = 1$:

$$
E[\hat{\beta}_1] = \beta_1
$$

So $\hat{\beta}_1$ is **unbiased**.

*Variance:*

$$
\operatorname{Var}(\hat{\beta}_1) = \sum_i c_i^2 \operatorname{Var}(Y_i) = \sigma^2 \sum_i c_i^2 = \frac{\sigma^2}{S_{xx}}
$$

*Normality:* Since $\hat{\beta}_1$ is a linear combination of independent normal random variables, it is normal. $\square$

## Sampling Distribution of the Intercept

!!! tip "Theorem -- Distribution of the Intercept Estimator"
    Under the same model:

    $$
    \hat{\beta}_0 \sim N\!\left(\beta_0,\; \sigma^2\left(\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}\right)\right)
    $$

**Proof.**

*Mean:* $E[\hat{\beta}_0] = E[\bar{Y}] - E[\hat{\beta}_1]\bar{x} = (\beta_0 + \beta_1\bar{x}) - \beta_1\bar{x} = \beta_0$. Unbiased.

*Variance:* Since $\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1\bar{x}$ and $\operatorname{Cov}(\bar{Y}, \hat{\beta}_1) = 0$ (because $\sum_i c_i = 0$):

$$
\operatorname{Var}(\hat{\beta}_0) = \operatorname{Var}(\bar{Y}) + \bar{x}^2\operatorname{Var}(\hat{\beta}_1) = \frac{\sigma^2}{n} + \frac{\bar{x}^2\sigma^2}{S_{xx}} = \sigma^2\!\left(\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}\right)
$$

$\square$

## Residual Sum of Squares and Estimation of Variance

The residual sum of squares is

$$
\text{SSE} = \sum_{i=1}^n (Y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2
$$

!!! tip "Theorem -- Distribution of SSE"
    Under the normal simple linear regression model:

    $$
    \frac{\text{SSE}}{\sigma^2} \sim \chi^2_{n-2}
    $$

    and SSE is independent of $(\hat{\beta}_0, \hat{\beta}_1)$.

The degrees of freedom are $n - 2$ because two parameters ($\beta_0$ and $\beta_1$) are estimated. The unbiased estimator of $\sigma^2$ is

$$
s^2 = \frac{\text{SSE}}{n - 2}
$$

with $E[s^2] = \sigma^2$.

## The t-Statistics

Since $\sigma^2$ is unknown in practice, we replace it with $s^2$ in the standard errors. This converts normal distributions into t-distributions.

!!! tip "Theorem -- t-Distribution for Slope"
    The statistic

    $$
    T = \frac{\hat{\beta}_1 - \beta_1}{s / \sqrt{S_{xx}}} \sim t_{n-2}
    $$

    has a Student's t-distribution with $n - 2$ degrees of freedom.

**Proof sketch.** The numerator $(\hat{\beta}_1 - \beta_1)/(\sigma/\sqrt{S_{xx}}) \sim N(0,1)$ and the denominator involves $s/\sigma = \sqrt{\text{SSE}/((n-2)\sigma^2)}$. Since $\text{SSE}/\sigma^2 \sim \chi^2_{n-2}$ and is independent of $\hat{\beta}_1$, the ratio has the form $N(0,1)/\sqrt{\chi^2_{n-2}/(n-2)}$, which defines the $t_{n-2}$ distribution. $\square$

Similarly, for the intercept:

$$
\frac{\hat{\beta}_0 - \beta_0}{s\sqrt{1/n + \bar{x}^2/S_{xx}}} \sim t_{n-2}
$$

## Confidence Intervals

The t-distribution results immediately yield confidence intervals.

A $100(1 - \alpha)\%$ confidence interval for $\beta_1$ is

$$
\hat{\beta}_1 \pm t_{\alpha/2,\,n-2} \cdot \frac{s}{\sqrt{S_{xx}}}
$$

and for $\beta_0$:

$$
\hat{\beta}_0 \pm t_{\alpha/2,\,n-2} \cdot s\sqrt{\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}}
$$

where $t_{\alpha/2,\,n-2}$ is the upper $\alpha/2$ quantile of the $t_{n-2}$ distribution.

## Example

Suppose $n = 5$ data points with $\bar{x} = 3$, $S_{xx} = 10$, $\hat{\beta}_1 = 2.5$, $\hat{\beta}_0 = 1.0$, and $\text{SSE} = 6.0$.

- **Estimated variance:** $s^2 = 6.0 / 3 = 2.0$, so $s = \sqrt{2} \approx 1.414$.
- **Standard error of slope:** $\text{SE}(\hat{\beta}_1) = s/\sqrt{S_{xx}} = \sqrt{2}/\sqrt{10} = \sqrt{0.2} \approx 0.447$.
- **t-statistic for $H_0: \beta_1 = 0$:** $T = 2.5 / 0.447 \approx 5.59$, compared to $t_{3}$.
- **95% CI for slope:** $2.5 \pm 3.182 \times 0.447 \approx 2.5 \pm 1.42 = (1.08, 3.92)$ (using $t_{0.025, 3} = 3.182$).

## Summary

In simple linear regression with normal errors, the slope estimator $\hat{\beta}_1$ and the intercept estimator $\hat{\beta}_0$ are normally distributed with means equal to the true parameters (unbiased) and variances that depend on $\sigma^2$ and the spread of the predictor values $S_{xx}$. The residual sum of squares $\text{SSE}/\sigma^2$ follows a $\chi^2_{n-2}$ distribution and is independent of the estimators. Replacing $\sigma$ with $s = \sqrt{\text{SSE}/(n-2)}$ in the standardized estimators produces t-statistics with $n - 2$ degrees of freedom, which form the basis for hypothesis tests and confidence intervals.

## Exercises

**Exercise 1.**
In a simple linear regression with $n = 10$ data points, $\bar{x} = 4$, $S_{xx} = 20$, $\hat{\beta}_1 = 3.0$, and $\text{SSE} = 16$, construct a 95% confidence interval for the slope $\beta_1$.

??? success "Solution to Exercise 1"
    First compute the estimated variance:

    $$
    s^2 = \frac{\text{SSE}}{n - 2} = \frac{16}{8} = 2.0
    $$

    The standard error of the slope is:

    $$
    \text{SE}(\hat{\beta}_1) = \frac{s}{\sqrt{S_{xx}}} = \frac{\sqrt{2}}{\sqrt{20}} = \sqrt{0.1} \approx 0.3162
    $$

    Using $t_{0.025, 8} = 2.306$, the 95% confidence interval is:

    $$
    3.0 \pm 2.306 \times 0.3162 = 3.0 \pm 0.729 = (2.271,\; 3.729)
    $$

---

**Exercise 2.**
Prove that $\hat{\beta}_1 = \sum_{i=1}^n c_i Y_i$ where $c_i = (x_i - \bar{x})/S_{xx}$, and use this to derive $\operatorname{Var}(\hat{\beta}_1) = \sigma^2 / S_{xx}$.

??? success "Solution to Exercise 2"
    The OLS slope estimator is:

    $$
    \hat{\beta}_1 = \frac{S_{xy}}{S_{xx}} = \frac{\sum_{i=1}^n (x_i - \bar{x})(Y_i - \bar{Y})}{S_{xx}} = \frac{\sum_{i=1}^n (x_i - \bar{x})Y_i}{S_{xx}}
    $$

    The last equality uses $\sum(x_i - \bar{x})\bar{Y} = \bar{Y}\sum(x_i - \bar{x}) = 0$. Setting $c_i = (x_i - \bar{x})/S_{xx}$, we have $\hat{\beta}_1 = \sum c_i Y_i$.

    Since the $Y_i$ are independent with $\operatorname{Var}(Y_i) = \sigma^2$:

    $$
    \operatorname{Var}(\hat{\beta}_1) = \sum_{i=1}^n c_i^2 \operatorname{Var}(Y_i) = \sigma^2 \sum_{i=1}^n \frac{(x_i - \bar{x})^2}{S_{xx}^2} = \sigma^2 \cdot \frac{S_{xx}}{S_{xx}^2} = \frac{\sigma^2}{S_{xx}}
    $$

    $\square$

---

**Exercise 3.**
Explain why $\text{SSE}/\sigma^2 \sim \chi^2_{n-2}$ has $n - 2$ degrees of freedom rather than $n$ degrees of freedom.

??? success "Solution to Exercise 3"
    The residual sum of squares can be written as $\text{SSE} = \mathbf{Y}^T(\mathbf{I} - \mathbf{H})\mathbf{Y}/\sigma^2$ (after dividing by $\sigma^2$), where $\mathbf{H}$ is the hat matrix. The matrix $\mathbf{I} - \mathbf{H}$ is idempotent with rank $n - 2$ (since $\text{rank}(\mathbf{H}) = 2$ in simple regression, corresponding to the intercept and slope).

    The rank of the idempotent matrix equals the degrees of freedom of the resulting chi-squared distribution. Intuitively, we start with $n$ independent pieces of information in $\mathbf{Y}$, but fitting the two parameters $\beta_0$ and $\beta_1$ "uses up" 2 degrees of freedom, leaving $n - 2$ for estimating $\sigma^2$.

---

**Exercise 4.**
Show that $\hat{\beta}_0$ and $\hat{\beta}_1$ are correlated, and derive $\operatorname{Cov}(\hat{\beta}_0, \hat{\beta}_1) = -\bar{x}\,\sigma^2/S_{xx}$.

??? success "Solution to Exercise 4"
    Since $\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{x}$, we compute:

    $$
    \operatorname{Cov}(\hat{\beta}_0, \hat{\beta}_1) = \operatorname{Cov}(\bar{Y} - \hat{\beta}_1 \bar{x},\; \hat{\beta}_1) = \operatorname{Cov}(\bar{Y}, \hat{\beta}_1) - \bar{x}\operatorname{Var}(\hat{\beta}_1)
    $$

    Now $\bar{Y} = \frac{1}{n}\sum Y_i$ and $\hat{\beta}_1 = \sum c_i Y_i$ with $c_i = (x_i - \bar{x})/S_{xx}$, so:

    $$
    \operatorname{Cov}(\bar{Y}, \hat{\beta}_1) = \frac{1}{n}\sum_{i=1}^n c_i \operatorname{Var}(Y_i) = \frac{\sigma^2}{n} \sum_{i=1}^n \frac{x_i - \bar{x}}{S_{xx}} = 0
    $$

    since $\sum(x_i - \bar{x}) = 0$. Therefore:

    $$
    \operatorname{Cov}(\hat{\beta}_0, \hat{\beta}_1) = 0 - \bar{x} \cdot \frac{\sigma^2}{S_{xx}} = -\frac{\bar{x}\,\sigma^2}{S_{xx}}
    $$

    The estimators are negatively correlated when $\bar{x} > 0$. $\square$
