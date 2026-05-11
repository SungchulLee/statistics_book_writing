# Sampling Distributions for Simple OLS Estimators


## Overview

In simple linear regression, the key inferential results depend on knowing the **sampling distributions** of the estimated coefficients and predictions. Under the classical assumptions—linearity, independence, homoscedasticity, and normality of errors—these distributions take elegant closed forms based on the $t$-distribution. This section derives the sampling distributions for three fundamental quantities: the slope estimator, the expected response at a given point, and an individual predicted response.

---

## 1. Slope Estimator

### Statement

$$
\frac{\hat{\beta}_1 - \beta_1}{s\sqrt{\dfrac{1}{\sum_{i=1}^n(x_i - \bar{x})^2}}} \sim t_{n-2}
$$

### Components

**Estimated slope** $\hat{\beta}_1$: The OLS estimate of the slope, representing the observed change in $y$ per unit increase in $x$ in the sample.

**True slope** $\beta_1$: The unknown population parameter we are estimating. Hypothesis tests typically examine whether $\beta_1 = 0$.

**Residual standard deviation** $s$: Measures the variability in $y$ not explained by the linear relationship with $x$:

$$
s = \sqrt{\frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{n - 2}}
$$

where $\hat{y}_i$ is the fitted value and $n - 2$ accounts for estimating both $\beta_0$ and $\beta_1$.

**Sum of squares of $x$**: The quantity $SS_x = \sum_{i=1}^n (x_i - \bar{x})^2$ captures the spread of the predictor values and directly determines the precision of the slope estimate.

**Standard error of the slope**: The denominator $s \sqrt{1 / SS_x}$ quantifies the uncertainty in $\hat{\beta}_1$ due to sampling variability, combining residual spread with predictor variability.

### Proof

**Step 1: Model Assumptions.** Consider the simple linear regression model:

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i
$$

where $\varepsilon_i \overset{\text{i.i.d.}}{\sim} N(0, \sigma^2)$.

**Step 2: OLS Estimator.** The least squares estimator for $\beta_1$ is:

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}
$$

**Step 3: Distribution of $\hat{\beta}_1$.** Since $\hat{\beta}_1$ is a linear combination of the normally distributed $y_i$ values:

$$
\hat{\beta}_1 \sim N\!\left(\beta_1,\; \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}\right)
$$

**Step 4: Standardization with known $\sigma$.** Dividing by the true standard deviation yields a standard normal:

$$
\frac{\hat{\beta}_1 - \beta_1}{\sigma\sqrt{\dfrac{1}{\sum_{i=1}^n(x_i - \bar{x})^2}}} \sim N(0, 1)
$$

**Step 5: Estimating $\sigma^2$.** Since $\sigma^2$ is unknown, we use the unbiased estimator:

$$
s^2 = \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{n - 2}
$$

**Step 6: Substitution.** Replacing $\sigma$ with $s$:

$$
\frac{\hat{\beta}_1 - \beta_1}{s\sqrt{\dfrac{1}{\sum_{i=1}^n(x_i - \bar{x})^2}}}
$$

**Step 7: $t$-distribution result.** The numerator is normal, $s^2$ follows a scaled chi-squared distribution with $n - 2$ degrees of freedom, and the two are independent. By the definition of the $t$-distribution as the ratio of a standard normal to the square root of an independent chi-squared divided by its degrees of freedom:

$$
\frac{\hat{\beta}_1 - \beta_1}{s\sqrt{\dfrac{1}{\sum_{i=1}^n(x_i - \bar{x})^2}}} \sim t_{n-2} \qquad \square
$$

---

## 2. Expectation of Response at a Given Point x_0
### Statement

$$
\frac{(\hat{\beta}_0 + \hat{\beta}_1 x_0) - (\beta_0 + \beta_1 x_0)}{s\sqrt{\dfrac{1}{n} + \dfrac{(x_0 - \bar{x})^2}{\sum_{i=1}^n(x_i - \bar{x})^2}}} \sim t_{n-2}
$$

### Components

**Predicted mean response** $\hat{\beta}_0 + \hat{\beta}_1 x_0$: The estimated expected value of $y$ at $x = x_0$ from the fitted regression line.

**True mean response** $\beta_0 + \beta_1 x_0$: The unknown population mean of $y$ at $x = x_0$.

**Standard error of the mean prediction**: The denominator consists of two variance components:

- $\dfrac{1}{n}$: Variability from estimating the intercept and slope.
- $\dfrac{(x_0 - \bar{x})^2}{\sum_{i=1}^n(x_i - \bar{x})^2}$: Additional variability when $x_0$ is far from $\bar{x}$.

This means the confidence band for the mean response is narrowest at $\bar{x}$ and widens as $x_0$ moves away from the center of the data.

### Proof

**Step 1: Prediction error decomposition.** The difference between estimated and true mean response is:

$$
(\hat{\beta}_0 + \hat{\beta}_1 x_0) - (\beta_0 + \beta_1 x_0) = (\hat{\beta}_0 - \beta_0) + (\hat{\beta}_1 - \beta_1)x_0
$$

This has mean zero since both estimators are unbiased.

**Step 2: Variance computation.** Using the OLS variance–covariance results:

- $\text{Var}(\hat{\beta}_0) = \sigma^2\!\left(\dfrac{1}{n} + \dfrac{\bar{x}^2}{SS_x}\right)$
- $\text{Var}(\hat{\beta}_1) = \dfrac{\sigma^2}{SS_x}$
- $\text{Cov}(\hat{\beta}_0, \hat{\beta}_1) = -\dfrac{\sigma^2 \bar{x}}{SS_x}$

Combining via $\text{Var}(\hat{\beta}_0 + \hat{\beta}_1 x_0) = \text{Var}(\hat{\beta}_0) + x_0^2\,\text{Var}(\hat{\beta}_1) + 2x_0\,\text{Cov}(\hat{\beta}_0, \hat{\beta}_1)$:

$$
\text{Var}(\hat{\beta}_0 + \hat{\beta}_1 x_0) = \sigma^2\!\left(\frac{1}{n} + \frac{(x_0 - \bar{x})^2}{SS_x}\right)
$$

**Step 3: Standardization.** With known $\sigma$:

$$
\frac{(\hat{\beta}_0 + \hat{\beta}_1 x_0) - (\beta_0 + \beta_1 x_0)}{\sigma\sqrt{\dfrac{1}{n} + \dfrac{(x_0 - \bar{x})^2}{SS_x}}} \sim N(0, 1)
$$

**Step 4: Substituting $s$ for $\sigma$.** Replacing $\sigma$ with the residual standard error $s$ and applying the same $t$-distribution argument as before:

$$
\frac{(\hat{\beta}_0 + \hat{\beta}_1 x_0) - (\beta_0 + \beta_1 x_0)}{s\sqrt{\dfrac{1}{n} + \dfrac{(x_0 - \bar{x})^2}{SS_x}}} \sim t_{n-2} \qquad \square
$$

---

## 3. Response (Prediction) at a Given Point x_0
### Statement

$$
\frac{(\hat{\beta}_0 + \hat{\beta}_1 x_0) - (\beta_0 + \beta_1 x_0 + \varepsilon)}{s\sqrt{1 + \dfrac{1}{n} + \dfrac{(x_0 - \bar{x})^2}{\sum_{i=1}^n(x_i - \bar{x})^2}}} \sim t_{n-2}
$$

### Components

**Predicted response** $\hat{y}_0 = \hat{\beta}_0 + \hat{\beta}_1 x_0$: The point prediction at $x_0$.

**True individual response** $y_0 = \beta_0 + \beta_1 x_0 + \varepsilon$: The actual observation, which differs from the mean response by the random error $\varepsilon \sim N(0, \sigma^2)$.

**Standard error of individual prediction**: The denominator includes three variance components:

- $1$: Residual variability of individual observations around the regression line.
- $\dfrac{1}{n}$: Sampling variability in estimating the regression coefficients.
- $\dfrac{(x_0 - \bar{x})^2}{SS_x}$: Additional uncertainty from extrapolation away from $\bar{x}$.

The leading $1$ term is the critical difference from the mean response case. It ensures that prediction intervals for individual observations are always wider than confidence intervals for the mean response.

### Proof

**Step 1: Prediction error.** Define the prediction error:

$$
\hat{y}_0 - y_0 = (\hat{\beta}_0 + \hat{\beta}_1 x_0) - (\beta_0 + \beta_1 x_0 + \varepsilon) = (\hat{\beta}_0 - \beta_0) + (\hat{\beta}_1 - \beta_1)x_0 - \varepsilon
$$

**Step 2: Variance decomposition.** The error $\varepsilon$ is independent of the estimators $\hat{\beta}_0$ and $\hat{\beta}_1$ (which depend on the training data), so:

$$
\text{Var}(\hat{y}_0 - y_0) = \underbrace{\sigma^2\!\left(\frac{1}{n} + \frac{(x_0 - \bar{x})^2}{SS_x}\right)}_{\text{estimation uncertainty}} + \underbrace{\sigma^2}_{\text{irreducible noise}} = \sigma^2\!\left(1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{SS_x}\right)
$$

**Step 3: Standardization with known $\sigma$.** The prediction error is normally distributed with mean zero:

$$
\frac{\hat{y}_0 - y_0}{\sigma\sqrt{1 + \dfrac{1}{n} + \dfrac{(x_0 - \bar{x})^2}{SS_x}}} \sim N(0, 1)
$$

**Step 4: Substituting $s$ for $\sigma$.** Replacing $\sigma$ with $s$ yields the $t$-distribution:

$$
\frac{(\hat{\beta}_0 + \hat{\beta}_1 x_0) - (\beta_0 + \beta_1 x_0 + \varepsilon)}{s\sqrt{1 + \dfrac{1}{n} + \dfrac{(x_0 - \bar{x})^2}{SS_x}}} \sim t_{n-2} \qquad \square
$$

---

## Summary Comparison

| Quantity | Standard Error | Distribution |
|:---|:---|:---|
| Slope $\hat{\beta}_1$ | $s\sqrt{\dfrac{1}{SS_x}}$ | $t_{n-2}$ |
| Mean response at $x_0$ | $s\sqrt{\dfrac{1}{n} + \dfrac{(x_0-\bar{x})^2}{SS_x}}$ | $t_{n-2}$ |
| Individual response at $x_0$ | $s\sqrt{1 + \dfrac{1}{n} + \dfrac{(x_0-\bar{x})^2}{SS_x}}$ | $t_{n-2}$ |

All three statistics share the same $t_{n-2}$ distribution but differ in their standard errors, reflecting the increasing sources of uncertainty from slope estimation alone, to mean prediction, to individual prediction.
## Exercises

**Exercise 1.**
State the sampling distribution of $\hat{\beta}_1$ in simple linear regression under the classical assumptions. What parameters does it depend on?

??? success "Solution to Exercise 1"
    Under the classical assumptions ($\varepsilon_i \sim N(0, \sigma^2)$, independent):

    $$
    \hat{\beta}_1 \sim N\left(\beta_1, \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}\right)
    $$

    It depends on: the true slope $\beta_1$, the error variance $\sigma^2$, and the spread of the predictor values $\sum(x_i - \bar{x})^2$. Greater spread in $X$ and smaller error variance both lead to a more precise estimate.

---

**Exercise 2.**
Explain why $\hat{\sigma}^2 = \text{SSE}/(n-2)$ is an unbiased estimator of $\sigma^2$ in simple linear regression, while $\text{SSE}/n$ is biased.

??? success "Solution to Exercise 2"
    The residuals $e_i = Y_i - \hat{Y}_i$ are constrained by two linear restrictions (the normal equations), so only $n - 2$ of them are free to vary. This means $\text{SSE}/\sigma^2 \sim \chi^2_{n-2}$, and:

    $$
    E\left[\frac{\text{SSE}}{\sigma^2}\right] = n - 2 \implies E[\text{SSE}] = (n-2)\sigma^2 \implies E\left[\frac{\text{SSE}}{n-2}\right] = \sigma^2
    $$

    Dividing by $n$ instead gives $E[\text{SSE}/n] = (n-2)\sigma^2/n < \sigma^2$, which underestimates $\sigma^2$.

---

**Exercise 3.**
A researcher wants to estimate the slope with high precision. Based on the formula for $\text{Var}(\hat{\beta}_1)$, give two practical strategies to reduce the standard error.

??? success "Solution to Exercise 3"
    From $\text{Var}(\hat{\beta}_1) = \sigma^2 / \sum(x_i - \bar{x})^2$:

    1. **Increase the spread of $X$ values.** Collecting data at more extreme values of $X$ increases $\sum(x_i - \bar{x})^2$, reducing variance. In experimental settings, this means choosing treatment levels that are far apart rather than closely spaced.

    2. **Increase the sample size $n$.** More observations increase $\sum(x_i - \bar{x})^2$ (assuming the new observations have similar spread), reducing variance proportionally.

    A third strategy is to reduce $\sigma^2$ by controlling for extraneous sources of variability (adding covariates or improving measurement precision), though this changes the model.
