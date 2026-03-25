# Normality in Regression (Residual Diagnostics)

## What the Assumption Requires

A common misconception is that linear regression requires the predictors or the response variable to be normally distributed. In fact, the normality assumption in regression applies to the **error terms**, not to the variables themselves. In the standard linear regression model

$$
Y_i = \beta_0 + \beta_1 X_{i1} + \cdots + \beta_p X_{ip} + \varepsilon_i
$$

the assumption is

$$
\varepsilon_i \overset{\text{iid}}{\sim} N(0, \sigma^2)
$$

This is equivalent to saying that, conditional on the predictors, the response is normally distributed:

$$
Y_i \mid X_i \sim N(\beta_0 + \beta_1 X_{i1} + \cdots + \beta_p X_{ip},\; \sigma^2)
$$

The predictors $X_{ij}$ can have any distribution. A skewed predictor, a binary predictor, or a heavy-tailed predictor does not violate the normality assumption.

??? warning "Normality Applies to Residuals, Not Predictors"
    Testing normality on the predictor variables or on the unconditional distribution of $Y$ is a common but incorrect practice. The normality check must be performed on the residuals $\hat{\varepsilon}_i = Y_i - \hat{Y}_i$, which estimate the unobservable errors $\varepsilon_i$.

## Why Normality Matters in Regression

The normality of errors is not needed for the OLS estimates $\hat{\beta}$ to be unbiased or consistent. The Gauss-Markov theorem guarantees that OLS is the Best Linear Unbiased Estimator (BLUE) under weaker conditions (linearity, exogeneity, homoscedasticity, no multicollinearity). Normality enters when we need:

1. **Exact $t$-tests and $F$-tests.** Under normality, the test statistic for $H_0: \beta_j = 0$,

    $$
    t_j = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)}
    $$

    follows a $t$-distribution with $n - p - 1$ degrees of freedom exactly. Without normality, this distributional result is only approximate (via the CLT for large $n$).

2. **Exact confidence intervals.** The $100(1-\alpha)\%$ confidence interval

    $$
    \hat{\beta}_j \pm t_{\alpha/2,\, n-p-1} \cdot \text{SE}(\hat{\beta}_j)
    $$

    has exact coverage only under normality. Without it, the coverage is approximate.

3. **Prediction intervals.** The prediction interval for a new observation is

    $$
    \hat{Y}_{\text{new}} \pm t_{\alpha/2,\, n-p-1} \cdot \hat{\sigma}\sqrt{1 + \mathbf{x}_{\text{new}}^T (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{x}_{\text{new}}}
    $$

    This interval relies on the normality of the errors to guarantee that the prediction error is normally distributed. Non-normal errors can cause the actual coverage to differ substantially from $1 - \alpha$.

## Checking Normality of Residuals

Since the true errors $\varepsilon_i$ are unobservable, we check normality using the **residuals** $\hat{\varepsilon}_i = Y_i - \hat{Y}_i$. The standard diagnostic tools are:

### Q-Q Plot of Residuals

The Q-Q plot compares the quantiles of the residuals against the quantiles of a standard normal distribution. If the errors are normal, the points should fall approximately along a straight line. Common patterns of departure include:

- **S-shaped curve**: indicates heavy tails (positive excess kurtosis)
- **Upward curvature**: indicates right skewness
- **Downward curvature at both ends**: indicates light tails (platykurtic)

### Shapiro-Wilk Test on Residuals

Apply the Shapiro-Wilk test to the residuals:

- $H_0$: The residuals are normally distributed
- $H_1$: The residuals are not normally distributed

For large $n$, the test may reject due to trivial departures; graphical methods become more informative in such cases.

### Histogram of Residuals

A histogram of residuals provides a quick visual check. Look for approximate symmetry and a bell-shaped profile. Strong skewness or multiple modes are signs of non-normality.

## Consequences of Non-Normal Residuals

The impact of non-normal residuals depends on the goal of the analysis:

**Point estimation.** OLS estimates remain unbiased and consistent regardless of the error distribution, as long as the other Gauss-Markov conditions hold. Non-normality does not affect $\hat{\beta}$.

**Hypothesis tests and CIs.** For small $n$, non-normal errors can distort $p$-values and coverage probabilities. For large $n$, the CLT ensures that $\hat{\beta}_j$ is approximately normal, so the $t$-tests and CIs remain approximately valid.

**Prediction intervals.** Prediction intervals are more sensitive to non-normality than confidence intervals for $\hat{\beta}_j$, because the prediction error includes the full error term $\varepsilon_{\text{new}}$ whose distribution directly enters the interval.

## Python Example

```python
import numpy as np
from scipy import stats

# ===================================================================
# Check normality of regression residuals
# ===================================================================

np.random.seed(42)
n = 100

# Generate data with normal errors
X = np.random.uniform(0, 10, size=n)
beta_0, beta_1 = 2.0, 3.0
epsilon = np.random.normal(0, 2, size=n)
Y = beta_0 + beta_1 * X + epsilon

# Fit OLS regression (manually via normal equations)
X_design = np.column_stack([np.ones(n), X])
beta_hat = np.linalg.lstsq(X_design, Y, rcond=None)[0]
Y_hat = X_design @ beta_hat
residuals = Y - Y_hat

# Normality test on residuals
sw_stat, sw_p = stats.shapiro(residuals)

# Skewness and kurtosis of residuals
skew_r = stats.skew(residuals)
kurt_r = stats.kurtosis(residuals)  # excess kurtosis

if __name__ == "__main__":
    print(f"Estimated coefficients: beta_0 = {beta_hat[0]:.3f}, "
          f"beta_1 = {beta_hat[1]:.3f}")
    print(f"\nResidual diagnostics:")
    print(f"  Shapiro-Wilk: W = {sw_stat:.4f}, p = {sw_p:.4f}")
    print(f"  Skewness:     {skew_r:.4f}")
    print(f"  Excess kurtosis: {kurt_r:.4f}")

    if sw_p > 0.05:
        print("\n  Residuals are consistent with normality.")
    else:
        print("\n  Evidence of non-normality in residuals.")
```

## Remedies for Non-Normal Residuals

When residual diagnostics reveal non-normality, several approaches can help:

1. **Transform the response.** A log or Box-Cox transformation of $Y$ can reduce skewness in the residuals. The transformed model is $g(Y_i) = \beta_0 + \beta_1 X_{i1} + \cdots + \varepsilon_i$.

2. **Remove outliers.** If the non-normality is driven by a few extreme residuals, investigate whether these observations are data errors or influential points.

3. **Use robust standard errors.** Heteroscedasticity-consistent (HC) standard errors provide valid inference without normality, at least asymptotically.

4. **Use bootstrap inference.** Bootstrap confidence intervals and $p$-values do not require the normality assumption.

5. **Use a generalized linear model.** If the response is inherently non-normal (e.g., counts, binary outcomes), a GLM with an appropriate link function is more appropriate than OLS with a transformation.

## Summary

In linear regression, the normality assumption applies to the error terms, not to the predictors or the unconditional response. Normality is needed for exact $t$-tests, $F$-tests, and prediction intervals, but OLS point estimates remain valid without it. Diagnostics should be performed on the residuals using Q-Q plots, formal tests, and histograms. When non-normality is detected, the practical impact depends on the sample size and whether the goal is estimation, testing, or prediction.
