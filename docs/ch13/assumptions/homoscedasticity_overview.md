# Homoscedasticity Assumption

## Definition

Homoscedasticity refers to the assumption that the variance of the errors (residuals) is constant across all levels of the independent variables. When this assumption holds, the spread of residuals should be roughly the same across the range of predicted values.

Formally:

$$
\text{Var}(\epsilon_i \mid X_i) = \sigma^2 \quad \text{for all } i
$$

where $\sigma^2$ is a constant that does not depend on the value of $X_i$.

The opposite condition, **heteroscedasticity**, occurs when the variance of errors changes systematically with the level of the independent variable:

$$
\text{Var}(\epsilon_i \mid X_i) = \sigma_i^2 \quad \text{(not constant)}
$$

## Importance

Homoscedasticity is essential because heteroscedasticity leads to:

- **Inefficient estimates** — OLS estimates remain unbiased but are no longer the best linear unbiased estimators (BLUE). There exist more efficient estimators.
- **Biased standard errors** — The usual OLS standard errors are incorrect, leading to unreliable confidence intervals and hypothesis tests.
- **Invalid inference** — t-statistics and F-statistics may be too large or too small, producing misleading p-values.

## Common Patterns of Heteroscedasticity

| Pattern | Description | Example |
|---------|-------------|---------|
| Fan-shaped | Variance increases with fitted values | Income vs. spending data |
| Inverse fan | Variance decreases with fitted values | Aggregated data with varying group sizes |
| Bow-tie | Variance increases then decreases | Data with natural bounds |
| Grouped | Different variances across groups | Multi-group comparisons |

## Diagnostics

- **Residual vs. Fitted Plot:** Plot the residuals against the fitted values (predicted values) of the dependent variable. Homoscedasticity is suggested if the residuals are evenly spread around the horizontal axis without showing patterns or funnel shapes.
- **Breusch-Pagan Test:** This statistical test assesses the presence of heteroscedasticity by regressing the squared residuals on the independent variables. A significant result indicates heteroscedasticity.
- **White Test:** A more general test that also checks for nonlinearity in addition to heteroscedasticity.
- **Scale-Location Plot:** Plots the square root of standardized residuals against fitted values.

## Remedies for Heteroscedasticity

- **Transformations:** Applying a logarithmic or square root transformation to the dependent variable can stabilize the variance of the residuals.
- **Weighted Least Squares (WLS):** WLS assigns different weights to different observations to account for heteroscedasticity, giving less weight to observations with higher variance.
- **Robust Standard Errors:** Heteroscedasticity-consistent (HC) standard errors (White's robust standard errors) provide valid inference without transforming the model.

For detailed diagnostic methods, see [Checking Homoscedasticity](checking_homoscedasticity.md).
