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
## Exercises

**Exercise 1.**
In a simple linear regression of salary on years of experience, the variance of residuals increases with experience level. Explain why OLS estimates remain unbiased under heteroscedasticity but are no longer efficient.

??? success "Solution to Exercise 1"
    OLS estimates are unbiased because the Gauss-Markov proof of unbiasedness ($E[\hat{\beta}] = \beta$) requires only that $E[\varepsilon|X] = 0$, which does not depend on constant variance.

    However, OLS is no longer **efficient** (no longer BLUE) because it assigns equal weight to all observations. Under heteroscedasticity, observations with larger variance carry less information about the regression line. Weighted least squares (which weights observations inversely to their variance) produces more efficient estimates.

---

**Exercise 2.**
Write the mathematical definition of heteroscedasticity and explain how it differs from homoscedasticity.

??? success "Solution to Exercise 2"
    **Homoscedasticity:** $\text{Var}(\varepsilon_i | X_i) = \sigma^2$ for all $i$ (constant variance).

    **Heteroscedasticity:** $\text{Var}(\varepsilon_i | X_i) = \sigma_i^2$ where $\sigma_i^2$ varies with $X_i$ (non-constant variance).

    Under homoscedasticity, the spread of residuals is the same across all values of $X$. Under heteroscedasticity, the spread changes systematically -- for example, increasing with $X$ (fan-shaped residual plot) or differing across groups.
