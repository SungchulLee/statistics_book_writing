# statsmodels Ordinary Least Squares Interface

The `statsmodels` library is Python's primary tool for statistical inference in regression. Unlike machine learning libraries that focus on prediction, `statsmodels` provides detailed summary tables with coefficient estimates, standard errors, t-statistics, p-values, and confidence intervals — the standard output expected in statistical analysis.

---

## 1. The OLS Class

The core interface for ordinary least squares regression is `statsmodels.api.OLS`. This class requires the user to explicitly add a constant (intercept) column to the predictor matrix.

```python
import numpy as np
import statsmodels.api as sm

# Generate example data
np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
beta_true = np.array([3.0, 1.5])
y = X @ beta_true + 2.0 + np.random.randn(n) * 0.5

# Add constant for intercept
X_with_const = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X_with_const)
results = model.fit()

print(results.summary())
```

The `sm.add_constant(X)` function prepends a column of ones to the predictor matrix, corresponding to the intercept term $\beta_0$ in the model $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \varepsilon$.

!!! warning "Forgetting the constant"
    If you pass `X` without calling `sm.add_constant()`, the model fits a regression through the origin (no intercept). This is almost never what you want. Always add the constant unless you have a specific reason to suppress the intercept.

---

## 2. Interpreting the Summary Table

The `results.summary()` output contains three panels. The most important fields are:

### Top Panel (Model Information)

| Field | Meaning |
|---|---|
| R-squared | Proportion of variance explained ($R^2$) |
| Adj. R-squared | $R^2$ adjusted for the number of predictors |
| F-statistic | Overall F-test for the significance of the regression |
| Prob (F-statistic) | P-value for the F-test |
| AIC / BIC | Information criteria for model comparison |

### Middle Panel (Coefficients)

| Column | Meaning |
|---|---|
| coef | Estimated regression coefficient $\hat{\beta}_j$ |
| std err | Standard error of $\hat{\beta}_j$ |
| t | t-statistic: $t = \hat{\beta}_j / \text{SE}(\hat{\beta}_j)$ |
| P>\|t\| | Two-sided p-value for $H_0: \beta_j = 0$ |
| [0.025, 0.975] | 95% confidence interval for $\beta_j$ |

A predictor is statistically significant at the 5% level when its p-value is below 0.05, equivalently when the 95% confidence interval does not contain zero.

### Bottom Panel (Diagnostics)

| Field | Meaning |
|---|---|
| Omnibus / Prob(Omnibus) | Test for normality of residuals |
| Durbin-Watson | Test for autocorrelation in residuals (values near 2 indicate no autocorrelation) |
| Jarque-Bera / Prob(JB) | Another normality test based on skewness and kurtosis |
| Cond. No. | Condition number of the design matrix (high values suggest multicollinearity) |

---

## 3. Formula API

The `statsmodels` formula API provides an R-like syntax using `patsy` formulas. This interface automatically adds the intercept and handles categorical variables.

```python
import pandas as pd
import statsmodels.formula.api as smf

# Create a DataFrame
df = pd.DataFrame({
    'y': y,
    'x1': X[:, 0],
    'x2': X[:, 1]
})

# Fit using formula API
results_formula = smf.ols('y ~ x1 + x2', data=df).fit()
print(results_formula.summary())
```

The formula `'y ~ x1 + x2'` specifies the model $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \varepsilon$. The intercept is included by default. To suppress it, use `'y ~ x1 + x2 - 1'`.

### Formula Syntax

| Formula | Model |
|---|---|
| `y ~ x1 + x2` | $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2$ |
| `y ~ x1 * x2` | $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2$ |
| `y ~ x1 + I(x1**2)` | $y = \beta_0 + \beta_1 x_1 + \beta_2 x_1^2$ |
| `y ~ C(group)` | One-hot encoding of categorical variable |
| `y ~ x1 + x2 - 1` | No intercept |

The `I()` wrapper tells `patsy` to interpret the expression arithmetically rather than as a formula operator. Without it, `x1**2` would not be recognized as "x1 squared."

---

## 4. Accessing Results Programmatically

The fitted `results` object stores all quantities needed for further analysis:

```python
# Coefficient estimates
print("Coefficients:", results.params)

# Standard errors
print("Standard errors:", results.bse)

# P-values
print("P-values:", results.pvalues)

# Confidence intervals
print("95% CI:\n", results.conf_int(alpha=0.05))

# R-squared and adjusted R-squared
print("R-squared:", results.rsquared)
print("Adjusted R-squared:", results.rsquared_adj)

# Residuals
residuals = results.resid

# Fitted values
fitted = results.fittedvalues

# AIC and BIC
print("AIC:", results.aic)
print("BIC:", results.bic)
```

---

## 5. Diagnostic Methods

The `results` object provides methods for model diagnostics:

```python
# Influence diagnostics (leverage, Cook's distance)
influence = results.get_influence()
cooks_d = influence.cooks_distance[0]
leverage = influence.hat_matrix_diag

# Heteroscedasticity tests
from statsmodels.stats.diagnostic import het_breuschpagan
bp_stat, bp_pval, _, _ = het_breuschpagan(results.resid, results.model.exog)
print(f"Breusch-Pagan p-value: {bp_pval:.4f}")

# Normality test on residuals
from statsmodels.stats.stattools import jarque_bera
jb_stat, jb_pval, skew, kurtosis = jarque_bera(results.resid)
print(f"Jarque-Bera p-value: {jb_pval:.4f}")

# Variance Inflation Factors
from statsmodels.stats.outliers_influence import variance_inflation_factor
for i in range(X_with_const.shape[1]):
    vif = variance_inflation_factor(X_with_const, i)
    print(f"VIF for variable {i}: {vif:.2f}")
```

!!! tip "When to use statsmodels"
    Use `statsmodels` when your primary goal is statistical inference: testing hypotheses about coefficients, constructing confidence intervals, and diagnosing model assumptions. For pure prediction tasks where inference is not needed, `sklearn.linear_model.LinearRegression` offers a simpler interface.

## Exercises

**Exercise 1.**
The following is the result of performing a linear regression analysis using the `statsmodels` package. The analysis predicts **sales** based on advertising expenses allocated to **TV**, **radio**, and **newspaper** media:

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  Sales   R-squared:                       0.894
Model:                            OLS   Adj. R-squared:                  0.891
Method:                 Least Squares   F-statistic:                     381.2
Date:                Mon, 11 Nov 2024   Prob (F-statistic):           5.60e-66
Time:                        02:39:45   Log-Likelihood:                -273.89
No. Observations:                 140   AIC:                             555.8
Df Residuals:                     136   BIC:                             567.5
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      3.0451      0.391      7.782      0.000       2.271       3.819
TV             0.0470      0.002     27.653      0.000       0.044       0.050
Radio          0.1797      0.011     16.665      0.000       0.158       0.201
Newspaper     -0.0030      0.007     -0.428      0.669      -0.017       0.011
==============================================================================
Omnibus:                       50.782   Durbin-Watson:                   2.089
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              131.355
Skew:                          -1.459   Prob(JB):                     3.00e-29
Kurtosis:                       6.741   Cond. No.                         457.
==============================================================================
```

**(a)** If advertising expenses for TV, radio, and newspapers are denoted by $x_1$, $x_2$, and $x_3$, respectively, and sales are denoted by $y$, what is the predicted value $\hat{y}$ based on the regression results?

**(b)** The coefficient for newspaper advertising is $-0.0030$. Can this be interpreted as suggesting that newspaper advertising decreases sales? Discuss its validity based on the $p$-value.

**(c)** What do the Jarque-Bera (JB) statistic (131.355) and its $p$-value (3.00e-29) signify?

??? success "Solution to Exercise 1"

    **(a)** The predicted value of sales is:

    $$
    \hat{y} = 3.0451 + 0.0470 \cdot x_1 + 0.1797 \cdot x_2 - 0.0030 \cdot x_3
    $$

    This equation combines the intercept and coefficients for each advertising medium to predict sales.

    **(b)** The $p$-value for the newspaper coefficient is $0.669$, which is much greater than the standard significance level of 0.05. This means we fail to reject the null hypothesis that the coefficient is zero. While the coefficient is negative, its large $p$-value implies this result is not statistically meaningful. It would be more appropriate to conclude that newspaper advertising does not have a statistically significant impact on sales, rather than interpreting it as having a negative effect.

    **(c)** The Jarque-Bera test assesses whether the residuals follow a normal distribution. The extremely small $p$-value ($3.00 \times 10^{-29}$) indicates that the null hypothesis (residuals are normally distributed) is strongly rejected. This suggests the residuals are likely **not normally distributed**, which could imply model issues such as non-normal errors or the presence of outliers.
