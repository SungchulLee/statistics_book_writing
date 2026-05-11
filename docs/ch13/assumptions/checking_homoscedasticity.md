# Checking Homoscedasticity in Linear Regression


Homoscedasticity, one of the key assumptions in linear regression, refers to the condition where the variance of the residuals (errors) is constant across all levels of the independent variables. If this assumption is violated, meaning the residuals exhibit non-constant variance (heteroscedasticity), it can lead to inefficient estimates, biased standard errors, and unreliable hypothesis tests. This section provides an overview of methods to check for homoscedasticity in linear regression, including visual inspections and statistical tests.

## 1. Understanding Homoscedasticity

**Definition:**
Homoscedasticity means that the spread (variance) of the residuals is the same across all levels of the independent variables. In other words, no matter what the value of the independent variable is, the distribution of the errors should be roughly the same.

Formally:

$$
\text{Var}(\epsilon_i \mid X_i) = \sigma^2 \quad \text{for all } i
$$

**Why It Matters:**
When homoscedasticity is violated:

- **Standard Errors:** The standard errors of the coefficients may be biased, leading to incorrect confidence intervals and hypothesis tests.
- **Model Efficiency:** The ordinary least squares (OLS) estimates may still be unbiased but are no longer efficient, meaning there could be better, more precise estimates available. Specifically, OLS is no longer the Best Linear Unbiased Estimator (BLUE) under the Gauss-Markov theorem.

## 2. Residuals vs. Fitted Values Plot

The **Residuals vs. Fitted Values plot** is one of the most common and effective methods for visually checking homoscedasticity. This plot helps you see if there is any systematic pattern in the spread of residuals.

**Steps:**

1. **Fit the Linear Regression Model:** Fit your model and obtain the residuals and fitted values.
2. **Create the Plot:** Plot the residuals on the y-axis and the fitted values on the x-axis.
3. **Assess the Plot:** Look for patterns in the spread of the residuals.

**Example:**

```python
import matplotlib.pyplot as plt

# Assuming 'model' is your fitted OLS model
residuals = model.resid
fitted = model.fittedvalues

plt.scatter(fitted, residuals, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.show()
```

**Interpretation:**

- **Homoscedasticity:** If the residuals are scattered randomly around the horizontal line (zero) with a constant spread, homoscedasticity is likely.
- **Heteroscedasticity:** If the spread of the residuals increases or decreases with the fitted values (forming a funnel shape, for example), this indicates heteroscedasticity.

## 3. Breusch-Pagan Test

The **Breusch-Pagan test** is a formal statistical test used to detect heteroscedasticity. It assesses whether the variance of the residuals is dependent on the independent variables.

**Hypotheses:**

- $H_0$: Homoscedasticity — $\text{Var}(\epsilon_i) = \sigma^2$ (constant)
- $H_1$: Heteroscedasticity — $\text{Var}(\epsilon_i)$ depends on one or more independent variables

**Procedure:**

The test regresses the squared residuals $e_i^2$ on the independent variables:

$$
e_i^2 = \gamma_0 + \gamma_1 X_{1i} + \gamma_2 X_{2i} + \cdots + \gamma_p X_{pi} + u_i
$$

The test statistic is $nR^2$ from this auxiliary regression, which follows a $\chi^2(p)$ distribution under $H_0$.

**Steps:**

1. **Fit the Linear Regression Model:** Obtain the residuals from your fitted model.
2. **Perform the Breusch-Pagan Test:** The test calculates a statistic based on the residuals.
3. **Interpret the Results:** A significant p-value (typically < 0.05) suggests heteroscedasticity.

**Example:**

```python
from statsmodels.stats.diagnostic import het_breuschpagan

# Assuming 'model' is your fitted OLS model and 'X' is the independent variable(s)
bp_test = het_breuschpagan(model.resid, model.model.exog)
labels = ['LM Statistic', 'LM p-value', 'F-Statistic', 'F p-value']
for label, value in zip(labels, bp_test):
    print(f'{label}: {value:.4f}')
```

**Interpretation:**

- **p-value > 0.05:** No significant evidence of heteroscedasticity.
- **p-value < 0.05:** Significant evidence of heteroscedasticity, indicating a violation of the homoscedasticity assumption.

## 4. White Test

The **White test** is another statistical test that not only checks for heteroscedasticity but also for more general forms of model misspecification, including nonlinearity.

**Key Difference from Breusch-Pagan:**

The White test includes not only the original independent variables but also their **squares** and **cross-products** in the auxiliary regression:

$$
e_i^2 = \gamma_0 + \gamma_1 X_{1i} + \gamma_2 X_{2i} + \gamma_3 X_{1i}^2 + \gamma_4 X_{2i}^2 + \gamma_5 X_{1i} X_{2i} + u_i
$$

This makes it more general but also uses more degrees of freedom.

**Steps:**

1. **Fit the Linear Regression Model:** Obtain the residuals from your fitted model.
2. **Perform the White Test:** The test involves regressing the squared residuals on the independent variables and their squares and cross-products.
3. **Interpret the Results:** A significant p-value indicates heteroscedasticity or other forms of misspecification.

**Example:**

```python
from statsmodels.stats.diagnostic import het_white

# Assuming 'model' is your fitted OLS model
white_test = het_white(model.resid, model.model.exog)
labels = ['LM Statistic', 'LM p-value', 'F-Statistic', 'F p-value']
for label, value in zip(labels, white_test):
    print(f'{label}: {value:.4f}')
```

**Interpretation:**

- **p-value > 0.05:** No significant evidence of heteroscedasticity or other misspecifications.
- **p-value < 0.05:** Significant evidence of heteroscedasticity or other model issues.

## 5. Scale-Location Plot

The **Scale-Location plot** (or Spread-Location plot) is another useful visualization for detecting heteroscedasticity. It plots the square root of the standardized residuals against the fitted values.

**Why the square root?** Taking the square root of the absolute standardized residuals linearizes the relationship, making trends in variance easier to detect visually. The standardization removes the effect of the mean, isolating the variance pattern.

**Steps:**

1. **Fit the Linear Regression Model:** Obtain the standardized residuals and fitted values.
2. **Create the Plot:** Plot the square root of the absolute standardized residuals on the y-axis and the fitted values on the x-axis.
3. **Assess the Plot:** Look for patterns in the spread of the residuals.

**Example:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'model' is your fitted OLS model
residuals = model.resid
fitted = model.fittedvalues
standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)

plt.scatter(fitted, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
plt.xlabel('Fitted Values')
plt.ylabel('√|Standardized Residuals|')
plt.title('Scale-Location Plot')

# Add a lowess smoothing line for trend detection
from statsmodels.nonparametric.smoothers_lowess import lowess
smooth = lowess(np.sqrt(np.abs(standardized_residuals)), fitted, frac=0.6)
plt.plot(smooth[:, 0], smooth[:, 1], color='red', linewidth=2)
plt.show()
```

**Interpretation:**

- **Homoscedasticity:** The points should be randomly scattered around a horizontal line with no clear pattern. The smoothing line should be approximately flat.
- **Heteroscedasticity:** A pattern (such as an upward trend or funnel shape) suggests non-constant variance, indicating heteroscedasticity.

## Comparison of Homoscedasticity Tests

| Test | Type | What It Detects | Pros | Cons |
|------|------|----------------|------|------|
| Residuals vs. Fitted | Visual | Any variance pattern | Intuitive, flexible | Subjective |
| Scale-Location | Visual | Variance trends | Clear trend visualization | Subjective |
| Breusch-Pagan | Formal | Linear heteroscedasticity | Simple, powerful | Assumes linear form |
| White | Formal | General heteroscedasticity + nonlinearity | Very general | Uses many degrees of freedom |

Checking for homoscedasticity is essential in linear regression analysis to ensure the validity and efficiency of the model's estimates. If heteroscedasticity is detected, it can often be addressed by transforming the dependent variable, using weighted least squares, or employing robust standard errors (such as HC0, HC1, HC2, or HC3 estimators) to correct for the non-constant variance.
## Exercises

**Exercise 1.**
A residual-vs-fitted plot shows residuals forming a clear funnel shape that widens to the right. Name the formal statistical test you would use to confirm this visual diagnosis and state its null hypothesis.

??? success "Solution to Exercise 1"
    The **Breusch-Pagan test** is appropriate. Its null hypothesis is $H_0$: the variance of the errors is constant (homoscedasticity). The test regresses the squared residuals on the original predictors and tests whether the resulting $R^2$ is significantly different from zero. A significant result confirms heteroscedasticity.

    Alternatively, the **White test** can be used; it also tests for heteroscedasticity but additionally checks for nonlinearity by including squared predictors and cross-products.

---

**Exercise 2.**
After detecting heteroscedasticity, a researcher applies a log transformation to the response variable and the funnel pattern disappears. Explain why this works for data where variance is proportional to the mean.

??? success "Solution to Exercise 2"
    When $\text{Var}(Y|X) \propto E[Y|X]$ (variance proportional to the mean), larger fitted values have larger residual spread, creating the funnel shape. The log transformation compresses large values more than small values.

    By the delta method, if $Y$ has variance proportional to $\mu^2$, then $\text{Var}(\log Y) \approx \text{Var}(Y)/\mu^2 = \text{constant}$. This stabilizes the variance, making residuals approximately homoscedastic.

---

**Exercise 3.**
Compare and contrast three remedies for heteroscedasticity: (1) weighted least squares, (2) robust standard errors, and (3) variable transformation. In which situations is each preferred?

??? success "Solution to Exercise 3"

    1. **Weighted least squares (WLS):** Assigns weights inversely proportional to the error variance. Preferred when the form of heteroscedasticity is known (e.g., variance proportional to a known variable). Produces efficient estimates.

    2. **Robust (HC) standard errors:** Leaves the OLS coefficient estimates unchanged but corrects the standard errors. Preferred when the goal is valid inference without changing the model or making assumptions about the variance structure.

    3. **Variable transformation:** Transforms $Y$ (e.g., log, square root) to stabilize variance. Preferred when the transformation also improves linearity or normality, addressing multiple assumption violations simultaneously. However, it changes the interpretation of the coefficients.
