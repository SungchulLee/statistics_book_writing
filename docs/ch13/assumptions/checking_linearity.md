# Checking Linearity in Linear Regression


Linearity is a foundational assumption in linear regression that posits a straight-line relationship between the dependent variable and each independent variable. Ensuring that this assumption holds is critical for the validity of the regression model. If the relationship between the variables is not linear, the model may yield biased estimates, resulting in poor predictions and incorrect inferences. This section explores various methods for assessing linearity in linear regression.

## 1. Visual Inspection Using Scatterplots

**Scatterplots** are one of the simplest and most intuitive ways to check for linearity. By plotting each independent variable against the dependent variable, you can visually inspect whether the relationship appears to be a straight line.

**Steps:**

1. **Plot the Data:** For each independent variable, create a scatterplot with the dependent variable on the y-axis and the independent variable on the x-axis.
2. **Assess the Shape:** Observe the overall shape of the data points. A linear relationship will typically appear as a cloud of points centered around a straight line.
3. **Look for Patterns:** If the data points form a curve or any other non-linear pattern, the linearity assumption is likely violated.

**Example:**

```python
import matplotlib.pyplot as plt

# Assuming X is the independent variable and Y is the dependent variable
plt.scatter(X, Y)
plt.xlabel('Independent Variable')
plt.ylabel('Dependent Variable')
plt.title('Scatterplot of Y vs X')
plt.show()
```

**Interpretation:**

- A straight-line pattern indicates that the linearity assumption is likely satisfied.
- Curves, clusters, or any non-linear patterns suggest a potential violation of the linearity assumption.

## 2. Residual Plots

A **residual plot** is another powerful tool for checking linearity. Residuals are the differences between the observed values and the values predicted by the regression model. Plotting residuals against predicted values can help you assess whether the linearity assumption holds.

**Steps:**

1. **Fit the Linear Regression Model:** First, fit your linear regression model to obtain the predicted values.
2. **Plot the Residuals:** Create a plot with the predicted values on the x-axis and the residuals on the y-axis.
3. **Assess the Residuals:** Check whether the residuals are randomly scattered around the horizontal axis.

**Example:**

```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Assuming X is the independent variable and Y is the dependent variable
model = sm.OLS(Y, sm.add_constant(X)).fit()
predictions = model.predict(sm.add_constant(X))
residuals = Y - predictions

plt.scatter(predictions, residuals)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.axhline(y=0, color='red', linestyle='--')
plt.show()
```

**Interpretation:**

- **Random Scatter:** If the residuals are randomly scattered around zero without any clear pattern, the linearity assumption is likely met.
- **Patterns in Residuals:** A curved pattern, systematic clustering, or any structure in the residuals indicates non-linearity, suggesting that a linear model may not be appropriate.

## 3. Component-Plus-Residual Plots (Partial Residual Plots)

**Component-Plus-Residual (CPR) plots**, also known as **Partial Residual Plots**, extend the concept of residual plots by allowing you to assess the linearity of individual predictors in a multiple regression setting.

**Mathematical Definition:**

For a predictor $X_j$ in a multiple regression model, the partial residual is defined as:

$$
e_j^{(\text{partial})} = \hat{\beta}_j X_j + e
$$

where $\hat{\beta}_j$ is the estimated coefficient for $X_j$ and $e$ is the ordinary residual from the full model.

**Steps:**

1. **Fit the Full Model:** Fit your multiple linear regression model.
2. **Compute the Partial Residuals:** For each independent variable, plot the partial residuals, which are the residuals plus the product of the estimated coefficient and the predictor.
3. **Assess Linearity:** Check if the relationship between the partial residuals and the predictor is linear.

**Example:**

```python
from statsmodels.graphics.regressionplots import plot_partregress
import matplotlib.pyplot as plt

# Assuming X is the independent variable and Y is the dependent variable
# And other_vars are other independent variables in the model
fig, ax = plt.subplots(figsize=(8, 6))
plot_partregress(endog='Y', exog='X', exog_others=other_vars, data=df, ax=ax)
plt.show()
```

**Interpretation:**

- **Linear Trend:** A straight-line trend in the CPR plot suggests that the linearity assumption for that predictor is met.
- **Non-Linear Trend:** A curved or non-linear trend suggests that the relationship between the predictor and the dependent variable is not linear.

## 4. Adding Polynomial Terms

If the relationship between the dependent variable and an independent variable is not linear, adding **polynomial terms** (e.g., squared or cubic terms) to the regression model can help in capturing the non-linear relationship. This method allows the model to account for curvature while still using a linear regression framework.

**Mathematical Formulation:**

A quadratic polynomial model for a single predictor:

$$
Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \epsilon
$$

A cubic polynomial model:

$$
Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3 + \epsilon
$$

Note that these are still **linear regression** models because they are linear in the parameters $\beta_0, \beta_1, \beta_2, \beta_3$.

**Steps:**

1. **Include Polynomial Terms:** Add higher-order terms of the independent variable to the regression model.
2. **Refit the Model:** Fit the model with these polynomial terms.
3. **Assess Linearity:** Check whether the addition of polynomial terms improves the fit of the model (e.g., by comparing $R^2$ values or using AIC/BIC).

**Example:**

```python
import numpy as np
import statsmodels.api as sm

# Assuming X is the independent variable and Y is the dependent variable
X_poly = np.column_stack((X, X**2))
model = sm.OLS(Y, sm.add_constant(X_poly)).fit()
print(model.summary())
```

**Interpretation:**

- **Improved Fit:** If the polynomial model significantly improves the fit (e.g., higher $R^2$, significant coefficient on the squared term), it suggests that the original relationship was non-linear.
- **No Improvement:** If there is no significant improvement, the original linearity assumption may still be valid.

## Summary of Linearity Diagnostics

| Method | Type | Best For | Key Indicator |
|--------|------|----------|---------------|
| Scatterplot | Visual | Simple regression, initial assessment | Non-linear pattern in data cloud |
| Residual Plot | Visual | Any regression model | Curved pattern in residuals |
| CPR Plot | Visual | Multiple regression | Non-linear trend in partial residuals |
| Polynomial Terms | Formal | Testing specific non-linear relationships | Significant higher-order coefficients |

Ensuring the linearity assumption in linear regression is critical for producing accurate and interpretable models. The methods discussed provide robust tools for diagnosing and addressing potential violations of this assumption. By carefully checking linearity and making necessary adjustments, you can enhance the reliability of your linear regression models and the validity of the conclusions drawn from them.
## Exercises

**Exercise 1.**
A component-plus-residual (CPR) plot for predictor $X_2$ in a multiple regression shows a clear U-shaped curve. Describe two ways to modify the model to address this non-linearity.

??? success "Solution to Exercise 1"

    1. **Add a polynomial term:** Include $X_2^2$ as an additional predictor in the model, changing the specification to $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_2^2 + \varepsilon$. This captures the quadratic relationship within the linear regression framework.

    2. **Apply a transformation:** Transform $X_2$ using $\log(X_2)$, $\sqrt{X_2}$, or another monotone transformation that linearizes the relationship before fitting the regression.

---

**Exercise 2.**
Explain the difference between a scatterplot of $Y$ vs. $X_j$ and a partial residual plot for $X_j$ in a multiple regression. When would they give different conclusions about linearity?

??? success "Solution to Exercise 2"
    A **scatterplot** of $Y$ vs. $X_j$ shows the marginal relationship, which may be confounded by the effects of other predictors. A **partial residual plot** (component-plus-residual plot) plots $e + \hat{\beta}_j X_j$ vs. $X_j$, isolating the relationship between $Y$ and $X_j$ after removing the effects of all other predictors.

    They differ when other predictors are correlated with $X_j$. For example, if $X_1$ and $X_2$ are positively correlated and both affect $Y$, the scatterplot of $Y$ vs. $X_2$ might appear linear (because $X_1$'s effect reinforces $X_2$'s), while the partial residual plot reveals a nonlinear partial relationship after adjusting for $X_1$.

---

**Exercise 3.**
A regression of house price on square footage appears linear in a scatterplot, but the residual-vs-fitted plot shows a subtle curvature. Explain why these diagnostics can disagree and which one to trust.

??? success "Solution to Exercise 3"
    The scatterplot shows the raw data, which may have large variance that masks subtle nonlinearity. The residual-vs-fitted plot removes the linear trend, making any remaining curvature more visible because the $y$-axis scale is compressed to the residual range.

    The **residual-vs-fitted plot** should be trusted because it directly assesses whether the model's linear assumption is adequate. The scatterplot is useful for initial exploration but cannot detect subtle departures from linearity when the signal-to-noise ratio is low.
