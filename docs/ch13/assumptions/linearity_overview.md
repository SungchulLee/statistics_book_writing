# Linearity Assumption

## Definition

The assumption of linearity posits that there is a straight-line relationship between the dependent variable and each independent variable. This means that the change in the dependent variable is proportional to the change in the independent variables.

## Mathematical Representation

In a simple linear regression model with one independent variable, the relationship can be expressed as:

$$
Y = \beta_0 + \beta_1 X + \epsilon
$$

where:

- $Y$ is the dependent variable.
- $\beta_0$ is the intercept.
- $\beta_1$ is the slope of the regression line.
- $X$ is the independent variable.
- $\epsilon$ is the error term.

For multiple linear regression with $p$ predictors:

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p + \epsilon
$$

The linearity assumption requires that the expected value of $Y$ is a linear function of the $X$ variables:

$$
E[Y \mid X_1, \ldots, X_p] = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p
$$

## Importance

Linearity is crucial because if the relationship between the variables is not linear, the model will either overestimate or underestimate the true relationship, leading to:

- **Biased predictions** — Systematic errors in the predicted values.
- **Invalid statistical inferences** — Confidence intervals and hypothesis tests become unreliable.
- **Poor model fit** — The model fails to capture the true pattern in the data.

## Diagnostics

To check the linearity assumption:

- **Residual Plots:** Plot the residuals (the differences between observed and predicted values) against the independent variables. If the residuals are randomly scattered around the horizontal axis, the linearity assumption is likely satisfied. However, patterns like curves or clusters suggest non-linearity.
- **Scatterplots:** Visual inspection of scatterplots between each independent variable and the dependent variable can also provide insights into whether the relationship is linear.
- **Component-Plus-Residual Plots:** In multiple regression, partial residual plots allow assessment of linearity for each predictor individually.

## Remedies for Non-Linearity

- **Transformations:** Apply transformations to the dependent or independent variables, such as logarithmic, square root, or polynomial transformations, to achieve linearity.
- **Polynomial Terms:** Add squared or cubic terms of the independent variable to capture curvature within the linear regression framework.
- **Non-linear Models:** Consider using non-linear regression models if transformations do not resolve the issue.

For detailed diagnostic methods, see [Checking Linearity](checking_linearity.md).
