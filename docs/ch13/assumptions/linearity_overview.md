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
## Exercises

**Exercise 1.**
A simple linear regression of test scores on study hours produces a residual plot with a clear parabolic pattern. Write the mathematical model that would capture this nonlinearity and explain how it remains within the linear regression framework.

??? success "Solution to Exercise 1"
    The appropriate model is:

    $$
    Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \varepsilon
    $$

    Despite the $X^2$ term, this is still a **linear regression** model because "linear" refers to linearity in the **parameters** ($\beta_0, \beta_1, \beta_2$), not in the predictors. The model can be estimated by OLS after creating a new variable $X_2 = X^2$ and fitting $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \varepsilon$.

---

**Exercise 2.**
Explain the difference between a violation of linearity and a violation of the correct functional form. Can a model satisfy the linearity assumption yet still be misspecified?

??? success "Solution to Exercise 2"
    The **linearity assumption** states that $E[Y|X]$ is a linear function of the parameters. A **misspecified functional form** means the model omits relevant predictors or transformations, even if the included terms enter linearly.

    Yes, a model can satisfy linearity yet be misspecified. For example, $Y = \beta_0 + \beta_1 X + \varepsilon$ satisfies linearity in parameters, but if the true relationship is $Y = \beta_0 + \beta_1 X + \beta_2 Z + \varepsilon$ (omitting variable $Z$), the model is misspecified due to omitted variable bias, even though the linearity assumption is not violated.
