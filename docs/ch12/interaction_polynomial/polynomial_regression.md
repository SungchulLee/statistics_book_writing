# Polynomial Regression

## Why Use Polynomial Regression?

**Polynomial regression** models non-linear relationships by including powers of the predictors in the regression equation. Unlike simple linear regression, which assumes a straight-line relationship, polynomial regression can model curves of various shapes.

Non-linear relationships appear frequently in practice:

- In physics, the relationship between distance and time for an accelerating object is quadratic: $y = ax^2 + bx + c$.
- In economics, diminishing returns often follow a polynomial relationship between input and output.
- In finance, the volatility smile suggests a curved relationship between option strike prices and implied volatility.

---

## Mathematical Formulation

Polynomial regression extends linear regression by including polynomial terms of the predictors:

$$
Y = \beta_0 + \beta_1 X + \beta_2 X^2 + \beta_3 X^3 + \cdots + \beta_d X^d + \epsilon
$$

where $X^2, X^3, \ldots, X^d$ are the polynomial terms of the predictor $X$, and $d$ is the **degree** of the polynomial.

!!! info "Still a Linear Model"
    Despite the non-linear relationship between $X$ and $Y$, polynomial regression is still a **linear** model in the parameters $\beta_0, \beta_1, \ldots, \beta_d$. This means that ordinary least squares (OLS) can be applied directly by treating $X, X^2, \ldots, X^d$ as separate predictor variables.

The degree of the polynomial determines the flexibility of the curve:

| Degree | Name | Shape |
|---|---|---|
| 1 | Linear | Straight line |
| 2 | Quadratic | Parabola (one bend) |
| 3 | Cubic | S-curve (two bends) |
| $d$ | Degree-$d$ | Up to $d - 1$ bends |

---

## Interpreting Polynomial Terms

The sign and magnitude of the coefficients indicate the direction and strength of the curvature:

- **$\beta_2 > 0$** (quadratic term): The curve is **U-shaped** (convex). $Y$ decreases and then increases as $X$ grows.
- **$\beta_2 < 0$** (quadratic term): The curve is **inverted-U-shaped** (concave). $Y$ increases and then decreases.
- **Higher-order terms** ($X^3$, $X^4$, etc.) capture more complex patterns with additional inflection points.

!!! warning "Overfitting Risk"
    Higher-order polynomial terms increase model flexibility but also increase the risk of **overfitting**, especially with limited data. A high-degree polynomial may fit the training data closely but generalize poorly to new observations. Model selection criteria such as AIC and BIC should be used to choose the appropriate polynomial degree.

---

## Example: Age and Income

In a study on the relationship between age and income, a linear model might miss the reality that income increases with age until a certain point, after which it plateaus or decreases (e.g., after retirement). A quadratic model captures this:

$$
\text{Income} = \beta_0 + \beta_1 \cdot \text{Age} + \beta_2 \cdot \text{Age}^2 + \epsilon
$$

If $\beta_1 > 0$ and $\beta_2 < 0$, the model describes income that rises with age, reaches a peak, and then declines—an inverted-U relationship.

---

## Combining Polynomial and Interaction Terms

In some cases, both interaction terms and polynomial regression can be combined to capture more complex relationships. For instance, an interaction term can be included between a linear predictor and a polynomial term:

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1^2 + \beta_4 (X_1 \times X_2) + \epsilon
$$

This models a situation where the non-linear relationship between $X_1$ and $Y$ depends on the level of $X_2$.

**Example (Agriculture):** The relationship between fertilizer use ($X_1$) and crop yield ($Y$) might be quadratic (increasing returns then diminishing returns), but this relationship could also depend on rainfall levels ($X_2$). Using both polynomial and interaction terms captures this more accurately.

---

## Assessing Model Fit

When adding polynomial terms, it is important to assess whether the additional complexity genuinely improves the model:

- **Adjusted $R^2$** penalizes for the number of terms and is useful for comparing models of different polynomial degrees.
- **AIC and BIC** provide a principled trade-off between fit and complexity. BIC's stronger penalty is especially useful for preventing overfitting with high-degree polynomials.
- **F-tests** can compare nested models (e.g., a quadratic model versus a linear model) to test whether the additional polynomial terms are statistically significant.

---

## Summary

Polynomial regression extends the linear framework to model curved relationships between predictors and the response. While powerful, higher-degree polynomials carry an increased risk of overfitting. Careful use of model selection criteria ensures that polynomial terms add genuine predictive value rather than fitting noise.
