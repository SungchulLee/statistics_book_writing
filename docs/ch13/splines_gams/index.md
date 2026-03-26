# Splines and Generalized Additive Models


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

## Overview

In [polynomial regression](../interaction_polynomial/polynomial_regression.md), a single polynomial of degree $d$ is fit to the entire range of a predictor. This global approach has a fundamental limitation: increasing the degree to capture local curvature in one region can introduce unwanted oscillations elsewhere (Runge's phenomenon). Splines overcome this by fitting low-degree polynomials within local intervals and joining them smoothly at boundaries called **knots**. Generalized Additive Models (GAMs) extend this idea to multiple predictors, modeling the response as an additive combination of smooth functions -- one for each predictor -- without specifying the functional form in advance.

This section covers the mathematical foundations and practical usage of these flexible non-linear regression methods. The key idea throughout is the trade-off between fit and smoothness: too few knots or too much penalization yields underfitting, while too many knots or too little penalization yields overfitting.

## Contents

This section contains the following pages:

- [**Generalized Additive Models**](generalized_additive_models.md) -- Flexible semi-parametric regression that models the response as a sum of smooth functions, one per predictor, estimated using penalized likelihood

## Key Concepts

### Basis Functions

A spline of degree $d$ with $K$ knots is a linear combination of **basis functions**. The most common choice is the **B-spline basis**, which consists of local polynomial functions that are nonzero only over a small interval. A regression spline model takes the form:

$$
f(x) = \sum_{j=1}^{K+d+1} \beta_j B_j(x)
$$

where $B_j(x)$ are the B-spline basis functions. Because each $B_j$ is local, splines adapt to the data region by region without the global oscillation problems of high-degree polynomials.

### Smoothing Penalty

Regression splines with many knots can overfit the data. **Smoothing splines** address this by penalizing the roughness of the fitted curve. The objective function minimizes:

$$
\sum_{i=1}^{n} (Y_i - f(x_i))^2 + \lambda \int [f''(x)]^2 \, dx
$$

The smoothing parameter $\lambda \geq 0$ controls the trade-off: $\lambda = 0$ interpolates the data, while $\lambda \to \infty$ forces $f$ toward a straight line. In practice, $\lambda$ is selected by cross-validation or generalized cross-validation (GCV).

### Effective Degrees of Freedom

The smoothing penalty means that the model's complexity is not simply the number of basis functions. The **effective degrees of freedom** (edf) quantifies the actual flexibility used:

$$
\text{edf} = \operatorname{tr}(\mathbf{S})
$$

where $\mathbf{S}$ is the smoother (hat) matrix satisfying $\hat{\mathbf{f}} = \mathbf{S} \mathbf{Y}$. When $\lambda = 0$, $\text{edf}$ equals the number of basis functions (interpolation). When $\lambda \to \infty$, $\text{edf} \to 2$ (a straight line with intercept and slope). Intermediate values of $\lambda$ produce $\text{edf}$ between these extremes, providing a continuous measure of model complexity.

### Additivity in GAMs

A GAM models the conditional mean as:

$$
g(\mathbb{E}[Y \mid X_1, \ldots, X_p]) = \alpha + \sum_{j=1}^{p} f_j(X_j)
$$

where $g$ is a link function and each $f_j$ is a smooth function estimated from the data. The **additivity** assumption means that each predictor's contribution to the response can be visualized and interpreted independently. This is more restrictive than allowing arbitrary interactions (which would require multivariate smooth functions) but far more flexible than assuming linearity.

## Practical Applications

- **Economics**: Modeling non-linear price-demand relationships using GAMs, where the effect of price on demand may flatten at extreme values
- **Finance**: Capturing the volatility smile in options pricing with splines, where implied volatility varies non-linearly with strike price
- **Medicine**: Estimating dose-response curves with smoothing splines, where the response plateaus at high doses
- **Environmental science**: GAMs for assessing non-linear effects of multiple pollutant concentrations on health outcomes, with separate smooth functions for each pollutant

## Prerequisites

This section builds on:

- [Simple and multiple linear regression](../linear_regression/simple.md) -- the linear model that splines and GAMs generalize
- [Polynomial regression](../interaction_polynomial/polynomial_regression.md) -- the global polynomial approach that motivates the piecewise construction of splines
- [AIC and BIC](../model_selection/aic_bic.md) -- information criteria used for selecting the smoothing parameter and number of knots
