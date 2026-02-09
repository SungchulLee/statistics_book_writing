# Ridge Regression (L2 Regularization)

## Introduction

Ridge Regression, also known as Tikhonov regularization, is a technique used to address the problem of **multicollinearity** in linear regression models. When independent variables are highly correlated, the ordinary least squares (OLS) estimates can become unstable, leading to large variances and potentially poor predictions. Ridge Regression introduces a penalty term to the regression model, which shrinks the coefficients towards zero, thereby reducing the model's complexity and variance, and improving prediction accuracy.

## Ridge Regression Objective

The Ridge Regression model modifies the linear regression cost function by adding a regularization term equal to the squared magnitude of the coefficients. This addition controls the size of the coefficients and, consequently, the flexibility of the model.

Given a dataset $\{(x_i, y_i)\}_{i=1}^n$ where $x_i \in \mathbb{R}^p$ is a vector of predictors and $y_i \in \mathbb{R}$ is the response variable, the Ridge Regression objective function is:

$$
J(\mathbf{w}) = \sum_{i=1}^n \left( y_i - \mathbf{w}^T \mathbf{x}_i \right)^2 + \lambda \sum_{j=1}^p w_j^2
$$

where:

- $\mathbf{w}$ is the vector of regression coefficients.
- $\lambda \geq 0$ is a regularization parameter controlling the strength of the penalty.
- The first term is the **sum of squared errors (SSE)**, the standard measure of model fit in OLS regression.
- The second term, $\lambda \sum_{j=1}^p w_j^2$, is the **L2 regularization term** that penalizes large coefficients.

When $\lambda = 0$, Ridge Regression reduces to ordinary least squares. As $\lambda$ increases, the magnitude of the coefficients is constrained more, leading to a more regularized model.

## Understanding the Regularization Term

The L2 regularization term limits the size of the coefficients by adding a penalty for large values. This shrinks coefficients towards zero but **never makes them exactly zero** (unlike Lasso Regression, which can produce sparse models). By shrinking the coefficients, Ridge Regression can reduce the model's variance without substantially increasing bias, which is particularly beneficial in the presence of multicollinearity.

The parameter $\lambda$ determines the balance between fitting the data well (low SSE) and keeping the coefficients small:

- **Large $\lambda$**: The model becomes more biased and simple, with coefficients close to zero, potentially underfitting the data.
- **Small $\lambda$**: The model resembles the OLS solution, which may overfit the data if multicollinearity is present.

## Solving the Ridge Regression Problem

Ridge Regression can be solved analytically by modifying the normal equations used in linear regression. The normal equation for OLS regression is:

$$
\mathbf{w}_{\text{OLS}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

where $\mathbf{X}$ is the design matrix of predictors and $\mathbf{y}$ is the vector of responses.

In Ridge Regression, the solution is obtained by solving the modified normal equations:

$$
\mathbf{w}_{\text{Ridge}} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}
$$

Here $\mathbf{I}$ is the identity matrix of appropriate dimensions. The addition of $\lambda \mathbf{I}$ makes the matrix $\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}$ **non-singular**, ensuring that the inverse exists even when $\mathbf{X}^T \mathbf{X}$ is nearly singular due to multicollinearity.

## Choosing the Regularization Parameter $\lambda$

The effectiveness of Ridge Regression depends on the appropriate choice of $\lambda$:

- Too small a value may fail to address multicollinearity.
- Too large a value may lead to underfitting.

The optimal value of $\lambda$ is typically chosen using **cross-validation**, where the model is trained on different subsets of the data and tested on the remaining data to evaluate its performance. The value of $\lambda$ that minimizes the cross-validation error is selected as the best regularization parameter.

## Advantages and Limitations

### Advantages

1. **Reduces multicollinearity**: Ridge Regression addresses multicollinearity by shrinking the coefficients, leading to more stable estimates.
2. **Better prediction accuracy**: By penalizing large coefficients, Ridge Regression can produce models with lower variance and better generalization to new data.
3. **Continuous shrinkage**: Unlike Lasso Regression, which can set some coefficients to exactly zero, Ridge Regression shrinks coefficients continuously, which can be beneficial when all predictors are believed to be relevant.

### Limitations

1. **No feature selection**: Ridge Regression does not perform feature selection, as all coefficients are shrunk towards zero but not exactly to zero. This can result in models that are harder to interpret when there are many predictors.
2. **Bias–variance tradeoff**: While Ridge Regression reduces variance, it introduces bias into the model. Finding the right balance is crucial to avoid underfitting or overfitting.

## Summary

Ridge Regression is a powerful regularization technique that mitigates the effects of multicollinearity in linear regression models by penalizing large coefficients. By introducing a regularization parameter $\lambda$, Ridge Regression controls the complexity of the model, leading to improved prediction accuracy and more stable estimates. The careful selection of $\lambda$ through cross-validation is essential for achieving optimal results.
