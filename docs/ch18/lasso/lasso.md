# Lasso Regression (L1 Regularization)

## Introduction

Lasso Regression (Least Absolute Shrinkage and Selection Operator) is a regularization technique used in linear models. Like Ridge Regression, it addresses multicollinearity and helps prevent overfitting by adding a penalty to the loss function. However, Lasso has a distinct characteristic: it can shrink some coefficients to **exactly zero**, effectively performing variable selection and producing sparse models. This makes Lasso particularly useful when dealing with high-dimensional datasets where many predictors may be irrelevant.

## Lasso Regression Objective

The Lasso Regression model modifies the linear regression cost function by adding a regularization term equal to the sum of the absolute values of the coefficients:

$$
J(\mathbf{w}) = \sum_{i=1}^n \left( y_i - \mathbf{w}^T \mathbf{x}_i \right)^2 + \lambda \sum_{j=1}^p |w_j|
$$

where:

- $\mathbf{w}$ is the vector of regression coefficients.
- $\lambda \geq 0$ is the regularization parameter controlling the strength of the penalty.
- The first term, $\sum_{i=1}^n \left( y_i - \mathbf{w}^T \mathbf{x}_i \right)^2$, is the **sum of squared errors (SSE)**.
- The second term, $\lambda \sum_{j=1}^p |w_j|$, is the **L1 regularization term**, which penalizes the absolute magnitude of the coefficients.

The key difference between Lasso and Ridge Regression lies in the regularization term. While Ridge uses the **L2 norm** (sum of squared coefficients), Lasso uses the **L1 norm** (sum of absolute coefficients). This seemingly simple change has profound implications for the behavior and outcome of the regression model.

## L1 Regularization and Sparsity

The L1 regularization term encourages **sparsity** in the coefficient estimates. As $\lambda$ increases, Lasso tends to shrink some coefficients exactly to zero, effectively excluding the corresponding predictors from the model. This feature selection property is particularly advantageous in high-dimensional datasets, where many predictors may be irrelevant or redundant.

The sparsity induced by Lasso can be understood geometrically. The constraint region for Lasso, defined by the L1 norm, is a **diamond-shaped region** (in contrast to the circular constraint region of Ridge, defined by the L2 norm). The sharp corners of this diamond intersect with the contours of the objective function at points where some coefficients are exactly zero, leading to sparse solutions.

## Solving the Lasso Problem

Solving the Lasso regression problem is more complex than Ridge due to the **non-differentiable** nature of the L1 norm at zero. Efficient algorithms have been developed to handle this:

- **Coordinate descent**: Updates one coefficient at a time while keeping the others fixed, repeating until convergence.
- **Least Angle Regression (LARS)**: An efficient algorithm that computes the entire regularization path.

In contrast to Ridge, there is **no closed-form solution** for Lasso. Iterative methods are employed to minimize the objective function.

## Choosing the Regularization Parameter $\lambda$

The value of $\lambda$ controls the tradeoff between fitting the model to the data and shrinking the coefficients towards zero:

- A **small $\lambda$** results in a model similar to ordinary least squares, with little or no regularization.
- A **large $\lambda$** produces a sparse model with many coefficients shrunk to zero.

**Cross-validation** is typically used to select the optimal $\lambda$. The data is split into training and validation sets multiple times, and the model's performance is evaluated for different values of $\lambda$. The $\lambda$ that minimizes the cross-validation error is chosen as the optimal regularization parameter.

## Advantages and Limitations

### Advantages

1. **Feature selection**: Lasso's ability to shrink coefficients to zero makes it a powerful tool for feature selection, especially in high-dimensional datasets with many irrelevant or redundant predictors.
2. **Interpretability**: By producing sparse models, Lasso leads to more interpretable models where only the most relevant predictors are included.
3. **Handling high-dimensional data**: Lasso is particularly useful when the number of predictors exceeds the number of observations, as it can effectively reduce the dimensionality of the problem.

### Limitations

1. **Bias**: Lasso introduces bias by shrinking coefficients, which can lead to underfitting if the true model is not sparse.
2. **Selection consistency**: Lasso may not consistently select the correct predictors, especially when predictors are highly correlated, leading to instability in the selected features.
3. **Struggles with grouped variables**: When predictors are highly correlated, Lasso tends to select only one variable from a group of correlated variables, ignoring the others, which can lead to suboptimal model performance.

## Connection to Elastic Net

One limitation of Lasso is its tendency to select a single variable from a group of highly correlated variables. To address this, the **Elastic Net** regularization technique combines both L1 and L2 penalties:

$$
J(\mathbf{w}) = \sum_{i=1}^n \left( y_i - \mathbf{w}^T \mathbf{x}_i \right)^2 + \lambda_1 \sum_{j=1}^p |w_j| + \lambda_2 \sum_{j=1}^p w_j^2
$$

Here $\lambda_1$ controls the L1 regularization (Lasso) and $\lambda_2$ controls the L2 regularization (Ridge). By tuning these parameters, Elastic Net achieves a balance between feature selection and coefficient shrinkage, making it a more robust regularization method in the presence of correlated predictors.

## Summary

Lasso Regression is a powerful tool for linear modeling, especially when dealing with high-dimensional data where feature selection is necessary. By adding an L1 penalty to the cost function, Lasso not only prevents overfitting but also produces sparse, interpretable models by excluding irrelevant predictors. However, the choice of $\lambda$ is critical, and the method can struggle with correlated predictors. Elastic Net provides a valuable extension by combining Lasso with Ridge Regression.
