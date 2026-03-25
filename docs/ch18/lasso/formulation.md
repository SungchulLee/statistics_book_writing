# Lasso Formulation and Sparsity

Ridge regression shrinks all coefficients toward zero but never eliminates any of them. In many applications, especially when $p$ is large, we want an estimator that automatically identifies irrelevant features by setting their coefficients to exactly zero. The **Lasso** (Least Absolute Shrinkage and Selection Operator), introduced by Tibshirani (1996), achieves both regularization and variable selection by replacing the L2 penalty with an L1 penalty.

## The Lasso Objective

The lasso estimator solves the penalized least squares problem:

$$
\hat{\boldsymbol{\beta}}_{\text{lasso}} = \arg\min_{\boldsymbol{\beta}} \left\{\frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|_1\right\}
$$

where $\|\boldsymbol{\beta}\|_1 = \sum_{j=1}^p |\beta_j|$ is the L1 norm and $\lambda \geq 0$ is the regularization parameter. The factor $1/(2n)$ is a convention that makes $\lambda$ comparable across different sample sizes.

The equivalent constrained form is:

$$
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 \quad \text{subject to} \quad \sum_{j=1}^p |\beta_j| \leq t
$$

where $t$ is determined by $\lambda$ through the KKT conditions.

## Non-Differentiability and the Subdifferential

Unlike the L2 penalty $\|\boldsymbol{\beta}\|_2^2$, the L1 norm $\|\boldsymbol{\beta}\|_1$ is **not differentiable** at any point where $\beta_j = 0$. This means we cannot simply set the gradient to zero. Instead, we use the **subdifferential**, which generalizes the gradient to non-smooth convex functions.

The subdifferential of $|\beta_j|$ is:

$$
\partial|\beta_j| = \begin{cases} \{+1\} & \text{if } \beta_j > 0 \\ [-1, +1] & \text{if } \beta_j = 0 \\ \{-1\} & \text{if } \beta_j < 0 \end{cases}
$$

At $\beta_j = 0$, the subdifferential is the entire interval $[-1, +1]$. This allows the optimality condition to be satisfied with $\beta_j = 0$ for a range of data configurations, which is precisely what produces sparse solutions.

## The Soft-Thresholding Operator

For the special case of **orthonormal design** ($\mathbf{X}^\top\mathbf{X} = n\mathbf{I}$), the lasso has a closed-form solution expressed through the soft-thresholding operator:

$$
\hat{\beta}_j^{\text{lasso}} = S_\lambda(\hat{\beta}_j^{\text{OLS}}) = \text{sign}(\hat{\beta}_j^{\text{OLS}})\,\max\bigl(|\hat{\beta}_j^{\text{OLS}}| - \lambda,\; 0\bigr)
$$

This operator acts on each OLS coefficient independently:

- If $|\hat{\beta}_j^{\text{OLS}}| \leq \lambda$, the coefficient is set to exactly zero.
- If $|\hat{\beta}_j^{\text{OLS}}| > \lambda$, the coefficient is shrunk by $\lambda$ toward zero.

!!! note "Soft vs Hard Thresholding"
    Soft-thresholding (lasso) both zeros out small coefficients and shrinks large ones. Hard-thresholding (best subset selection) zeros out small coefficients but leaves large ones unchanged. Ridge applies proportional shrinkage $\hat{\beta}_j^{\text{ridge}} = \hat{\beta}_j^{\text{OLS}}/(1 + \lambda)$ without any thresholding.

## No Closed-Form Solution in General

For non-orthonormal designs ($\mathbf{X}^\top\mathbf{X} \neq n\mathbf{I}$), the lasso does not have a closed-form solution. The L1 penalty couples the coefficients through the cross-terms in $\mathbf{X}^\top\mathbf{X}$, making the problem a convex but non-smooth optimization. Numerical algorithms, particularly coordinate descent, are required and are discussed in a subsequent section.

## The Sparsity Property

The defining feature of the lasso is its ability to produce **sparse** solutions: for sufficiently large $\lambda$, many components of $\hat{\boldsymbol{\beta}}_{\text{lasso}}$ are exactly zero. The number of nonzero coefficients depends on $\lambda$:

- At $\lambda = 0$, the solution is OLS (assuming $p < n$), with all coefficients typically nonzero.
- At $\lambda_{\max} = \frac{1}{n}\|\mathbf{X}^\top\mathbf{y}\|_\infty = \max_j \left|\frac{1}{n}\sum_{i=1}^n x_{ij}y_i\right|$, all coefficients are zero.
- Between these extremes, increasing $\lambda$ progressively zeros out coefficients.

!!! warning "Lasso Limitation with p Greater Than n"
    When $p > n$, the lasso can select at most $n$ nonzero coefficients. This is because the lasso solution lies in an at-most $n$-dimensional subspace. If more than $n$ features are truly relevant, the lasso cannot recover all of them simultaneously.

## Comparison of Shrinkage Operators

The fundamental difference between ridge and lasso is captured by their shrinkage behavior in the orthonormal case:

| Method | Shrinkage rule | Exact zeros | Type |
|---|---|---|---|
| OLS | $\hat{\beta}_j^{\text{OLS}}$ | No | No shrinkage |
| Ridge | $\hat{\beta}_j^{\text{OLS}} / (1 + \lambda)$ | No | Proportional shrinkage |
| Lasso | $\text{sign}(\hat{\beta}_j^{\text{OLS}})\max(\lvert\hat{\beta}_j^{\text{OLS}}\rvert - \lambda, 0)$ | Yes | Translational shrinkage |
| Best subset | $\hat{\beta}_j^{\text{OLS}} \cdot \mathbf{1}(\lvert\hat{\beta}_j^{\text{OLS}}\rvert > \lambda)$ | Yes | Hard thresholding |

Ridge applies uniform proportional shrinkage. Lasso applies uniform translational shrinkage with a hard cutoff at $\lambda$. Best subset selection applies a hard threshold but leaves retained coefficients unshrunken.

## Summary

The lasso replaces the L2 penalty of ridge regression with an L1 penalty $\lambda\|\boldsymbol{\beta}\|_1$. The non-differentiability of the absolute value function at zero is the mathematical mechanism that produces sparse solutions: the subdifferential allows the optimality condition to be satisfied with $\beta_j = 0$ for a range of configurations. In the orthonormal case, the lasso solution is given by the soft-thresholding operator. For general designs, no closed-form solution exists, and iterative algorithms are needed. The sparsity property makes lasso simultaneously a regularization method and a feature selection method.
