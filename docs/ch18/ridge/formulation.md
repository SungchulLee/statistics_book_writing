# Ridge Formulation and Closed-Form Solution

OLS minimizes the residual sum of squares without any constraint on the size of the coefficients. When predictors are correlated or the number of predictors is large relative to the sample size, this freedom allows OLS coefficients to grow excessively, inflating variance. Ridge regression, introduced by Hoerl and Kennard (1970), addresses this by adding a quadratic penalty on the coefficient vector, producing an estimator that always has a unique, closed-form solution.

## The Ridge Objective

Ridge regression solves the penalized least squares problem:

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = \arg\min_{\boldsymbol{\beta}} \left\{\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|_2^2\right\}
$$

where $\|\boldsymbol{\beta}\|_2^2 = \sum_{j=1}^p \beta_j^2$ is the squared L2 norm and $\lambda \geq 0$ is the regularization parameter. The first term measures fit to the data; the second penalizes large coefficients.

!!! note "The Intercept Is Not Penalized"
    In practice, the response $\mathbf{y}$ is centered and the predictors are standardized to have mean zero and unit variance. The intercept $\beta_0$ is estimated separately as $\bar{y}$ and is not included in the penalty. This ensures the penalty treats all slope coefficients equally regardless of the original measurement scales.

## Derivation of the Closed-Form Solution

The ridge objective $L(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|^2$ is a convex quadratic function. Taking the gradient and setting it to zero:

$$
\nabla_{\boldsymbol{\beta}} L = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + 2\lambda\boldsymbol{\beta} = \mathbf{0}
$$

Rearranging:

$$
\mathbf{X}^\top\mathbf{X}\boldsymbol{\beta} + \lambda\boldsymbol{\beta} = \mathbf{X}^\top\mathbf{y}
$$

$$
(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})\boldsymbol{\beta} = \mathbf{X}^\top\mathbf{y}
$$

Since $\mathbf{X}^\top\mathbf{X}$ is positive semi-definite and $\lambda\mathbf{I}$ is positive definite for $\lambda > 0$, their sum $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$ is positive definite and therefore invertible. The unique solution is:

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}
$$

This solution exists and is unique for every $\lambda > 0$, even when $\mathbf{X}^\top\mathbf{X}$ is singular (i.e., when $p > n$ or exact collinearity is present).

## Constrained Optimization Form

The penalized form is equivalent to the constrained optimization problem:

$$
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 \quad \text{subject to} \quad \|\boldsymbol{\beta}\|_2^2 \leq t
$$

where $t = t(\lambda)$ is a budget on the total squared magnitude of the coefficients. By the KKT conditions, there is a one-to-one correspondence between $\lambda$ and $t$: increasing $\lambda$ decreases $t$, tightening the constraint.

## SVD Interpretation

The singular value decomposition $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$ provides insight into how ridge regression modifies OLS. The OLS solution can be written as:

$$
\hat{\boldsymbol{\beta}}_{\text{OLS}} = \sum_{j=1}^p \frac{\mathbf{u}_j^\top\mathbf{y}}{d_j}\,\mathbf{v}_j
$$

The ridge solution replaces each term with a shrunk version:

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}\cdot\frac{\mathbf{u}_j^\top\mathbf{y}}{d_j}\,\mathbf{v}_j
$$

The factor $d_j^2/(d_j^2 + \lambda)$ is the **shrinkage factor** for the $j$-th principal component. It lies between 0 and 1, and components with small singular values (the directions most affected by collinearity) are shrunk the most. This selective shrinkage is the mechanism by which ridge regression reduces variance along the most unstable directions.

## Bias and Variance

The ridge estimator is biased. Its bias and covariance are:

$$
\text{Bias}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = E[\hat{\boldsymbol{\beta}}_{\text{ridge}}] - \boldsymbol{\beta} = -\lambda(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\boldsymbol{\beta}
$$

$$
\text{Cov}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = \sigma^2(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}
$$

As $\lambda$ increases, the bias increases while the variance decreases. The key result of Hoerl and Kennard (1970) is that there always exists a $\lambda > 0$ for which the total MSE of the ridge estimator is strictly less than that of OLS.

## Effective Degrees of Freedom

OLS uses $p$ degrees of freedom. The ridge estimator uses fewer, captured by:

$$
\text{df}(\lambda) = \text{tr}\bigl[\mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\bigr] = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}
$$

This quantity decreases monotonically from $p$ (when $\lambda = 0$) to $0$ (as $\lambda \to \infty$), providing a continuous measure of model complexity that enables comparison across different values of $\lambda$.

!!! tip "Connecting to Information Criteria"
    The effective degrees of freedom allow the use of AIC and BIC for ridge regression, replacing the usual count of parameters with $\text{df}(\lambda)$. This connection is developed in the tuning section of this chapter.

## Comparison with OLS

| Property | OLS | Ridge ($\lambda > 0$) |
|---|---|---|
| Bias | Zero | Nonzero, increases with $\lambda$ |
| Variance | $\sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$ | Reduced, especially along collinear directions |
| Existence | Requires $\text{rank}(\mathbf{X}) = p$ | Always exists |
| Uniqueness | Unique iff $\mathbf{X}^\top\mathbf{X}$ invertible | Always unique |
| Degrees of freedom | $p$ | $\text{df}(\lambda) < p$ |
| Feature selection | No | No |

## Summary

Ridge regression adds the penalty $\lambda\|\boldsymbol{\beta}\|_2^2$ to the OLS objective, yielding the closed-form solution $\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$. This solution always exists and is unique for $\lambda > 0$. Through the SVD lens, ridge applies differential shrinkage that targets the most unstable coefficient directions, reducing variance at the cost of introducing bias. The effective degrees of freedom provide a continuous complexity measure that decreases with $\lambda$.

## Exercises

**Exercise 1.**
Consider the Ridge regression estimator $\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$.

(a) Show that as $\lambda \to 0$, $\hat{\boldsymbol{\beta}}_{\text{ridge}} \to \hat{\boldsymbol{\beta}}_{\text{OLS}}$.

(b) Show that as $\lambda \to \infty$, $\hat{\boldsymbol{\beta}}_{\text{ridge}} \to \mathbf{0}$.

(c) Prove that $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$ is positive definite for all $\lambda > 0$, even when $\mathbf{X}^\top\mathbf{X}$ is singular.

---

**Exercise 2.**
The effective degrees of freedom for Ridge regression is $\text{df}(\lambda) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}$ where $d_j$ are the singular values of $\mathbf{X}$.

(a) Show that $\text{df}(0) = p$ and $\text{df}(\infty) = 0$.

(b) Is $\text{df}(\lambda)$ monotonically decreasing in $\lambda$?

(c) How would you define an analogous quantity for Lasso?

---

**Exercise 3.**
Implement Ridge regression **from scratch** (without sklearn):

(a) Write a function that takes $\mathbf{X}, \mathbf{y}, \lambda$ and returns $\hat{\boldsymbol{\beta}}_{\text{ridge}}$ using the closed-form formula.

(b) Implement leave-one-out CV using the hat matrix shortcut.

(c) Verify your implementation matches `sklearn.linear_model.Ridge`.
