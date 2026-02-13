# Exercises: Regularization Techniques

## Conceptual

**Exercise 1.** Consider the Ridge regression estimator $\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$.

(a) Show that as $\lambda \to 0$, $\hat{\boldsymbol{\beta}}_{\text{ridge}} \to \hat{\boldsymbol{\beta}}_{\text{OLS}}$.

(b) Show that as $\lambda \to \infty$, $\hat{\boldsymbol{\beta}}_{\text{ridge}} \to \mathbf{0}$.

(c) Prove that $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$ is positive definite for all $\lambda > 0$, even when $\mathbf{X}^\top\mathbf{X}$ is singular.

**Exercise 2.** For orthonormal design ($\mathbf{X}^\top\mathbf{X} = n\mathbf{I}$), derive the closed-form solutions for Ridge and Lasso. Explain geometrically why Lasso produces exact zeros but Ridge does not.

**Exercise 3.** Explain the "grouping effect" of Elastic Net. Why does Lasso fail with highly correlated predictors? Give an example where $X_1 = X_2 + \varepsilon$ (with small $\varepsilon$) and describe what Lasso, Ridge, and Elastic Net would do.

**Exercise 4.** The effective degrees of freedom for Ridge regression is $\text{df}(\lambda) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}$ where $d_j$ are the singular values of $\mathbf{X}$.

(a) Show that $\text{df}(0) = p$ and $\text{df}(\infty) = 0$.

(b) Is $\text{df}(\lambda)$ monotonically decreasing in $\lambda$?

(c) How would you define an analogous quantity for Lasso?

## Computation

**Exercise 5.** Generate $n = 100$ observations from the model $y = 3x_1 - 2x_2 + 0.5x_3 + \varepsilon$ where $\varepsilon \sim N(0, 1)$, along with 17 noise predictors ($x_4, \ldots, x_{20}$).

(a) Fit OLS, Ridge, Lasso, and Elastic Net. Compare the coefficient estimates.

(b) Which methods correctly identify the 3 true predictors?

(c) Use 5-fold CV to select the best $\lambda$ for each method. Report test MSE.

**Exercise 6.** Create a dataset where predictors come in correlated groups: $X_1, X_2, X_3$ are highly correlated ($\rho = 0.95$) and all have nonzero effects. $X_4, X_5, X_6$ are another correlated group with nonzero effects. The remaining 14 predictors are noise.

(a) Fit Lasso. Does it select one predictor per group or multiple?

(b) Fit Elastic Net with $\alpha = 0.5$. Compare the selected features.

(c) Plot the coefficient paths for both methods.

**Exercise 7.** Implement Ridge regression **from scratch** (without sklearn):

(a) Write a function that takes $\mathbf{X}, \mathbf{y}, \lambda$ and returns $\hat{\boldsymbol{\beta}}_{\text{ridge}}$ using the closed-form formula.

(b) Implement leave-one-out CV using the hat matrix shortcut.

(c) Verify your implementation matches `sklearn.linear_model.Ridge`.

**Exercise 8.** Implement the coordinate descent algorithm for Lasso:

(a) Write the soft-thresholding operator $S_\lambda(z)$.

(b) Implement the full coordinate descent loop with convergence check.

(c) Compare your solution paths with `sklearn.linear_model.Lasso`.

## Applied

**Exercise 9 (Finance).** Consider predicting monthly stock returns using Fama-French factors and macroeconomic variables. Simulate a dataset with $p = 50$ potential predictors and $n = 120$ monthly observations.

(a) Why is regularization essential in this setting?

(b) Compare out-of-sample $R^2$ for OLS, Ridge, Lasso, and Elastic Net using rolling-window cross-validation (train on 60 months, predict next month, roll forward).

(c) Which predictors does Lasso select? Are they stable across rolling windows?

**Exercise 10.** Using the Boston Housing or California Housing dataset:

(a) Fit all three regularized models and OLS. Compare test MSE.

(b) Plot the regularization paths. At what $\lambda$ values do coefficients become zero for Lasso?

(c) Use the one-standard-error rule for model selection. How does the selected model compare to the minimum-CV-error model?
