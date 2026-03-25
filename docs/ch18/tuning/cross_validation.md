# Cross-Validation for Lambda Selection

The regularization parameter $\lambda$ controls the tradeoff between fitting the data and constraining the model. Training error always decreases as $\lambda$ decreases (less regularization), so it cannot be used to select $\lambda$. We need an estimate of test error, and cross-validation (CV) provides the standard approach. This section covers the K-fold CV procedure for regularized regression, the one-standard-error rule, and practical implementation details.

## Why Cross-Validation Is Needed

For a fixed $\lambda$, the training error of a regularized model is:

$$
\text{Err}_{\text{train}}(\lambda) = \frac{1}{n}\|\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}(\lambda)\|^2
$$

This quantity is a biased estimate of the true test error because the same data are used for both fitting and evaluation. As $\lambda$ decreases, the model becomes more flexible, training error decreases, but test error may increase due to overfitting. Cross-validation provides a nearly unbiased estimate of test error by holding out data for evaluation.

## The K-Fold CV Procedure

**Setup.** Choose a grid of candidate $\lambda$ values $\lambda_1 > \lambda_2 > \cdots > \lambda_M$ and a number of folds $K$ (typically $K = 5$ or $K = 10$).

**Algorithm.**

1. Randomly partition the $n$ observations into $K$ folds $F_1, F_2, \ldots, F_K$ of approximately equal size.
2. For each fold $k = 1, \ldots, K$:
    - Let $\mathbf{X}^{(-k)}$ and $\mathbf{y}^{(-k)}$ denote the training data (all observations except fold $k$).
    - For each $\lambda_m$:
        - Fit the regularized model on the training data to obtain $\hat{\boldsymbol{\beta}}^{(-k)}(\lambda_m)$.
        - Predict the held-out fold: $\hat{\mathbf{y}}^{(k)} = \mathbf{X}^{(k)}\hat{\boldsymbol{\beta}}^{(-k)}(\lambda_m)$.
        - Compute the fold error: $\text{MSE}_k(\lambda_m) = \frac{1}{|F_k|}\sum_{i \in F_k}(y_i - \hat{y}_i)^2$.
3. Average across folds:

$$
\text{CV}(\lambda_m) = \frac{1}{K}\sum_{k=1}^K \text{MSE}_k(\lambda_m)
$$

4. Compute the standard error of the CV estimate:

$$
\text{SE}(\lambda_m) = \sqrt{\frac{1}{K}\cdot\frac{1}{K-1}\sum_{k=1}^K \bigl(\text{MSE}_k(\lambda_m) - \text{CV}(\lambda_m)\bigr)^2}
$$

## Selecting Lambda-min

The simplest selection rule chooses the $\lambda$ that minimizes the CV error:

$$
\lambda_{\min} = \arg\min_{\lambda_m} \text{CV}(\lambda_m)
$$

This produces the model with the best estimated prediction accuracy. The CV curve is typically U-shaped: high error at large $\lambda$ (underfitting), decreasing to a minimum, then increasing at small $\lambda$ (overfitting).

## The One-Standard-Error Rule

The CV estimates are noisy, and $\lambda_{\min}$ may correspond to a more complex model than necessary. The **one-standard-error rule** (1-SE rule) selects a more regularized model:

$$
\lambda_{1\text{SE}} = \max\bigl\{\lambda_m : \text{CV}(\lambda_m) \leq \text{CV}(\lambda_{\min}) + \text{SE}(\lambda_{\min})\bigr\}
$$

This is the largest (most regularized) $\lambda$ whose CV error is within one standard error of the minimum. The rationale is that models with CV error within one standard error of the best are statistically indistinguishable, so we prefer the simplest (most regularized) among them.

!!! tip "When to Use Which Rule"
    Use $\lambda_{\min}$ when prediction accuracy is the primary goal. Use $\lambda_{1\text{SE}}$ when interpretability, stability, or parsimony is valued. In scientific applications where understanding which predictors matter is as important as prediction, $\lambda_{1\text{SE}}$ is often preferred.

## Grid Construction

The grid of $\lambda$ values should span the range from $\lambda_{\max}$ (where all coefficients are zero for lasso) down to a small fraction of $\lambda_{\max}$.

For lasso and elastic net:

$$
\lambda_{\max} = \frac{1}{\alpha n}\|\mathbf{X}^\top\mathbf{y}\|_\infty
$$

The grid is typically logarithmic:

$$
\lambda_m = \lambda_{\max} \cdot 10^{-m \cdot r / M}, \quad m = 0, 1, \ldots, M
$$

where $M$ is the number of grid points (commonly 100) and $r$ controls the range (commonly $r = 3$ or $r = 4$, so the smallest $\lambda$ is $10^{-3}$ or $10^{-4}$ times $\lambda_{\max}$).

For ridge regression, $\lambda_{\max}$ is not defined by a sparsity threshold, but a similar logarithmic grid from a large value to a small value works well.

## Implementation Details

### Standardization Within Folds

Predictors should be standardized (mean zero, unit variance) before fitting. Crucially, the standardization must be computed on the training folds only and then applied to the held-out fold. Using the full dataset for standardization introduces subtle data leakage that biases the CV error estimate downward.

### Warm Starts Along the Path

For lasso and elastic net, the regularization path is computed for the full $\lambda$ grid using coordinate descent with warm starts. Each fold-specific model fit benefits from the solution at the previous $\lambda$ value, making the full CV procedure efficient.

### Fold Assignment

For time-series data, use contiguous blocks rather than random fold assignment to respect temporal ordering. For grouped data (e.g., repeated measurements on the same subject), assign all observations from the same group to the same fold.

!!! warning "Common Pitfall: Data Leakage"
    Do not perform feature selection, variable transformation, or any data-dependent preprocessing on the full dataset before creating folds. All preprocessing steps must be performed within each fold's training set to avoid optimistically biased CV estimates.

## Choice of K

| $K$ | Bias | Variance | Computation |
|---|---|---|---|
| $K = 5$ | Moderate (pessimistic) | Low | Fast |
| $K = 10$ | Lower | Moderate | Moderate |
| $K = n$ (LOOCV) | Nearly unbiased | Can be high | Expensive (unless shortcut exists) |

The most common choices are $K = 5$ and $K = 10$. LOOCV has a closed-form shortcut for ridge regression (using the hat matrix diagonal) but not for lasso.

## Two-Dimensional CV for Elastic Net

The elastic net has two hyperparameters: $\lambda$ and $\alpha$. A common approach:

1. Fix a grid of $\alpha$ values (e.g., $\alpha \in \{0.1, 0.25, 0.5, 0.75, 0.9, 1.0\}$).
2. For each $\alpha$, perform $K$-fold CV over the $\lambda$ grid.
3. Select the $(\lambda, \alpha)$ pair with the lowest CV error.

Alternatively, fix $\alpha$ based on domain knowledge and optimize only over $\lambda$.

## Summary

Cross-validation estimates test error by rotating through held-out folds, providing a data-driven method for selecting the regularization parameter $\lambda$. The minimum CV error identifies $\lambda_{\min}$; the one-standard-error rule identifies the more parsimonious $\lambda_{1\text{SE}}$. Practical implementation requires a logarithmic $\lambda$ grid starting from $\lambda_{\max}$, standardization within folds to avoid data leakage, and warm starts for computational efficiency. For the elastic net, either a two-dimensional grid search over $(\lambda, \alpha)$ or a fixed $\alpha$ with one-dimensional $\lambda$ optimization provides effective tuning.
