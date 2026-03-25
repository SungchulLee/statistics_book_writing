# Information Criteria for Regularized Models

Cross-validation provides a reliable method for selecting $\lambda$ but requires fitting the model multiple times across folds. Information criteria offer an analytical alternative: they estimate prediction error from a single model fit by adding a complexity penalty to the training error. For regularized models, the key challenge is defining an appropriate measure of model complexity, since the number of parameters is not a meaningful concept when coefficients are continuously shrunk.

## Review of AIC and BIC

For a model with log-likelihood $\ell(\hat{\boldsymbol{\beta}})$ and $d$ estimated parameters, the classical information criteria are:

$$
\text{AIC} = -2\ell(\hat{\boldsymbol{\beta}}) + 2d
$$

$$
\text{BIC} = -2\ell(\hat{\boldsymbol{\beta}}) + d\log n
$$

Under the Gaussian linear model with known $\sigma^2$, the log-likelihood is proportional to the negative RSS, so:

$$
\text{AIC} = \frac{n}{\sigma^2}\text{RSS} + 2d
$$

$$
\text{BIC} = \frac{n}{\sigma^2}\text{RSS} + d\log n
$$

AIC targets prediction error minimization (asymptotically equivalent to LOOCV). BIC targets model selection consistency (selecting the true model as $n \to \infty$). BIC applies a heavier penalty for large $n$, favoring simpler models.

## Effective Degrees of Freedom

For regularized models, the integer parameter count $d$ is replaced by the **effective degrees of freedom** $\text{df}(\lambda)$, which measures the effective complexity of the model as a continuous function of $\lambda$.

### Ridge Regression

For ridge regression, the effective degrees of freedom have a clean analytical form:

$$
\text{df}_{\text{ridge}}(\lambda) = \text{tr}\bigl[\mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\bigr] = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}
$$

where $d_1, \ldots, d_p$ are the singular values of $\mathbf{X}$. Each term $d_j^2/(d_j^2 + \lambda)$ is the shrinkage factor for the $j$-th component. As $\lambda \to 0$, $\text{df} \to p$ (OLS). As $\lambda \to \infty$, $\text{df} \to 0$.

This formula is the trace of the hat matrix $\mathbf{H}(\lambda) = \mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top$, which is the standard definition of degrees of freedom for linear smoothers.

### Lasso

For the lasso, the effective degrees of freedom are:

$$
\text{df}_{\text{lasso}}(\lambda) = |\hat{S}(\lambda)| = \text{number of nonzero coefficients}
$$

This remarkable result, proved by Zou, Hastie, and Tibshirani (2007) using Stein's unbiased risk estimate (SURE), shows that the effective degrees of freedom for the lasso equal the number of selected features. This holds under mild conditions on $\mathbf{X}$ and when the errors are Gaussian.

!!! note "Lasso df Is Discontinuous"
    Unlike ridge, where $\text{df}(\lambda)$ varies continuously, the lasso's degrees of freedom jump by 1 each time a coefficient enters or leaves the active set. This discontinuity reflects the discrete nature of variable selection.

### Elastic Net

For the elastic net, the effective degrees of freedom do not have as simple a formula. An approximation uses the number of nonzero coefficients adjusted for the L2 penalty:

$$
\text{df}_{\text{EN}}(\lambda, \alpha) \approx \text{tr}\bigl[\mathbf{X}_{\hat{S}}(\mathbf{X}_{\hat{S}}^\top\mathbf{X}_{\hat{S}} + \lambda(1-\alpha)\mathbf{I})^{-1}\mathbf{X}_{\hat{S}}^\top\bigr]
$$

where $\mathbf{X}_{\hat{S}}$ is the submatrix of $\mathbf{X}$ corresponding to the selected features.

## AIC and BIC for Regularized Models

Replacing the integer $d$ with $\text{df}(\lambda)$:

$$
\text{AIC}(\lambda) = \frac{n}{\hat{\sigma}^2}\text{RSS}(\lambda) + 2\,\text{df}(\lambda)
$$

$$
\text{BIC}(\lambda) = \frac{n}{\hat{\sigma}^2}\text{RSS}(\lambda) + \text{df}(\lambda)\log n
$$

where $\hat{\sigma}^2$ is an estimate of the error variance (often from the full OLS model if $p < n$, or from a low-dimensional model).

The optimal $\lambda$ is chosen by minimizing AIC or BIC over the $\lambda$ grid.

!!! warning "Estimating Sigma-Squared"
    The information criteria require an estimate of $\sigma^2$. If $p > n$, the OLS residual variance is not available. Options include using a low-dimensional model, the scaled lasso, or cross-validation to estimate $\sigma^2$ separately.

## Mallows' Cp

A closely related criterion is Mallows' $C_p$, which for ridge regression takes the form:

$$
C_p(\lambda) = \frac{\text{RSS}(\lambda)}{\hat{\sigma}^2} - n + 2\,\text{df}(\lambda)
$$

Under Gaussian errors, $C_p$ is equivalent to AIC (up to an additive constant). Minimizing $C_p$ over $\lambda$ gives the same selection as minimizing AIC.

## Comparison: Information Criteria versus Cross-Validation

| Criterion | Computation | Theory | Practical use |
|---|---|---|---|
| AIC | Single fit per $\lambda$ | Asymptotically optimal for prediction | Good for prediction-focused selection |
| BIC | Single fit per $\lambda$ | Consistent for model selection | Good when true model is sparse |
| K-fold CV | $K$ fits per $\lambda$ | Distribution-free, finite-sample | Most robust, standard choice |
| GCV | Single fit per $\lambda$ | Asymptotically equivalent to LOOCV | Efficient for ridge |

Information criteria are faster (one model fit per $\lambda$) but rely on the Gaussian assumption and require an estimate of $\sigma^2$. Cross-validation is more robust but computationally heavier. In practice, CV is the default choice, with information criteria used when computational resources are limited or as a sanity check.

## Practical Recommendations

1. **Use CV as the primary method** for selecting $\lambda$ unless computational constraints are severe.

2. **Use AIC** when the goal is prediction and the Gaussian assumption is reasonable.

3. **Use BIC** when the goal is identifying the true sparse model (BIC penalizes complexity more heavily than AIC for $n > 8$).

4. **For ridge regression**, GCV provides an efficient analytical alternative to CV that is approximately equivalent to LOOCV.

5. **For lasso and elastic net**, the simple formula $\text{df} = |\hat{S}|$ makes AIC and BIC easy to compute along the regularization path.

## Summary

Information criteria extend to regularized models by replacing the parameter count with the effective degrees of freedom. For ridge, $\text{df}(\lambda)$ is the sum of shrinkage factors. For lasso, $\text{df}(\lambda)$ equals the number of nonzero coefficients. AIC targets optimal prediction, BIC targets model selection consistency, and both require an estimate of the error variance. While cross-validation remains the most robust and widely used approach, information criteria provide computationally efficient alternatives that are especially useful for exploring the regularization path.
