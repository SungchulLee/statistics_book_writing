# Ridge vs Lasso vs Elastic Net

This section provides a systematic comparison of the three main regularization methods: ridge regression (L2), lasso (L1), and elastic net (L1 + L2). Each method makes different tradeoffs between shrinkage, sparsity, stability, and computational cost. Understanding these tradeoffs is essential for choosing the right method for a given problem.

## Penalty and Objective Comparison

All three methods minimize a penalized least squares objective of the form:

$$
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \left\{\frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\, P(\boldsymbol{\beta})\right\}
$$

The methods differ in the penalty $P(\boldsymbol{\beta})$:

| Method | Penalty $P(\boldsymbol{\beta})$ | Constraint shape |
|---|---|---|
| Ridge | $\frac{1}{2}\|\boldsymbol{\beta}\|_2^2 = \frac{1}{2}\sum_j \beta_j^2$ | Sphere (L2 ball) |
| Lasso | $\|\boldsymbol{\beta}\|_1 = \sum_j \lvert\beta_j\rvert$ | Diamond (L1 ball) |
| Elastic net | $\alpha\|\boldsymbol{\beta}\|_1 + (1-\alpha)\frac{1}{2}\|\boldsymbol{\beta}\|_2^2$ | Rounded diamond |

## Comprehensive Property Comparison

| Property | Ridge | Lasso | Elastic Net |
|---|---|---|---|
| Feature selection | No | Yes | Yes |
| Exact zeros | Never | Yes | Yes |
| Shrinkage type | Proportional | Translational (soft-threshold) | Combined |
| Handles multicollinearity | Yes (distributes coefficients) | Poorly (selects one) | Yes (grouping effect) |
| Solution uniqueness | Always ($\lambda > 0$) | Not when $p > n$ | Always ($\alpha < 1$) |
| Closed-form solution | Yes | Only for orthonormal $\mathbf{X}$ | No |
| Max features selected ($p > n$) | All $p$ (nonzero) | At most $n$ | No limit |
| Bayesian prior | Gaussian $N(0, \tau^2)$ | Laplace$(0, b)$ | Mixture |
| Computation | Matrix inversion | Coordinate descent | Coordinate descent |
| Hyperparameters | $\lambda$ | $\lambda$ | $\lambda, \alpha$ |
| Effective df formula | $\sum d_j^2/(d_j^2 + \lambda)$ | Number of nonzero coefficients | Approximate |

## Coefficient Path Behavior

As $\lambda$ decreases from a large value toward zero, the three methods trace different coefficient paths.

**Ridge path.** All coefficients are nonzero for any $\lambda > 0$. They start near zero (large $\lambda$) and grow continuously toward their OLS values. The paths are smooth curves. No coefficient ever reaches exactly zero.

**Lasso path.** Coefficients enter the model one at a time as $\lambda$ decreases. The path is piecewise linear. At each breakpoint, a new variable enters (or occasionally leaves) the active set. The path begins at $\hat{\boldsymbol{\beta}} = \mathbf{0}$ for $\lambda \geq \lambda_{\max}$ and approaches OLS as $\lambda \to 0$.

**Elastic net path.** Similar to lasso in that coefficients enter progressively, but the paths are smoother due to the L2 component. Correlated predictors tend to enter together (grouping effect), and the paths are more stable across perturbations of the data.

## Shrinkage Behavior in the Orthonormal Case

When $\mathbf{X}^\top\mathbf{X} = n\mathbf{I}$, each method has a closed-form solution that reveals its shrinkage behavior:

$$
\hat{\beta}_j^{\text{ridge}} = \frac{\hat{\beta}_j^{\text{OLS}}}{1 + \lambda}
$$

$$
\hat{\beta}_j^{\text{lasso}} = \text{sign}(\hat{\beta}_j^{\text{OLS}})\,\max\bigl(|\hat{\beta}_j^{\text{OLS}}| - \lambda,\; 0\bigr)
$$

$$
\hat{\beta}_j^{\text{EN}} = \frac{\text{sign}(\hat{\beta}_j^{\text{OLS}})\,\max\bigl(|\hat{\beta}_j^{\text{OLS}}| - \alpha\lambda,\; 0\bigr)}{1 + (1-\alpha)\lambda}
$$

Ridge divides by a constant factor (proportional shrinkage). Lasso subtracts a constant amount (translational shrinkage) and truncates at zero. The elastic net combines both: it subtracts $\alpha\lambda$ (soft-thresholding from L1) and then divides by $1 + (1-\alpha)\lambda$ (proportional shrinkage from L2).

## Bias-Variance Profiles

All three methods introduce bias to reduce variance. Their bias-variance profiles differ:

**Ridge.** Moderate bias on all coefficients. Variance reduction is strongest along the smallest eigenvector directions of $\mathbf{X}^\top\mathbf{X}$. Total MSE is reduced because the variance reduction exceeds the bias increase.

**Lasso.** Zero bias on eliminated coefficients (they are set to zero, and if the true coefficient is zero, this is correct). However, retained coefficients have bias due to soft-thresholding. The variance reduction comes from eliminating noisy coefficient estimates.

**Elastic net.** Combines the bias-variance profiles of ridge and lasso. The L2 component reduces variance on all coefficients; the L1 component eliminates irrelevant ones.

## Performance in Different Scenarios

!!! note "No Universally Best Method"
    No single method dominates in all scenarios. The best choice depends on the true data-generating process, the correlation structure of predictors, the ratio $p/n$, and the goals of the analysis.

**Scenario 1: Many small effects, no sparsity.**
All predictors contribute a small amount to the response. Ridge regression performs best because it retains all predictors and distributes shrinkage optimally.

**Scenario 2: Few large effects, true sparsity.**
Only a handful of predictors are relevant, and they are not strongly correlated. Lasso performs best because it identifies the relevant predictors and sets the rest to zero.

**Scenario 3: Grouped effects with sparsity.**
Several groups of correlated predictors are relevant. The elastic net performs best because it selects groups of correlated predictors together (grouping effect) while maintaining sparsity.

**Scenario 4: High-dimensional, $p \gg n$.**
With more predictors than observations and moderate sparsity, the elastic net is preferred because the lasso's $n$-feature limit may be binding.

## Computational Cost

| Method | Per-lambda cost | Path (M lambdas) | LOOCV shortcut |
|---|---|---|---|
| Ridge | $O(p^3)$ or $O(np^2)$ | $O(Mp^3)$ | Yes (closed-form) |
| Lasso | $O(np \times \text{iters})$ | $O(Mnp)$ with warm starts | No |
| Elastic net | $O(np \times \text{iters})$ | $O(Mnp)$ with warm starts | No |

Ridge has the advantage of a closed-form solution and a LOOCV shortcut. Lasso and elastic net require iterative coordinate descent but benefit from warm starts along the regularization path.

## Summary

Ridge regression provides smooth, non-sparse shrinkage that handles multicollinearity well. Lasso provides sparse solutions through the L1 penalty but can be unstable with correlated predictors and is limited to $n$ features when $p > n$. The elastic net combines both penalties, achieving sparsity, grouping of correlated features, solution uniqueness, and no upper limit on selected features. The choice among them depends on the specific characteristics of the problem: the degree of true sparsity, the correlation structure of predictors, the ratio of $p$ to $n$, and whether the goal is prediction or interpretation.

## Exercises

**Exercise 1.**
For orthonormal design ($\mathbf{X}^\top\mathbf{X} = n\mathbf{I}$), derive the closed-form solutions for Ridge and Lasso. Explain geometrically why Lasso produces exact zeros but Ridge does not.

---

**Exercise 2.**
Generate $n = 100$ observations from the model $y = 3x_1 - 2x_2 + 0.5x_3 + \varepsilon$ where $\varepsilon \sim N(0, 1)$, along with 17 noise predictors ($x_4, \ldots, x_{20}$).

(a) Fit OLS, Ridge, Lasso, and Elastic Net. Compare the coefficient estimates.

(b) Which methods correctly identify the 3 true predictors?

(c) Use 5-fold CV to select the best $\lambda$ for each method. Report test MSE.
