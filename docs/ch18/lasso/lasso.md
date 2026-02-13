# Lasso Regression (L1 Regularization)

## Motivation: Sparsity and Feature Selection

Ridge regression shrinks coefficients toward zero but never sets them exactly to zero. In many applications — especially when $p$ is large — we want an estimator that automatically **selects** relevant features by setting irrelevant coefficients to exactly zero.

The **Lasso** (Least Absolute Shrinkage and Selection Operator), introduced by Tibshirani (1996), achieves both regularization and feature selection by using an L1 penalty.

## Lasso Formulation

$$
\hat{\boldsymbol{\beta}}_{\text{lasso}} = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_1 \right\}
$$

where $\|\boldsymbol{\beta}\|_1 = \sum_{j=1}^p |\beta_j|$ is the L1 norm.

The equivalent constrained form is:

$$
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 \quad \text{subject to} \quad \sum_{j=1}^p |\beta_j| \leq t
$$

## Why L1 Produces Sparsity: Geometric Argument

The L1 constraint set is a **diamond** (in 2D) or cross-polytope (in higher dimensions). Its corners lie on the coordinate axes. Because the OLS contours are elliptical, the first contact point between the ellipse and the diamond typically occurs at a corner, where one or more coordinates are exactly zero.

In contrast, the L2 constraint (sphere) has no corners — the contact point almost never lies on a coordinate axis.

This geometric argument explains the fundamental difference:

| Penalty | Constraint Shape | Corners on Axes | Exact Zeros |
|---|---|---|---|
| L1 (Lasso) | Diamond / cross-polytope | Yes | Yes |
| L2 (Ridge) | Sphere / hypersphere | No | No |

## Subdifferential and the Soft-Thresholding Operator

Unlike the L2 penalty, the L1 norm $|\beta_j|$ is **not differentiable** at $\beta_j = 0$. We use the **subdifferential**:

$$
\partial |\beta_j| = \begin{cases} \{+1\} & \beta_j > 0 \\ [-1, +1] & \beta_j = 0 \\ \{-1\} & \beta_j < 0 \end{cases}
$$

For the special case of **orthonormal design** ($\mathbf{X}^\top\mathbf{X} = n\mathbf{I}$), the Lasso solution has a closed form:

$$
\hat{\beta}_j^{\text{lasso}} = S_\lambda(\hat{\beta}_j^{\text{OLS}}) = \text{sign}(\hat{\beta}_j^{\text{OLS}})\max(|\hat{\beta}_j^{\text{OLS}}| - \lambda, 0)
$$

This is the **soft-thresholding** operator: coefficients within $[-\lambda, \lambda]$ are set to exactly zero, and larger coefficients are shrunk by $\lambda$.

Compare with ridge (orthonormal case):

$$
\hat{\beta}_j^{\text{ridge}} = \frac{\hat{\beta}_j^{\text{OLS}}}{1 + \lambda}
$$

Ridge applies proportional shrinkage; Lasso applies translational shrinkage with hard cutoff.

## Coordinate Descent Algorithm

For general (non-orthonormal) design matrices, no closed-form Lasso solution exists. The standard algorithm is **coordinate descent** (Friedman et al., 2010):

**Algorithm:**

1. Initialize $\hat{\boldsymbol{\beta}} = \mathbf{0}$ (or OLS solution)
2. Cycle through $j = 1, 2, \ldots, p$:
    - Compute the partial residual: $r_i^{(j)} = y_i - \sum_{k \neq j} x_{ik}\hat{\beta}_k$
    - Update: $\hat{\beta}_j \leftarrow S_\lambda\left(\frac{1}{n}\sum_{i=1}^n x_{ij} r_i^{(j)}\right)$
3. Repeat until convergence

Each coordinate update has the soft-thresholding form, making the algorithm efficient. The `glmnet` package (R) and `sklearn.linear_model.Lasso` (Python) use this algorithm.

## The Lasso Solution Path

As $\lambda$ varies from $\lambda_{\max}$ (where all coefficients are zero) down to 0 (OLS), coefficients enter the model one at a time. The path $\hat{\beta}_j(\lambda)$ is **piecewise linear** in $\lambda$ (the LARS result of Efron et al., 2004).

$$
\lambda_{\max} = \frac{1}{n}\|\mathbf{X}^\top\mathbf{y}\|_\infty = \max_j \left|\frac{1}{n}\sum_{i=1}^n x_{ij}y_i\right|
$$

For $\lambda \geq \lambda_{\max}$, the Lasso solution is $\hat{\boldsymbol{\beta}} = \mathbf{0}$.

## Choosing $\lambda$

As with ridge, $\lambda$ is selected by **cross-validation**:

$$
\hat{\lambda} = \arg\min_\lambda \text{CV}(\lambda)
$$

The standard practice is to compute the full solution path on a grid of $\lambda$ values (typically 100 values on a log scale from $\lambda_{\max}$ to $0.001 \cdot \lambda_{\max}$), compute CV error at each, and select the optimal $\lambda$.

## Properties

**1. Sparsity.** The Lasso produces sparse solutions (some $\hat{\beta}_j = 0$), enabling automatic feature selection.

**2. Bias.** Lasso coefficients are biased toward zero (more so than ridge for retained coefficients). The bias does not vanish even asymptotically for the selected variables, though consistent model selection is possible under certain conditions.

**3. Consistency.** Under the **irrepresentable condition** (Zhao and Yu, 2006), the Lasso selects the correct set of nonzero variables with probability approaching 1. Without this condition, model selection consistency can fail.

**4. Prediction.** The Lasso achieves near-optimal prediction error rates under sparsity assumptions.

## Limitations

1. **Group selection.** With highly correlated predictors, Lasso tends to select one and ignore the rest (arbitrary choice). Ridge retains all and distributes the coefficient mass.

2. **$p > n$ limitation.** Lasso can select at most $n$ variables (when $p > n$), since the solution lies in an $n$-dimensional subspace.

3. **Bias of selected coefficients.** Nonzero Lasso coefficients are systematically biased toward zero. Post-Lasso OLS (refit OLS on the Lasso-selected variables) can reduce this bias.

## Bayesian Interpretation

The Lasso is the MAP estimator under a **Laplace (double-exponential) prior**:

$$
\beta_j \overset{\text{iid}}{\sim} \text{Laplace}(0, 1/\lambda)
$$

The Laplace distribution has a sharp peak at zero and heavier tails than the Gaussian, which encourages exact sparsity in the MAP estimate.

| Regularization | Prior | Density at 0 |
|---|---|---|
| Ridge (L2) | $N(0, \tau^2)$ | Smooth, finite | 
| Lasso (L1) | Laplace$(0, b)$ | Cusp, finite |
| Best subset | Spike-and-slab | Point mass at 0 |
