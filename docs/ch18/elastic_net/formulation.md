# Elastic Net Formulation

Ridge regression handles correlated predictors well but cannot perform feature selection. Lasso performs feature selection but struggles with correlated predictors, arbitrarily selecting one from a group and discarding the rest. The **elastic net** (Zou and Hastie, 2005) combines both penalties, inheriting the sparsity of the lasso and the stability of ridge regression. This section defines the elastic net objective, introduces the mixing parameter, and establishes the connection to its two special cases.

## The Elastic Net Objective

The elastic net solves:

$$
\hat{\boldsymbol{\beta}}_{\text{EN}} = \arg\min_{\boldsymbol{\beta}} \left\{\frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\bigl[\alpha\|\boldsymbol{\beta}\|_1 + (1 - \alpha)\tfrac{1}{2}\|\boldsymbol{\beta}\|_2^2\bigr]\right\}
$$

where:

- $\lambda \geq 0$ is the overall regularization strength.
- $\alpha \in [0, 1]$ is the **mixing parameter** that balances the L1 and L2 penalties.
- The factor $1/2$ in front of $\|\boldsymbol{\beta}\|_2^2$ is a convention that simplifies the coordinate descent update.

## Special Cases

The mixing parameter $\alpha$ interpolates between ridge and lasso:

| $\alpha$ | Method | Penalty |
|---|---|---|
| $\alpha = 0$ | Ridge regression | $\frac{\lambda}{2}\|\boldsymbol{\beta}\|_2^2$ |
| $\alpha = 1$ | Lasso | $\lambda\|\boldsymbol{\beta}\|_1$ |
| $0 < \alpha < 1$ | Elastic net | $\lambda\bigl[\alpha\|\boldsymbol{\beta}\|_1 + (1-\alpha)\tfrac{1}{2}\|\boldsymbol{\beta}\|_2^2\bigr]$ |

Values of $\alpha$ near 1 produce sparser models (closer to lasso behavior), while values near 0 produce denser models with more shrinkage (closer to ridge behavior).

## Alternative Parameterization

Some formulations use two separate regularization parameters:

$$
\hat{\boldsymbol{\beta}}_{\text{EN}} = \arg\min_{\boldsymbol{\beta}} \left\{\frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda_1\|\boldsymbol{\beta}\|_1 + \lambda_2\|\boldsymbol{\beta}\|_2^2\right\}
$$

The two parameterizations are related by $\lambda_1 = \lambda\alpha$ and $\lambda_2 = \lambda(1 - \alpha)/2$. The $(\lambda, \alpha)$ parameterization is more common in practice because it separates the total penalty strength ($\lambda$) from the penalty mixture ($\alpha$), allowing cross-validation over $\lambda$ for a fixed $\alpha$.

## Constrained Optimization Form

The elastic net penalty defines a constraint set that blends the L1 diamond and L2 sphere:

$$
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 \quad \text{subject to} \quad \alpha\|\boldsymbol{\beta}\|_1 + (1-\alpha)\tfrac{1}{2}\|\boldsymbol{\beta}\|_2^2 \leq t
$$

In two dimensions, this constraint region is a rounded diamond: the corners of the L1 diamond are present (enabling sparsity) but the flat edges are replaced by curves (providing stability). As $\alpha \to 0$, the shape approaches a circle; as $\alpha \to 1$, it approaches a diamond.

!!! note "Geometry of the Elastic Net Constraint"
    The elastic net constraint set inherits corners from the L1 ball (which produce exact zeros) and smoothness from the L2 ball (which provides stability for correlated predictors). This geometric combination is the key to the elastic net's ability to perform variable selection while handling multicollinearity.

## Coordinate Descent for Elastic Net

The coordinate descent algorithm for the lasso extends naturally to the elastic net. For standardized predictors, the update for coefficient $\beta_j$ is:

$$
\hat{\beta}_j \leftarrow \frac{S_{\alpha\lambda}(z_j)}{1 + (1-\alpha)\lambda}
$$

where $z_j = \frac{1}{n}\mathbf{x}_j^\top\mathbf{r}^{(j)}$ is the simple regression coefficient of the partial residual on predictor $j$, and $S_{\alpha\lambda}$ is the soft-thresholding operator with threshold $\alpha\lambda$.

The numerator applies L1 soft-thresholding (the lasso component), and the denominator applies L2 shrinkage (the ridge component). When $\alpha = 1$, the denominator becomes 1 and the update reduces to the lasso update. When $\alpha = 0$, the numerator has no thresholding and the update reduces to the ridge update.

## Strict Convexity

The elastic net objective is **strictly convex** whenever $\alpha < 1$ (i.e., when the L2 component is present), even if $p > n$. This is because the L2 penalty adds $\lambda(1 - \alpha)\|\boldsymbol{\beta}\|_2^2/2$ to the objective, making the Hessian positive definite. In contrast, the pure lasso objective ($\alpha = 1$) is convex but not strictly convex when $p > n$, and its solution may not be unique.

!!! tip "Practical Implication of Strict Convexity"
    The unique solution property of the elastic net (for $\alpha < 1$) means the solution path is more stable and reproducible than the lasso path when $p > n$. This is one reason to prefer the elastic net over the pure lasso in high-dimensional settings.

## Choosing Alpha

The mixing parameter $\alpha$ is typically selected by one of two approaches:

1. **Grid search with cross-validation.** Perform a two-dimensional grid search over $(\lambda, \alpha)$, using $K$-fold CV to evaluate each pair. This is the most thorough approach but computationally expensive.

2. **Fix $\alpha$ and optimize $\lambda$.** Choose $\alpha$ based on domain knowledge (e.g., $\alpha = 0.5$ for a balanced penalty) and then optimize $\lambda$ via CV. This is faster and often sufficient.

Common choices include $\alpha = 0.5$ (equal weight to L1 and L2), $\alpha = 0.9$ (mostly lasso with slight ridge stabilization), and $\alpha = 0.1$ (mostly ridge with slight lasso sparsity).

## Summary

The elastic net combines L1 and L2 penalties through the mixing parameter $\alpha$, interpolating between ridge ($\alpha = 0$) and lasso ($\alpha = 1$). The L1 component provides sparsity; the L2 component provides strict convexity and stability for correlated predictors. The coordinate descent update naturally incorporates both penalties, and the elastic net constraint set is a rounded diamond that inherits corners (for sparsity) from L1 and smoothness (for stability) from L2. In practice, $\alpha$ is chosen by cross-validation or domain knowledge, with $\lambda$ optimized for each fixed $\alpha$.
