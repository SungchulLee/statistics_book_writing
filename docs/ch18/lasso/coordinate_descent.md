# Coordinate Descent Algorithm

The lasso objective is convex but non-smooth because of the L1 penalty $\lambda\|\boldsymbol{\beta}\|_1$. Standard gradient descent requires a differentiable objective and therefore cannot be applied directly. Coordinate descent avoids this difficulty by optimizing one coefficient at a time, holding all others fixed. Each single-coordinate subproblem has a closed-form solution given by the soft-thresholding operator, making the algorithm both simple and efficient.

## Idea Behind Coordinate Descent

Instead of optimizing all $p$ coefficients simultaneously, coordinate descent cycles through them one at a time. At each step, it minimizes the lasso objective with respect to a single coefficient $\beta_j$, treating all other coefficients as fixed at their current values.

The full lasso objective is:

$$
L(\boldsymbol{\beta}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\sum_{k=1}^p|\beta_k|
$$

When all coordinates except $\beta_j$ are held fixed, the terms involving $\beta_k$ for $k \neq j$ are constant. Define the **partial residual** with the $j$-th predictor removed:

$$
\mathbf{r}^{(j)} = \mathbf{y} - \sum_{k \neq j}\mathbf{x}_k\hat{\beta}_k
$$

The subproblem for $\beta_j$ becomes:

$$
\min_{\beta_j} \left\{\frac{1}{2n}\|\mathbf{r}^{(j)} - \mathbf{x}_j\beta_j\|^2 + \lambda|\beta_j|\right\}
$$

## The Single-Coordinate Update

Expanding the squared term in the subproblem and ignoring constants:

$$
\min_{\beta_j} \left\{\frac{1}{2n}\left(\|\mathbf{r}^{(j)}\|^2 - 2\beta_j\mathbf{x}_j^\top\mathbf{r}^{(j)} + \beta_j^2\|\mathbf{x}_j\|^2\right) + \lambda|\beta_j|\right\}
$$

Assuming the predictors are standardized so that $\|\mathbf{x}_j\|^2 = n$, define:

$$
z_j = \frac{1}{n}\mathbf{x}_j^\top\mathbf{r}^{(j)}
$$

This is the simple regression coefficient of the partial residual on the $j$-th predictor. The subproblem reduces to:

$$
\min_{\beta_j} \left\{\frac{1}{2}(\beta_j - z_j)^2 + \lambda|\beta_j|\right\}
$$

The solution is the soft-thresholding operator applied to $z_j$:

$$
\hat{\beta}_j \leftarrow S_\lambda(z_j) = \text{sign}(z_j)\,\max(|z_j| - \lambda,\; 0)
$$

## The Full Algorithm

The complete coordinate descent algorithm for the lasso is:

**Input:** Data $(\mathbf{X}, \mathbf{y})$, regularization parameter $\lambda$, convergence tolerance $\epsilon$.

1. **Initialize.** Set $\hat{\boldsymbol{\beta}}^{(0)} = \mathbf{0}$ (or the OLS solution if $p < n$).
2. **Cycle.** For $t = 1, 2, \ldots$ until convergence:
    - For $j = 1, 2, \ldots, p$:
        - Compute the partial residual: $\mathbf{r}^{(j)} = \mathbf{y} - \sum_{k \neq j}\mathbf{x}_k\hat{\beta}_k$
        - Compute $z_j = \frac{1}{n}\mathbf{x}_j^\top\mathbf{r}^{(j)}$
        - Update $\hat{\beta}_j \leftarrow S_\lambda(z_j)$
3. **Check convergence.** Stop when $\max_j |\hat{\beta}_j^{(t)} - \hat{\beta}_j^{(t-1)}| < \epsilon$.

**Output:** The lasso estimate $\hat{\boldsymbol{\beta}}_{\text{lasso}}$.

!!! tip "Efficient Residual Updates"
    Rather than recomputing $\mathbf{r}^{(j)}$ from scratch at each step, maintain a running residual $\mathbf{r} = \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}$ and update it incrementally: before updating $\beta_j$, add back the old contribution $\mathbf{r} \leftarrow \mathbf{r} + \mathbf{x}_j\hat{\beta}_j^{\text{old}}$, then after updating, subtract the new contribution $\mathbf{r} \leftarrow \mathbf{r} - \mathbf{x}_j\hat{\beta}_j^{\text{new}}$. This reduces the per-coordinate cost from $O(np)$ to $O(n)$.

## Convergence Properties

Coordinate descent for the lasso converges to the global optimum because:

1. **Convexity.** The lasso objective is convex (the sum of a convex quadratic and a convex L1 term).
2. **Separability of the non-smooth term.** The L1 penalty $\sum_j |\beta_j|$ is separable across coordinates, meaning $\partial_{\beta_j}\|\boldsymbol{\beta}\|_1 = \partial|\beta_j|$ does not depend on the other coordinates.
3. **Block coordinate descent theorem.** For convex objectives where the non-smooth part is separable, cyclic coordinate descent converges to the global minimum (Tseng, 2001).

!!! note "Non-Separable Penalties"
    Coordinate descent is not guaranteed to converge for non-separable penalties such as the group lasso $\sum_g \|\boldsymbol{\beta}_g\|_2$, where the L2 norm couples coefficients within groups. For such penalties, block coordinate descent (updating groups rather than individual coordinates) is required.

## Warm Starts and the Solution Path

In practice, the lasso is solved for a grid of $\lambda$ values from $\lambda_{\max}$ down to a small fraction of $\lambda_{\max}$. **Warm starting** uses the solution at $\lambda_{m}$ as the initial point for $\lambda_{m+1}$. Because nearby $\lambda$ values produce similar solutions, warm starts dramatically reduce the number of iterations needed.

The computation of the full regularization path (solutions for all $\lambda$ values) using coordinate descent with warm starts requires roughly the same computational effort as solving the lasso at a single $\lambda$ value from scratch. This efficiency is one of the main reasons coordinate descent has become the standard algorithm for lasso.

## Computational Complexity

| Operation | Cost |
|---|---|
| One coordinate update | $O(n)$ |
| One full cycle (all $p$ coordinates) | $O(np)$ |
| Convergence (typically 10-100 cycles) | $O(np \times \text{iterations})$ |
| Full path ($M$ lambda values with warm starts) | $O(Mnp)$ but with small constants |

For large $p$ and sparse solutions, active set strategies further reduce computation by only updating coefficients that are currently nonzero or candidates for becoming nonzero.

## Extension to Elastic Net

The coordinate descent framework extends naturally to the elastic net, which combines L1 and L2 penalties. The single-coordinate update becomes:

$$
\hat{\beta}_j \leftarrow \frac{S_{\alpha\lambda}(z_j)}{1 + (1-\alpha)\lambda}
$$

where $\alpha$ is the mixing parameter between L1 and L2. The denominator $1 + (1-\alpha)\lambda$ accounts for the L2 penalty, and the numerator applies soft-thresholding for the L1 penalty. This unified framework allows the same algorithm to handle ridge, lasso, and elastic net by varying $\alpha$.

## Summary

Coordinate descent solves the lasso by cycling through coordinates and applying the soft-thresholding operator to each one. The algorithm is efficient ($O(n)$ per coordinate update), guaranteed to converge to the global optimum for separable non-smooth penalties, and naturally accommodates warm starts for computing the full regularization path. Its simplicity and efficiency have made it the standard algorithm for lasso and elastic net, implemented in widely used packages such as `glmnet` and `scikit-learn`.

## Exercises

**Exercise 1.**
Implement the coordinate descent algorithm for Lasso:

(a) Write the soft-thresholding operator $S_\lambda(z)$.

(b) Implement the full coordinate descent loop with convergence check.

(c) Compare your solution paths with `sklearn.linear_model.Lasso`.
