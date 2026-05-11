# Elastic Net Examples

## Overview

The Elastic Net combines the $L_1$ (Lasso) and $L_2$ (Ridge) penalties into a single
regularization framework. It inherits the sparsity-inducing property of the Lasso while
retaining ridge regression's ability to handle groups of correlated predictors. This page
develops the Elastic Net objective, derives its solution in the orthonormal case, and discusses
practical guidance for choosing its two tuning parameters.

## The Elastic Net Objective

Given $X \in \mathbb{R}^{n \times p}$ and $y \in \mathbb{R}^n$, the Elastic Net solves

$$
\hat{\beta}^{\text{EN}} = \arg\min_{\beta} \left\{ \frac{1}{2n}\| y - X\beta \|_2^2 + \lambda \left[ \alpha \|\beta\|_1 + \frac{1 - \alpha}{2} \|\beta\|_2^2 \right] \right\},
$$

where $\lambda \ge 0$ controls the overall regularization strength and $\alpha \in [0, 1]$
controls the mix between the two penalties:

- $\alpha = 1$: pure Lasso.
- $\alpha = 0$: pure Ridge.
- $0 < \alpha < 1$: Elastic Net.

## Orthonormal Design Solution

When $X^\top X = n I_p$, the Elastic Net estimate for the $j$-th coefficient reduces to

$$
\hat{\beta}_j^{\text{EN}} = \frac{1}{1 + \lambda(1 - \alpha)}\, S\!\left(\hat{\beta}_j^{\text{OLS}},\; \lambda \alpha\right),
$$

where $S(\cdot, \cdot)$ is the soft-thresholding operator. This shows that the Elastic Net
first soft-thresholds (Lasso step) and then rescales (Ridge step).

## Grouped Selection

A major advantage of the Elastic Net over the Lasso is its behavior with correlated predictors.
When several predictors are highly correlated:

- **Lasso** tends to select one and set the others to zero (unstable selection).
- **Elastic Net** tends to select or exclude the entire group together.

This **grouping effect** is a consequence of the strictly convex $L_2$ component of the
penalty, which was proved by Zou and Hastie (2005).

## Code: Basic Demonstration

The following script generates sample data. A full implementation would extend this to fit the
Elastic Net via coordinate descent with both penalties.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

n = 100
data = np.random.normal(loc=0, scale=1, size=n)

print(f"Sample size: {n}")
print(f"Sample mean: {data.mean():.4f}")
print(f"Sample std:  {data.std(ddof=1):.4f}")
```

In practice, one would use `sklearn.linear_model.ElasticNetCV` to jointly select $\lambda$ and
$\alpha$ via cross-validation.

## Coordinate Descent for Elastic Net

The update for the $j$-th coefficient in coordinate descent takes the form

$$
\beta_j \leftarrow \frac{S\!\left(X_j^\top r_j / n,\; \lambda \alpha\right)}{1 + \lambda(1 - \alpha)},
$$

where $r_j = y - X_{-j}\beta_{-j}$ is the partial residual. Compared to the Lasso update, the
only difference is the denominator $1 + \lambda(1 - \alpha)$ from the Ridge component.

## Choosing the Tuning Parameters

The Elastic Net has two hyperparameters, $\lambda$ and $\alpha$. A common strategy:

1. Fix a grid of $\alpha$ values (e.g., $\{0.1, 0.5, 0.7, 0.9, 0.95\}$).
2. For each $\alpha$, use cross-validation over a grid of $\lambda$ values.
3. Select the $(\alpha, \lambda)$ pair with the lowest CV error.

## Interpretation

- **Sparsity + stability.** The Elastic Net achieves variable selection (some coefficients
  exactly zero) while being more stable than the Lasso when predictors are correlated.
- **Unique solution.** Unlike the Lasso, the Elastic Net objective is strictly convex when
  $\alpha < 1$, so the solution is always unique.
- **Computational cost.** Coordinate descent for the Elastic Net is essentially the same as for
  the Lasso, with a minor modification to the denominator.

## Exercises

**Exercise 1.** Starting from the Elastic Net objective, derive the coordinate descent update
$\beta_j \leftarrow S(X_j^\top r_j / n,\, \lambda\alpha) / (1 + \lambda(1 - \alpha))$.

??? success "Solution to Exercise 1"

    Fix all coefficients except $\beta_j$. The objective as a function of $\beta_j$ alone is

    $$
    g(\beta_j) = \frac{1}{2n}\|r_j - X_j \beta_j\|_2^2 + \lambda\alpha |\beta_j| + \frac{\lambda(1 - \alpha)}{2}\beta_j^2,
    $$

    where $r_j = y - X_{-j}\beta_{-j}$. Expanding the quadratic and ignoring terms not
    involving $\beta_j$:

    $$
    g(\beta_j) = \frac{1}{2}\!\left(\frac{\|X_j\|^2}{n} + \lambda(1-\alpha)\right)\beta_j^2 - \frac{X_j^\top r_j}{n}\,\beta_j + \lambda\alpha|\beta_j| + C.
    $$

    Assuming standardized features with $\|X_j\|^2/n = 1$, the minimizer of a function of the
    form $\frac{1}{2}a\, z^2 - b\, z + \lambda\alpha|z|$ with $a = 1 + \lambda(1-\alpha)$ is

    $$
    \beta_j^* = \frac{S(b,\, \lambda\alpha)}{a} = \frac{S(X_j^\top r_j/n,\, \lambda\alpha)}{1 + \lambda(1-\alpha)}. \quad \square
    $$

---

**Exercise 2.** Prove that the Elastic Net objective is strictly convex when $\alpha < 1$ and
$\lambda > 0$, and conclude that the solution is unique.

??? success "Solution to Exercise 2"

    The Elastic Net objective is $f(\beta) = h(\beta) + \lambda\alpha\|\beta\|_1$, where

    $$
    h(\beta) = \frac{1}{2n}\|y - X\beta\|_2^2 + \frac{\lambda(1-\alpha)}{2}\|\beta\|_2^2.
    $$

    The Hessian of $h$ is $\nabla^2 h = \frac{1}{n}X^\top X + \lambda(1-\alpha)I_p$. For
    $\alpha < 1$ and $\lambda > 0$, the term $\lambda(1-\alpha)I_p$ is positive definite, so
    $\nabla^2 h$ is positive definite. Hence $h$ is strictly convex.

    Since $f = h + \lambda\alpha\|\cdot\|_1$ is the sum of a strictly convex function and a
    convex function, $f$ is strictly convex. A strictly convex function has at most one
    minimizer, and the coercivity of $f$ guarantees existence. Therefore the Elastic Net
    solution is unique. $\square$

---

**Exercise 3.** Consider two predictors $x_1$ and $x_2$ with correlation $\rho$ close to 1.
Explain qualitatively why the Lasso might select only one of them while the Elastic Net tends
to select both, and relate this to the geometry of the constraint regions.

??? success "Solution to Exercise 3"

    The Lasso constraint region $\|\beta\|_1 \le t$ has corners along the coordinate axes. When
    $x_1 \approx x_2$, the loss function contours are elongated ellipses nearly parallel to the
    line $\beta_1 = \beta_2$. The first contact between these ellipses and the diamond-shaped
    $L_1$ ball typically occurs at a corner, where one coefficient is zero---thus the Lasso
    selects one but not the other.

    The Elastic Net constraint region is a blend of the $L_1$ diamond and the $L_2$ ball. The
    $L_2$ component rounds the corners, so the boundary near $(\beta_1, \beta_2) = (c, c)$ is
    smooth. The elongated ellipses are more likely to make first contact along this smooth
    boundary where both coefficients are nonzero, leading to grouped selection.

    Formally, Zou and Hastie (2005) proved that if $x_i^\top x_j / n = \rho$ and both
    $\hat{\beta}_i, \hat{\beta}_j \ne 0$, then
    $|\hat{\beta}_i - \hat{\beta}_j| \le \frac{\|y\|_1}{\lambda(1-\alpha)n}\sqrt{2(1 - \rho)}$,
    so the coefficients are close when the correlation is high. $\square$

---

**Exercise 4.** Use `ElasticNetCV` from scikit-learn to fit an Elastic Net on a synthetic
dataset with $n = 200$, $p = 50$, and 5 true nonzero coefficients with pairwise correlation
$\rho = 0.95$ among the first 10 predictors. Report the selected $\alpha$, $\lambda$, and the
number of nonzero coefficients.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    n, p = 200, 50

    # Correlated block in first 10 features
    Sigma = np.eye(p)
    for i in range(10):
        for j in range(10):
            if i != j:
                Sigma[i, j] = 0.95
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T

    beta_true = np.zeros(p)
    beta_true[:5] = [3, -2, 4, -1, 2]
    y = X @ beta_true + np.random.randn(n)

    X_s = StandardScaler().fit_transform(X)

    enet_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
        n_alphas=100, cv=5, max_iter=10000
    )
    enet_cv.fit(X_s, y)

    n_nonzero = np.sum(np.abs(enet_cv.coef_) > 1e-6)
    print(f"Selected alpha (l1_ratio): {enet_cv.l1_ratio_}")
    print(f"Selected lambda:           {enet_cv.alpha_:.6f}")
    print(f"Non-zero coefficients:     {n_nonzero}")
    ```

    Typical results: $\alpha \in \{0.5, 0.7\}$, and the Elastic Net selects roughly 5--10
    features (retaining correlated partners of the true features). $\square$

---

**Exercise 5.** Show that in the orthonormal design case ($X^\top X = nI_p$), the Elastic Net
estimator can be written as $\hat{\beta}_j^{\text{EN}} = \frac{1}{1+\lambda(1-\alpha)}\,S(\hat{\beta}_j^{\text{OLS}},\, \lambda\alpha)$. Interpret the two operations (soft-thresholding followed by rescaling) geometrically.

??? success "Solution to Exercise 5"

    With $X^\top X = nI_p$, the Elastic Net objective separates into $p$ independent
    univariate problems:

    $$
    \min_{\beta_j} \left\{ \frac{1}{2}(\hat{\beta}_j^{\text{OLS}} - \beta_j)^2 + \lambda\alpha|\beta_j| + \frac{\lambda(1-\alpha)}{2}\beta_j^2 \right\}.
    $$

    Combining the quadratic terms:

    $$
    \min_{\beta_j} \left\{ \frac{1 + \lambda(1-\alpha)}{2}\beta_j^2 - \hat{\beta}_j^{\text{OLS}}\beta_j + \lambda\alpha|\beta_j| \right\}.
    $$

    By the proximal operator result (Exercise 1), the solution is

    $$
    \hat{\beta}_j^{\text{EN}} = \frac{S(\hat{\beta}_j^{\text{OLS}},\, \lambda\alpha)}{1 + \lambda(1-\alpha)}.
    $$

    **Geometric interpretation:** Soft-thresholding translates the OLS estimate toward zero and
    clips small values to exactly zero (the Lasso step, producing sparsity). Dividing by
    $1 + \lambda(1-\alpha)$ then uniformly scales the surviving coefficients toward zero (the
    Ridge step, providing additional shrinkage). The two operations together give both sparsity
    and continuous shrinkage. $\square$
