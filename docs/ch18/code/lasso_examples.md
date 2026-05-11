# Lasso Regression Examples

## Overview

Lasso (Least Absolute Shrinkage and Selection Operator) regression adds an $L_1$ penalty to the
OLS objective, which both shrinks coefficients and sets some exactly to zero. This page
implements Lasso from scratch using coordinate descent, visualizes the regularization path, and
selects the tuning parameter $\lambda$ via cross-validation.

## The Lasso Objective

Given $X \in \mathbb{R}^{n \times p}$ and $y \in \mathbb{R}^n$, the Lasso solves

$$
\hat{\beta}^{\text{lasso}} = \arg\min_{\beta} \left\{ \frac{1}{2n}\| y - X\beta \|_2^2 + \lambda \sum_{j=1}^{p} |\beta_j| \right\}.
$$

Unlike ridge regression, the $L_1$ penalty $\lambda \|\beta\|_1$ produces sparse solutions:
for sufficiently large $\lambda$, some coefficients are driven to exactly zero.

## Soft-Thresholding Operator

The key building block of coordinate descent for the Lasso is the **soft-thresholding**
(proximal) operator:

$$
S(\rho,\, \lambda) = \text{sign}(\rho)\, \max(|\rho| - \lambda,\, 0) =
\begin{cases}
\rho - \lambda & \text{if } \rho > \lambda, \\
0 & \text{if } |\rho| \le \lambda, \\
\rho + \lambda & \text{if } \rho < -\lambda.
\end{cases}
$$

## Code: Coordinate Descent Solver

The following pure-NumPy implementation solves the Lasso via cyclic coordinate descent.

```python
import numpy as np

def soft_threshold(rho, lam):
    """Soft-thresholding operator for coordinate descent."""
    if rho > lam:
        return rho - lam
    elif rho < -lam:
        return rho + lam
    return 0.0

def lasso_cd(X, y, lam, max_iter=1000, tol=1e-6):
    """
    Lasso regression via coordinate descent.

    Parameters
    ----------
    X   : (n, p) design matrix (should be standardised).
    y   : (n,)   response vector.
    lam : float  L1 penalty parameter.

    Returns
    -------
    beta : (p,) coefficient vector.
    """
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(p):
            r_j = y - X @ beta + X[:, j] * beta[j]
            rho_j = X[:, j] @ r_j / n
            beta[j] = soft_threshold(rho_j, lam)
        if np.max(np.abs(beta - beta_old)) < tol:
            break
    return beta
```

At each step the algorithm computes the partial residual $r_j = y - X\beta + X_j \beta_j$,
calculates the univariate least-squares slope $\rho_j = X_j^\top r_j / n$, and applies
soft-thresholding.

## Code: Regularization Path

Sweeping over a grid of $\lambda$ values from large to small reveals how coefficients enter the
model.

```python
def lasso_path(X, y, lambdas):
    """Compute coefficient path over a grid of lambda values."""
    coefs = []
    for lam in lambdas:
        beta = lasso_cd(X, y, lam)
        coefs.append(beta.copy())
    return np.array(coefs)
```

## Code: Cross-Validated Lambda Selection

Five-fold CV estimates prediction error for each candidate $\lambda$.

```python
def cv_lasso(X, y, lambdas, folds=5):
    """K-fold cross-validated MSE for each lambda."""
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // folds
    cv_mse = np.zeros(len(lambdas))

    for k in range(folds):
        val_idx = indices[k * fold_size:(k + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        for i, lam in enumerate(lambdas):
            beta = lasso_cd(X_tr, y_tr, lam)
            pred = X_va @ beta
            cv_mse[i] += np.mean((y_va - pred) ** 2)
    return cv_mse / folds
```

## Code: Full Demonstration

Generating synthetic data with 10 predictors (only 3 truly relevant) and running the pipeline:

```python
import matplotlib.pyplot as plt

np.random.seed(42)

n, p = 150, 10
X_raw = np.random.randn(n, p)
beta_true = np.array([4.0, -3.0, 2.0, 0, 0, 0, 0, 0, 0, 0])
y = X_raw @ beta_true + np.random.randn(n) * 2

# Standardise columns
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X = (X_raw - X_mean) / X_std

# Regularisation path
lambdas = np.logspace(1, -2, 60)
path = lasso_path(X, y, lambdas)

# Cross-validated lambda selection
lambdas_cv = np.logspace(1, -2, 30)
mse_cv = cv_lasso(X, y, lambdas_cv)
best_idx = int(np.argmin(mse_cv))
best_lam = lambdas_cv[best_idx]

beta_best = lasso_cd(X, y, best_lam)
n_nonzero = np.sum(np.abs(beta_best) > 1e-8)

print(f"Best lambda (5-fold CV):  {best_lam:.4f}")
print(f"Non-zero coefficients:    {n_nonzero}  (true: 3)")
print(f"Min CV MSE:               {mse_cv[best_idx]:.3f}")
```

## Interpretation

- **Sparsity.** The Lasso correctly identifies the 3 truly relevant predictors and sets the
  remaining 7 to zero (or near zero) at the optimal $\lambda$.
- **Regularization path.** Coefficients enter the model one at a time as $\lambda$ decreases.
  The order in which they appear often reflects their importance.
- **Cross-validation curve.** The CV MSE curve is typically U-shaped: too large a $\lambda$
  underfits (high bias), too small overfits (high variance). The minimum balances the two.
- **Comparison with ridge.** Ridge would retain all 10 predictors with small but nonzero
  coefficients, while Lasso achieves exact sparsity.

## Exercises

**Exercise 1.** Verify by hand that $S(\rho, \lambda)$ is the proximal operator of
$\lambda |\cdot|$, i.e., show that
$S(\rho, \lambda) = \arg\min_{z} \left\{ \frac{1}{2}(z - \rho)^2 + \lambda |z| \right\}$.

??? success "Solution to Exercise 1"

    Define $g(z) = \frac{1}{2}(z - \rho)^2 + \lambda |z|$. Consider three cases.

    **Case 1: $z > 0$.** Then $g(z) = \frac{1}{2}(z-\rho)^2 + \lambda z$, and
    $g'(z) = z - \rho + \lambda = 0$ gives $z^* = \rho - \lambda$. This is positive only when
    $\rho > \lambda$.

    **Case 2: $z < 0$.** Then $g(z) = \frac{1}{2}(z-\rho)^2 - \lambda z$, and
    $g'(z) = z - \rho - \lambda = 0$ gives $z^* = \rho + \lambda$. This is negative only when
    $\rho < -\lambda$.

    **Case 3: $z = 0$.** The subdifferential condition $0 \in \{-\rho\} + \lambda[-1, 1]$
    requires $|\rho| \le \lambda$.

    Combining, $z^* = \text{sign}(\rho)\max(|\rho| - \lambda, 0) = S(\rho, \lambda)$. $\square$

---

**Exercise 2.** In the coordinate descent algorithm, explain why computing the partial residual
$r_j = y - X\beta + X_j \beta_j$ before updating $\beta_j$ is necessary. What would go wrong
if you used the full residual $r = y - X\beta$ instead?

??? success "Solution to Exercise 2"

    The coordinate descent update for $\beta_j$ minimizes the Lasso objective with respect to
    $\beta_j$ while holding all other coefficients fixed. The partial residual $r_j$ removes the
    contribution of feature $j$ from the current fit, so the update depends only on the
    correlation between $X_j$ and the residual **excluding** $X_j$'s own contribution.

    If you used the full residual $r = y - X\beta$, the term $X_j \beta_j$ would still be
    subtracted, effectively double-counting feature $j$'s contribution and leading to a biased
    update. Concretely, $\rho_j$ would be $X_j^\top(y - X\beta)/n$ rather than
    $X_j^\top(y - X_{-j}\beta_{-j})/n$, and the iterates would not converge to the Lasso
    solution. $\square$

---

**Exercise 3.** Modify the `lasso_cd` function to accept warm starts (i.e., an initial
$\beta^{(0)}$ rather than always starting from zero). Explain why warm starts accelerate the
computation of a regularization path.

??? success "Solution to Exercise 3"

    ```python
    def lasso_cd_warm(X, y, lam, beta_init=None, max_iter=1000, tol=1e-6):
        n, p = X.shape
        beta = beta_init.copy() if beta_init is not None else np.zeros(p)
        for _ in range(max_iter):
            beta_old = beta.copy()
            for j in range(p):
                r_j = y - X @ beta + X[:, j] * beta[j]
                rho_j = X[:, j] @ r_j / n
                beta[j] = soft_threshold(rho_j, lam)
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        return beta
    ```

    When computing a regularization path from large to small $\lambda$, the solution at
    $\lambda_k$ is close to the solution at $\lambda_{k+1}$ (by continuity of the solution
    map). Using the previous solution as the initial point typically reduces the number of
    coordinate descent iterations from hundreds to just a few. $\square$

---

**Exercise 4.** Generate a dataset with $n = 100$ and $p = 200$ (more predictors than
observations) where only 5 coefficients are nonzero. Fit the Lasso with CV-selected $\lambda$
and report the number of true positives and false positives among the selected features.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np

    np.random.seed(0)
    n, p = 100, 200
    X = np.random.randn(n, p)
    X = (X - X.mean(0)) / X.std(0)
    beta_true = np.zeros(p)
    beta_true[:5] = [3, -2, 4, -1, 2]
    y = X @ beta_true + np.random.randn(n) * 1.5

    lambdas_cv = np.logspace(1, -2, 30)
    mse_cv = cv_lasso(X, y, lambdas_cv)
    best_lam = lambdas_cv[np.argmin(mse_cv)]
    beta_hat = lasso_cd(X, y, best_lam)

    selected = np.abs(beta_hat) > 1e-8
    true_support = np.abs(beta_true) > 0

    tp = np.sum(selected & true_support)
    fp = np.sum(selected & ~true_support)
    print(f"True positives:  {tp}/5")
    print(f"False positives: {fp}")
    ```

    Typical output: TP = 5, FP = 0--3. The Lasso performs well in high-dimensional sparse
    settings, recovering most true features with few false discoveries. $\square$

---

**Exercise 5.** Prove that the Lasso solution is unique when $X$ has full column rank, but may
not be unique when columns of $X$ are linearly dependent.

??? success "Solution to Exercise 5"

    The Lasso objective is

    $$
    f(\beta) = \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \|\beta\|_1.
    $$

    The first term is a convex quadratic with Hessian $\frac{1}{n}X^\top X$. If $X$ has full
    column rank, $X^\top X$ is positive definite, so $f$ is **strictly convex**. A strictly
    convex function has at most one minimizer, so the solution is unique.

    When columns of $X$ are linearly dependent, $X^\top X$ is only positive semidefinite.
    The quadratic term is convex but not strictly convex, and the $L_1$ term is convex but not
    strictly convex either. The sum is convex, so the set of minimizers is convex, but it may
    contain more than one point.

    **Counterexample:** Let $X = [x \mid x]$ (two identical columns), $y = x$. For any
    $\alpha \in [0,1]$, $\beta = (\alpha, 1 - \alpha)^\top$ achieves the same loss and the same
    $L_1$ norm (when $\alpha \ge 0$), so the solution is not unique. $\square$
