# Cross-Validation and Lambda Tuning

## Overview

The regularization parameter $\lambda$ controls the strength of the penalty in Ridge, Lasso,
and Elastic Net regression. Choosing $\lambda$ well is critical: too small and the model
overfits, too large and it underfits. Cross-validation (CV) provides a principled,
data-driven method for selecting $\lambda$ by estimating out-of-sample prediction error.

## The Tuning Problem

For any penalized regression method, the in-sample loss decreases monotonically as $\lambda$
decreases toward zero (less regularization = better fit to training data). However,
out-of-sample prediction error typically has a U-shaped curve:

$$
\text{CV}(\lambda) = \frac{1}{K}\sum_{k=1}^{K} \text{MSE}^{(-k)}(\lambda),
$$

where $\text{MSE}^{(-k)}$ is the mean squared error on fold $k$ when the model is trained on
the remaining $K - 1$ folds. The optimal $\lambda$ minimizes this curve.

## K-Fold Cross-Validation

The standard procedure:

1. Randomly partition the data into $K$ folds of approximately equal size.
2. For each candidate $\lambda$ in a grid $\{\lambda_1, \dots, \lambda_M\}$:
    - For $k = 1, \dots, K$: hold out fold $k$, fit the model on the remaining data, and
      compute the prediction error on fold $k$.
    - Average the $K$ prediction errors to get $\text{CV}(\lambda)$.
3. Select $\hat{\lambda} = \arg\min_\lambda \text{CV}(\lambda)$.

Common choices are $K = 5$ or $K = 10$.

## The Lambda Grid

The grid of candidate $\lambda$ values is typically chosen on a logarithmic scale:

$$
\lambda_1 > \lambda_2 > \cdots > \lambda_M, \quad \text{where } \lambda_m = 10^{a + (b-a)\frac{m-1}{M-1}}
$$

for some range $[a, b]$ (e.g., $a = -4$, $b = 2$). In practice:

- $\lambda_{\max}$ is the smallest value that sets all Lasso coefficients to zero:
  $\lambda_{\max} = \frac{1}{n}\|X^\top y\|_\infty$.
- The grid extends from $\lambda_{\max}$ down to some small fraction
  $\epsilon \cdot \lambda_{\max}$ (e.g., $\epsilon = 10^{-4}$).

## Code: Basic CV Demonstration

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

A complete implementation partitions the data into folds, loops over a grid of $\lambda$ values,
and records the MSE for each fold--$\lambda$ combination.

## The One-Standard-Error Rule

Instead of choosing $\hat{\lambda}$ as the exact minimizer of the CV curve, a common
conservative rule is to choose the largest $\lambda$ whose CV error is within one standard
error of the minimum:

$$
\hat{\lambda}_{1\text{SE}} = \max\left\{ \lambda : \text{CV}(\lambda) \le \text{CV}(\hat{\lambda}) + \text{SE}(\hat{\lambda}) \right\}.
$$

This rule favors simpler (more regularized) models with only a negligible increase in
prediction error.

## Practical Considerations

- **Standardization.** Always standardize predictors before CV. Apply the standardization
  parameters (mean and standard deviation) computed on the training folds, not the full
  dataset, to avoid data leakage.
- **Nested CV.** If you also want to estimate the final model's generalization error, use
  nested (double) cross-validation: an outer loop estimates test error, and an inner loop
  selects $\lambda$.
- **Computational cost.** Warm starts (initializing coordinate descent from the previous
  $\lambda$'s solution) dramatically speed up the path computation.
- **Random splits.** Shuffling before splitting or using repeated CV reduces sensitivity to
  a single partition.

## Interpretation

- The CV curve should be U-shaped (or at least non-monotone). If it is still decreasing at the
  smallest $\lambda$, extend the grid toward smaller values.
- If it is still decreasing at the largest $\lambda$, the data may not need regularization.
- The 1SE rule produces sparser, more interpretable models at the cost of a slight increase in
  prediction error.

## Exercises

**Exercise 1.** Derive the formula $\lambda_{\max} = \frac{1}{n}\|X^\top y\|_\infty$ for the
Lasso. That is, show that for $\lambda \ge \lambda_{\max}$, the Lasso solution is
$\hat{\beta} = 0$.

??? success "Solution to Exercise 1"

    The Lasso objective is $f(\beta) = \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda\|\beta\|_1$.
    The subdifferential at $\beta = 0$ is

    $$
    \partial f(0) = \left\{-\frac{1}{n}X^\top y + \lambda s : s \in \partial\|\cdot\|_1(0)\right\} = \left\{-\frac{1}{n}X^\top y + \lambda s : s_j \in [-1,1]\right\}.
    $$

    The optimality condition $0 \in \partial f(0)$ requires $\frac{1}{n}X^\top y = \lambda s$
    for some $s$ with $|s_j| \le 1$. This is achievable if and only if
    $\frac{1}{n}|X_j^\top y| \le \lambda$ for all $j$, i.e.,
    $\lambda \ge \frac{1}{n}\|X^\top y\|_\infty = \lambda_{\max}$. $\square$

---

**Exercise 2.** Explain why using the full dataset to standardize predictors before
cross-validation introduces data leakage, and describe the correct procedure.

??? success "Solution to Exercise 2"

    If you compute $\bar{x}_j$ and $s_j$ on the full dataset (including the validation fold)
    and then standardize, the validation fold's data influences the transformation applied to
    it. This means the model has indirectly "seen" information from the validation fold during
    training, making CV error estimates optimistically biased.

    **Correct procedure:** Inside each CV iteration, compute the mean and standard deviation
    from the training folds only, then apply that same transformation to the held-out fold:

    ```python
    for k in range(K):
        X_train, X_val = X[train_idx], X[val_idx]
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0)
        X_train_s = (X_train - mu) / sigma
        X_val_s = (X_val - mu) / sigma
        # fit on X_train_s, evaluate on X_val_s
    ```

    This ensures the validation fold is truly unseen. $\square$

---

**Exercise 3.** Implement the one-standard-error rule. Given arrays `lambdas`, `cv_mean`, and
`cv_se` (mean and standard error of CV MSE for each $\lambda$), write a function that returns
$\hat{\lambda}_{1\text{SE}}$.

??? success "Solution to Exercise 3"

    ```python
    def one_se_rule(lambdas, cv_mean, cv_se):
        """
        Select the largest lambda whose CV error is within
        one SE of the minimum CV error.
        """
        best_idx = np.argmin(cv_mean)
        threshold = cv_mean[best_idx] + cv_se[best_idx]

        # Among lambdas with CV error <= threshold, pick the largest
        candidates = np.where(cv_mean <= threshold)[0]
        # Largest lambda = smallest index if lambdas sorted descending
        # or largest value directly
        best_1se_idx = candidates[np.argmax(lambdas[candidates])]
        return lambdas[best_1se_idx]
    ```

    The function finds the minimum CV error, adds one standard error to get the threshold,
    and then picks the most regularized model (largest $\lambda$) within that threshold.
    $\square$

---

**Exercise 4.** Run 5-fold CV for the Lasso on a synthetic dataset ($n = 200$, $p = 30$, 5
true nonzero coefficients) over 50 logarithmically spaced $\lambda$ values. Plot the CV curve
with error bars ($\pm 1$ SE) and mark both $\hat{\lambda}_{\min}$ and
$\hat{\lambda}_{1\text{SE}}$.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    n, p = 200, 30
    X = np.random.randn(n, p)
    beta_true = np.zeros(p)
    beta_true[:5] = [3, -2, 4, -1, 2]
    y = X @ beta_true + np.random.randn(n)

    X_s = StandardScaler().fit_transform(X)
    lambdas = np.logspace(1, -3, 50)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    mse_folds = np.zeros((len(lambdas), 5))
    for fold_idx, (tr, va) in enumerate(kf.split(X_s)):
        mu, sig = X_s[tr].mean(0), X_s[tr].std(0)
        Xtr = (X_s[tr] - mu) / sig
        Xva = (X_s[va] - mu) / sig
        for i, lam in enumerate(lambdas):
            m = Lasso(alpha=lam, max_iter=10000).fit(Xtr, y[tr])
            mse_folds[i, fold_idx] = np.mean((y[va] - m.predict(Xva))**2)

    cv_mean = mse_folds.mean(axis=1)
    cv_se = mse_folds.std(axis=1) / np.sqrt(5)

    best_idx = np.argmin(cv_mean)
    lam_min = lambdas[best_idx]
    lam_1se = one_se_rule(lambdas, cv_mean, cv_se)

    plt.errorbar(np.log10(lambdas), cv_mean, yerr=cv_se, fmt='o-', ms=4)
    plt.axvline(np.log10(lam_min), color='red', ls='--', label='lambda_min')
    plt.axvline(np.log10(lam_1se), color='blue', ls='--', label='lambda_1SE')
    plt.xlabel('log10(lambda)')
    plt.ylabel('CV MSE')
    plt.legend()
    plt.show()
    ```

    The plot shows the characteristic U-shape with error bars. The 1SE rule selects a larger
    $\lambda$ (more regularization) than the minimum, leading to a sparser model. $\square$

---

**Exercise 5.** Prove that leave-one-out cross-validation (LOOCV) for ridge regression can be
computed in closed form via the formula
$\text{CV}_{\text{LOO}} = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2$,
where $H = X(X^\top X + \lambda I)^{-1}X^\top$ is the ridge hat matrix and $h_{ii}$ is its
$i$-th diagonal entry.

??? success "Solution to Exercise 5"

    When observation $i$ is removed, the ridge fit on the remaining $n-1$ observations gives
    prediction $\hat{y}_{(-i), i}$. By the Sherman--Morrison formula, removing row $i$ from
    $X$ and entry $i$ from $y$ and refitting is equivalent to

    $$
    \hat{y}_{(-i), i} = x_i^\top (X_{(-i)}^\top X_{(-i)} + \lambda I)^{-1} X_{(-i)}^\top y_{(-i)}.
    $$

    A standard result in linear models (extending the OLS LOOCV shortcut) shows that the
    LOO residual for ridge regression satisfies

    $$
    y_i - \hat{y}_{(-i),i} = \frac{y_i - \hat{y}_i}{1 - h_{ii}},
    $$

    where $\hat{y}_i = h_i^\top y$ is the full-data ridge prediction and $h_{ii}$ is the
    $(i,i)$ entry of the hat matrix $H = X(X^\top X + \lambda I)^{-1}X^\top$.

    Therefore

    $$
    \text{CV}_{\text{LOO}} = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2.
    $$

    This requires only a single fit of the full model plus the diagonal of $H$, making it
    $O(np^2)$ instead of $O(n^2 p^2)$ for the naive approach. $\square$
