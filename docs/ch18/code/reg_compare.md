# Ridge, Lasso, and Elastic Net Comparison

## Overview

This page brings Ridge, Lasso, and Elastic Net together in a unified comparison. Using
synthetic data with correlated predictors, we fit all three methods with cross-validated tuning
parameters and examine their coefficient estimates, regularization paths, and bias--variance
behavior. The goal is to build intuition for when each method is preferable.

## Unified Formulation

All three methods can be expressed as special cases of

$$
\hat{\beta} = \arg\min_{\beta} \left\{ \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \left[\alpha \|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|_2^2\right] \right\},
$$

with the following correspondence:

| Method | $\alpha$ | Penalty | Sparsity |
|---|---|---|---|
| Ridge | 0 | $\frac{\lambda}{2}\|\beta\|_2^2$ | No |
| Lasso | 1 | $\lambda\|\beta\|_1$ | Yes |
| Elastic Net | $(0,1)$ | $\lambda[\alpha\|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|_2^2]$ | Yes |

## Code: Data Generation with Multicollinearity

The script generates $n = 200$ observations with $p = 20$ predictors that have a Toeplitz
correlation structure $\Sigma_{ij} = \rho^{|i-j|}$ with $\rho = 0.8$.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def generate_data(n=200, p=20, s=5, rho=0.8, noise=1.0):
    """
    Generate regression data with correlated predictors.
    - n: samples, p: predictors, s: true nonzero coefficients
    - rho: correlation between adjacent predictors
    """
    Sigma = np.array([[rho**abs(i-j) for j in range(p)] for i in range(p)])
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T

    beta_true = np.zeros(p)
    beta_true[:s] = np.array([3, -2, 1.5, -1, 0.5])

    y = X @ beta_true + noise * np.random.randn(n)
    return X, y, beta_true

np.random.seed(42)
X, y, beta_true = generate_data()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Only 5 of the 20 coefficients are nonzero, creating a sparse ground truth.

## Code: Cross-Validated Fitting

Each method uses built-in CV from scikit-learn to select the optimal $\lambda$ (and $\alpha$
for Elastic Net).

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

alphas = np.logspace(-4, 2, 100)

ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_scaled, y)

lasso_cv = LassoCV(n_alphas=100, cv=5, max_iter=10000)
lasso_cv.fit(X_scaled, y)

enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
    n_alphas=100, cv=5, max_iter=10000
)
enet_cv.fit(X_scaled, y)
```

## Coefficient Comparison

A bar chart comparing the true coefficients with the three estimates reveals:

- **Ridge** keeps all 20 coefficients nonzero, shrinking irrelevant ones but not to zero.
- **Lasso** sets many coefficients exactly to zero, closely recovering the true sparsity
  pattern.
- **Elastic Net** behaves similarly to Lasso but may retain a few more correlated predictors.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
p = X_scaled.shape[1]

for ax, name, coefs in [
    (axes[0], "True", beta_true),
    (axes[1], "Ridge", ridge_cv.coef_),
    (axes[2], "Lasso", lasso_cv.coef_),
    (axes[3], "Elastic Net", enet_cv.coef_),
]:
    colors = ['#d32f2f' if abs(c) > 1e-6 else '#90a4ae' for c in coefs]
    ax.bar(range(p), coefs, color=colors, edgecolor='black', linewidth=0.3)
    ax.set_title(name)
    ax.set_xlabel("Feature index")
    ax.axhline(0, color='black', linewidth=0.5)

axes[0].set_ylabel("Coefficient value")
plt.tight_layout()
plt.show()
```

## Regularization Paths

Plotting coefficient magnitude as a function of $\log_{10}(\lambda)$ highlights the shrinkage
behavior:

- **Ridge path:** Coefficients shrink smoothly and continuously toward zero; none ever reach
  exactly zero.
- **Lasso path:** Coefficients shrink and are set to exactly zero at different $\lambda$
  thresholds, producing a piecewise-linear path.

```python
from sklearn.linear_model import Ridge, Lasso

alphas_path = np.logspace(-3, 3, 200)

# Ridge path
ridge_coefs = []
for a in alphas_path:
    model = Ridge(alpha=a).fit(X_scaled, y)
    ridge_coefs.append(model.coef_.copy())
ridge_coefs = np.array(ridge_coefs)

# Lasso path
lasso_coefs = []
alphas_lasso = np.logspace(-4, 1, 200)
for a in alphas_lasso:
    model = Lasso(alpha=a, max_iter=10000).fit(X_scaled, y)
    lasso_coefs.append(model.coef_.copy())
lasso_coefs = np.array(lasso_coefs)
```

## Shrinkage Operators

In the orthonormal design case ($X^\top X = I$), the three methods correspond to distinct
shrinkage operators applied to the OLS estimate $\hat{\beta}^{\text{OLS}}$:

$$
\hat{\beta}_j^{\text{Ridge}} = \frac{\hat{\beta}_j^{\text{OLS}}}{1 + \lambda}, \qquad
\hat{\beta}_j^{\text{Lasso}} = S(\hat{\beta}_j^{\text{OLS}}, \lambda), \qquad
\hat{\beta}_j^{\text{Hard}} = \hat{\beta}_j^{\text{OLS}} \cdot \mathbf{1}(|\hat{\beta}_j^{\text{OLS}}| > \lambda).
$$

```python
def plot_shrinkage_operators(lam=1.0):
    z = np.linspace(-4, 4, 500)
    ridge = z / (1 + lam)
    lasso = np.sign(z) * np.maximum(np.abs(z) - lam, 0)
    hard = z * (np.abs(z) > lam)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(z, z, 'k--', alpha=0.3, label="OLS (no shrinkage)")
    ax.plot(z, ridge, linewidth=2, label=f"Ridge")
    ax.plot(z, lasso, linewidth=2, label=f"Lasso")
    ax.plot(z, hard, linewidth=2, label=f"Hard threshold")
    ax.set_xlabel("OLS estimate")
    ax.set_ylabel("Regularized estimate")
    ax.set_title("Shrinkage Operators (Orthonormal Design)")
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_shrinkage_operators()
```

## Bias--Variance Tradeoff

A simulation study across 500 replications reveals:

- **Ridge** has a smooth, U-shaped MSE curve. It performs best when many coefficients are small
  but nonzero.
- **Lasso** can achieve lower MSE in truly sparse settings because its variable selection
  eliminates noise dimensions.
- The optimal $\lambda$ for each method balances bias (underfitting due to excessive shrinkage)
  against variance (overfitting due to insufficient shrinkage).

## Interpretation

| Criterion | Ridge | Lasso | Elastic Net |
|---|---|---|---|
| Sparsity | No | Yes | Yes |
| Unique solution | Always | Only if $X$ full rank | Always ($\alpha < 1$) |
| Correlated groups | Retains all | Selects one | Selects group |
| Computation | Closed-form | Coordinate descent | Coordinate descent |
| Best when | Dense signals | Sparse signals | Sparse + correlated |

## Exercises

**Exercise 1.** For the orthonormal design case ($X^\top X = I_p$), derive the three shrinkage
formulas (Ridge, Lasso, hard thresholding) and sketch them on a single plot.

??? success "Solution to Exercise 1"

    When $X^\top X = I_p$, the OLS estimator is $\hat{\beta}^{\text{OLS}} = X^\top y$. Each
    penalty leads to an independent univariate problem:

    **Ridge:** $\min_{\beta_j} \frac{1}{2}(\hat{\beta}_j^{\text{OLS}} - \beta_j)^2 + \frac{\lambda}{2}\beta_j^2$. Differentiating:
    $(1+\lambda)\beta_j = \hat{\beta}_j^{\text{OLS}}$, so
    $\hat{\beta}_j^{\text{Ridge}} = \hat{\beta}_j^{\text{OLS}} / (1+\lambda)$.

    **Lasso:** $\min_{\beta_j} \frac{1}{2}(\hat{\beta}_j^{\text{OLS}} - \beta_j)^2 + \lambda|\beta_j|$. By the proximal operator:
    $\hat{\beta}_j^{\text{Lasso}} = S(\hat{\beta}_j^{\text{OLS}}, \lambda)$.

    **Hard thresholding:** $\min_{\beta_j} \frac{1}{2}(\hat{\beta}_j^{\text{OLS}} - \beta_j)^2 + \lambda \cdot \mathbf{1}(\beta_j \ne 0)$. The solution either keeps the OLS value
    (cost $\lambda$) or sets to zero (cost $\frac{1}{2}(\hat{\beta}_j^{\text{OLS}})^2$):
    $\hat{\beta}_j^{\text{Hard}} = \hat{\beta}_j^{\text{OLS}} \cdot \mathbf{1}(|\hat{\beta}_j^{\text{OLS}}| > \sqrt{2\lambda})$.

    The plot shows Ridge as a line through the origin with slope $1/(1+\lambda)$, Lasso as a
    piecewise-linear function with a dead zone $[-\lambda, \lambda]$, and hard thresholding as
    the identity outside $[-\sqrt{2\lambda}, \sqrt{2\lambda}]$ and zero inside. $\square$

---

**Exercise 2.** Generate data with $p = 20$, $\rho = 0.9$, and 5 true nonzero coefficients.
Fit all three methods with CV and compare the number of nonzero coefficients selected by each.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    X, y, beta_true = generate_data(n=200, p=20, s=5, rho=0.9)
    X_s = StandardScaler().fit_transform(X)

    ridge = RidgeCV(alphas=np.logspace(-4, 2, 100), cv=5).fit(X_s, y)
    lasso = LassoCV(n_alphas=100, cv=5, max_iter=10000).fit(X_s, y)
    enet = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9], n_alphas=100, cv=5
    ).fit(X_s, y)

    for name, m in [("Ridge", ridge), ("Lasso", lasso), ("Elastic Net", enet)]:
        nz = np.sum(np.abs(m.coef_) > 1e-6)
        print(f"{name}: {nz} nonzero coefficients")
    ```

    Typical results: Ridge keeps all 20, Lasso selects roughly 5--8, Elastic Net selects
    roughly 6--10 (retaining correlated partners). $\square$

---

**Exercise 3.** Explain why standardizing predictors before applying regularization is
important. Give a concrete numerical example where failing to standardize leads to a misleading
result.

??? success "Solution to Exercise 3"

    The penalties $\|\beta\|_1$ and $\|\beta\|_2^2$ treat all coefficients equally, but the
    OLS estimate $\hat{\beta}_j$ depends on the scale of $x_j$. If $x_1$ is measured in meters
    and $x_2$ in millimeters, then $\hat{\beta}_1$ is 1000 times larger than $\hat{\beta}_2$
    for the same physical effect. The penalty would then shrink $\hat{\beta}_1$ much more
    aggressively, effectively penalizing the unit choice rather than the importance of the
    predictor.

    **Example:** Let $x_1 \in [0, 1]$ and $x_2 \in [0, 1000]$, with $y = x_1 + x_2/1000 + \varepsilon$. Without standardization, $\hat{\beta}_1 \approx 1$ and
    $\hat{\beta}_2 \approx 0.001$. Lasso with moderate $\lambda$ would set $\hat{\beta}_2$ to
    zero while keeping $\hat{\beta}_1$, despite both predictors being equally important.
    After standardization, both coefficients are comparable in magnitude and Lasso treats them
    symmetrically. $\square$

---

**Exercise 4.** Using the bias--variance simulation framework from the code, determine at which
value of $\lambda$ the ridge MSE and lasso MSE are minimized. Which method achieves a lower
minimum MSE in this (sparse, correlated) setting?

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import StandardScaler

    np.random.seed(0)
    alphas_test = np.logspace(-3, 2, 30)
    n_sim = 200
    _, _, beta_true_bv = generate_data(n=2, p=20, s=5)

    ridge_mse = {a: [] for a in alphas_test}
    lasso_mse = {a: [] for a in alphas_test}

    for _ in range(n_sim):
        X_sim, y_sim, _ = generate_data(n=100, p=20, s=5)
        X_sim = StandardScaler().fit_transform(X_sim)
        for a in alphas_test:
            r = Ridge(alpha=a).fit(X_sim, y_sim)
            l = Lasso(alpha=a, max_iter=5000).fit(X_sim, y_sim)
            ridge_mse[a].append(np.sum((r.coef_ - beta_true_bv)**2))
            lasso_mse[a].append(np.sum((l.coef_ - beta_true_bv)**2))

    ridge_avg = {a: np.mean(v) for a, v in ridge_mse.items()}
    lasso_avg = {a: np.mean(v) for a, v in lasso_mse.items()}

    best_ridge = min(ridge_avg, key=ridge_avg.get)
    best_lasso = min(lasso_avg, key=lasso_avg.get)
    print(f"Ridge best lambda: {best_ridge:.4f}, MSE: {ridge_avg[best_ridge]:.4f}")
    print(f"Lasso best lambda: {best_lasso:.4f}, MSE: {lasso_avg[best_lasso]:.4f}")
    ```

    In sparse settings, Lasso typically achieves a lower minimum MSE because it eliminates the
    15 irrelevant dimensions, reducing variance without incurring much bias. $\square$

---

**Exercise 5.** Prove that for the constrained formulation
$\min \|y - X\beta\|_2^2$ subject to $\alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2 \le t$,
the constraint region is convex for all $\alpha \in [0,1]$.

??? success "Solution to Exercise 5"

    Define $C = \{\beta : \alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2 \le t\}$. We need to
    show that for any $\beta_1, \beta_2 \in C$ and $\theta \in [0,1]$,
    $\beta_\theta = \theta\beta_1 + (1-\theta)\beta_2 \in C$.

    The function $f(\beta) = \alpha\|\beta\|_1 + (1-\alpha)\|\beta\|_2^2$ is a non-negative
    combination of two convex functions:

    - $\|\beta\|_1$ is convex (it is a norm).
    - $\|\beta\|_2^2$ is convex (its Hessian is $2I$, which is positive semidefinite).

    Therefore $f$ is convex. By convexity:

    $$
    f(\beta_\theta) \le \theta f(\beta_1) + (1-\theta)f(\beta_2) \le \theta t + (1-\theta)t = t.
    $$

    Hence $\beta_\theta \in C$, and $C$ is convex. $\square$
