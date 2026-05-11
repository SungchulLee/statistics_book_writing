# Cross-Validation Polynomial Model Selection

## Overview

This page demonstrates three resampling approaches for selecting the optimal polynomial degree in regression: the validation set approach, Leave-One-Out Cross-Validation (LOOCV), and $k$-fold cross-validation. Using synthetic data with a true quadratic relationship, we show that the validation set approach has high variability, LOOCV has low variance but high computational cost, and $k$-fold CV provides a practical balance.

## Mathematical Background

### Polynomial Regression

A degree-$d$ polynomial model is

$$
y_i = \beta_0 + \beta_1 x_i + \beta_2 x_i^2 + \cdots + \beta_d x_i^d + \varepsilon_i.
$$

Increasing $d$ reduces training error but may increase test error (overfitting). Model selection aims to find the $d$ that minimizes expected test error.

### Validation Set Approach

Split the data into training and validation sets. Fit each candidate model on training data and evaluate on validation data. The MSE on the validation set estimates test error. Drawback: the estimate depends heavily on the random split.

### Leave-One-Out Cross-Validation (LOOCV)

LOOCV uses $n$ folds, each leaving out one observation:

$$
\mathrm{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i^{(-i)})^2.
$$

For linear models, this can be computed efficiently using the hat matrix:

$$
\mathrm{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^n \left(\frac{e_i}{1 - h_{ii}}\right)^2.
$$

### k-Fold Cross-Validation

Partition the data into $K$ roughly equal folds, train on $K-1$ folds, and test on the held-out fold:

$$
\mathrm{CV}_{(K)} = \frac{1}{K}\sum_{k=1}^K \mathrm{MSE}_k.
$$

Typical choices are $K = 5$ or $K = 10$.

## Code

### Validation Set Approach

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 200
X = np.random.uniform(1, 10, n)
y = 5 + 2 * X - 0.3 * X**2 + np.random.normal(0, 2, n)
X_2d = X.reshape(-1, 1)

degrees = np.arange(1, 11)
n_validations = 20
val_mse_multiple = np.zeros((n_validations, len(degrees)))

for run in range(n_validations):
    val_size = int(0.2 * n)
    indices = np.random.permutation(n)
    train_idx, val_idx = indices[:-val_size], indices[-val_size:]
    for i, degree in enumerate(degrees):
        poly = PolynomialFeatures(degree)
        X_tr = poly.fit_transform(X_2d[train_idx])
        X_va = poly.transform(X_2d[val_idx])
        model = LinearRegression().fit(X_tr, y[train_idx])
        val_mse_multiple[run, i] = np.mean((y[val_idx] - model.predict(X_va)) ** 2)
```

### LOOCV

```python
from sklearn.model_selection import cross_val_score, LeaveOneOut

loo = LeaveOneOut()
loocv_mse = np.zeros(len(degrees))
for i, degree in enumerate(degrees):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X_2d)
    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y, cv=loo,
                             scoring='neg_mean_squared_error')
    loocv_mse[i] = -scores.mean()
```

### k-Fold CV

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
kfold_mse = np.zeros(len(degrees))
for i, degree in enumerate(degrees):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X_2d)
    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y, cv=kfold,
                             scoring='neg_mean_squared_error')
    kfold_mse[i] = -scores.mean()
```

## Interpretation

- **Validation set**: Fast but unstable. Different random splits can select different optimal degrees, making the choice unreliable for a single split.
- **LOOCV**: Nearly unbiased (each training set has $n-1$ observations) but computationally expensive ($n$ model fits per candidate) and can have high variance because the $n$ training sets are nearly identical.
- **10-fold CV**: A practical compromise. Each training set has $90\%$ of the data (modest bias), and the 10 folds provide some averaging to reduce variance.
- Since the true relationship is quadratic ($d = 2$), all three methods should select degree 2 or close to it. Higher degrees overfit, and the test error curve rises after the minimum.

## Exercises

**Exercise 1.** Use the hat matrix shortcut to compute LOOCV MSE for degree-2 polynomial regression without refitting $n$ models. Verify it matches the brute-force result.

??? success "Solution to Exercise 1"

    ```python
    poly = PolynomialFeatures(2)
    X_poly = poly.fit_transform(X_2d)
    model = LinearRegression().fit(X_poly, y)
    H = X_poly @ np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T
    e = y - model.predict(X_poly)
    h = np.diag(H)
    loocv_shortcut = np.mean((e / (1 - h)) ** 2)
    print(f"LOOCV (shortcut): {loocv_shortcut:.4f}")
    print(f"LOOCV (brute):    {loocv_mse[1]:.4f}")
    ```

    Both values match because the shortcut formula is algebraically equivalent to leave-one-out refitting for linear models. $\square$

---

**Exercise 2.** Increase $n$ from 200 to 1000. How does this affect the variability of the validation set approach and the gap between LOOCV and 10-fold CV?

??? success "Solution to Exercise 2"

    With $n = 1000$: (1) The validation set approach becomes more stable because both training and validation sets are larger, reducing the sensitivity to the particular split. (2) The gap between LOOCV and 10-fold CV narrows because the bias of 10-fold CV (from using $90\%$ of data for training) becomes negligible when $n$ is large. Both methods converge to similar estimates of the true test error. $\square$

---

**Exercise 3.** Plot the training MSE alongside the test MSE (from 10-fold CV) as a function of polynomial degree. Explain the bias-variance tradeoff visible in the plot.

??? success "Solution to Exercise 3"

    ```python
    train_mse = []
    for degree in degrees:
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X_2d)
        model = LinearRegression().fit(X_poly, y)
        train_mse.append(np.mean((y - model.predict(X_poly)) ** 2))
    ```

    Training MSE monotonically decreases with degree (more parameters always fit training data better). Test MSE initially decreases (reducing bias) then increases (increasing variance). The optimal degree is where test MSE is minimized. This U-shaped test error curve is the hallmark of the bias-variance tradeoff. $\square$

---

**Exercise 4.** Derive the formula $\mathrm{CV}_{(n)} = \frac{1}{n}\sum_i \left(\frac{e_i}{1 - h_{ii}}\right)^2$ from the Sherman-Morrison-Woodbury formula for rank-1 updates.

??? success "Solution to Exercise 4"

    Deleting observation $i$ is equivalent to a rank-1 update of $\mathbf{X}^\top\mathbf{X}$. By the Sherman-Morrison formula:

    $$
    (\mathbf{X}_{(-i)}^\top\mathbf{X}_{(-i)})^{-1} = (\mathbf{X}^\top\mathbf{X} - \mathbf{x}_i\mathbf{x}_i^\top)^{-1} = (\mathbf{X}^\top\mathbf{X})^{-1} + \frac{(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{x}_i\mathbf{x}_i^\top(\mathbf{X}^\top\mathbf{X})^{-1}}{1 - h_{ii}}.
    $$

    After algebra, the leave-one-out prediction error for observation $i$ simplifies to $y_i - \hat{y}_i^{(-i)} = e_i/(1 - h_{ii})$, giving the formula. $\square$

---

**Exercise 5.** Explain why $K$-fold CV has a bias-variance tradeoff in the choice of $K$. What are the extreme cases $K = 2$ and $K = n$?

??? success "Solution to Exercise 5"

    When $K$ is small (e.g., $K = 2$), each training set uses only $50\%$ of the data, introducing upward bias in the error estimate (models trained on less data perform worse). But the $K$ estimates are based on non-overlapping training sets, so their variance is low. When $K = n$ (LOOCV), each training set uses $n-1$ observations (minimal bias), but the $n$ training sets overlap almost completely, producing highly correlated estimates with potentially high variance. $K = 5$ or $K = 10$ balances these extremes. $\square$
