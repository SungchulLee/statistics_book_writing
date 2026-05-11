# Model Selection Comparison

## Overview

This page demonstrates model selection using three criteria: Akaike Information Criterion (AIC), Bayesian Information Criterion (BIC), and cross-validation (CV). Using synthetic data with 8 predictors (only 3 truly relevant), we perform forward selection and compare which model size each criterion identifies as optimal.

## Mathematical Background

### Akaike Information Criterion (AIC)

$$
\mathrm{AIC} = n \ln\!\left(\frac{\mathrm{RSS}}{n}\right) + 2k,
$$

where $k$ is the number of estimated parameters (including intercept). AIC estimates the out-of-sample prediction error and penalizes complexity with a factor of $2$ per parameter.

### Bayesian Information Criterion (BIC)

$$
\mathrm{BIC} = n \ln\!\left(\frac{\mathrm{RSS}}{n}\right) + k \ln(n).
$$

BIC uses a penalty of $\ln(n)$ per parameter, which exceeds 2 when $n > e^2 \approx 7.4$. BIC therefore favors smaller models for moderate to large sample sizes.

### Cross-Validation MSE

The $K$-fold CV estimate of prediction error is

$$
\mathrm{CV}(K) = \frac{1}{K}\sum_{k=1}^{K} \mathrm{MSE}_k, \qquad \mathrm{MSE}_k = \frac{1}{|V_k|}\sum_{i \in V_k}(y_i - \hat{y}_i^{(-k)})^2,
$$

where $\hat{y}_i^{(-k)}$ is the prediction for observation $i$ from the model trained without fold $k$.

## Code

### Information Criteria Functions

```python
import numpy as np

def aic(n, rss, k):
    return n * np.log(rss / n) + 2 * k

def bic(n, rss, k):
    return n * np.log(rss / n) + k * np.log(n)
```

### Cross-Validation MSE

```python
def cv_mse(X, y, folds=5):
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // folds
    mses = []
    for k in range(folds):
        val_idx = indices[k * fold_size:(k + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        beta = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
        pred = X_va @ beta
        mses.append(np.mean((y_va - pred) ** 2))
    return np.mean(mses)
```

### Forward Selection

```python
np.random.seed(42)
n, p_total = 200, 8
X_raw = np.random.randn(n, p_total)
beta_true = np.array([3.0, -2.0, 1.5, 0, 0, 0, 0, 0])
y = X_raw @ beta_true + np.random.randn(n) * 2

remaining = list(range(p_total))
selected = []
aic_history, bic_history = [], []

for step in range(p_total):
    best_score, best_j = np.inf, None
    for j in remaining:
        cols = selected + [j]
        X_cand = np.column_stack([np.ones(n), X_raw[:, cols]])
        beta = np.linalg.lstsq(X_cand, y, rcond=None)[0]
        rss = np.sum((y - X_cand @ beta) ** 2)
        score = aic(n, rss, len(cols) + 1)
        if score < best_score:
            best_score, best_j = score, j
    selected.append(best_j)
    remaining.remove(best_j)

    X_sel = np.column_stack([np.ones(n), X_raw[:, selected]])
    beta = np.linalg.lstsq(X_sel, y, rcond=None)[0]
    rss = np.sum((y - X_sel @ beta) ** 2)
    k = len(selected) + 1
    aic_history.append(aic(n, rss, k))
    bic_history.append(bic(n, rss, k))
```

## Interpretation

- **AIC** tends to select slightly larger models because its penalty ($2k$) is relatively mild. It targets prediction accuracy.
- **BIC** tends to select smaller models because $k \ln(n)$ grows with sample size. It is consistent, meaning it will select the true model as $n \to \infty$ (if the true model is among the candidates).
- **Cross-validation** directly estimates out-of-sample prediction error without relying on asymptotic theory. It is computationally more expensive but makes fewer distributional assumptions.
- In this example with 3 truly relevant predictors, all three methods should identify a model size near 3, confirming that the noise predictors are correctly excluded.

## Exercises

**Exercise 1.** Run the forward selection procedure using BIC instead of AIC to determine the next predictor at each step. Does the order of selected predictors change?

??? success "Solution to Exercise 1"

    Replace `score = aic(n, rss, len(cols) + 1)` with `score = bic(n, rss, len(cols) + 1)` in the inner loop. The order of selection often remains the same (the most important predictors are selected first regardless), but the optimal stopping point changes: BIC typically stops at fewer predictors. $\square$

---

**Exercise 2.** Increase the noise level from $\sigma = 2$ to $\sigma = 5$. How does this affect the optimal model size selected by each criterion?

??? success "Solution to Exercise 2"

    With more noise, the signal-to-noise ratio decreases. The RSS differences between models with and without the true predictors become smaller relative to the total RSS. AIC and BIC may select fewer predictors (possibly 1 or 2 instead of 3), and CV MSE curves become flatter with less distinct minima. The noise predictors become harder to distinguish from true predictors. $\square$

---

**Exercise 3.** Implement 10-fold CV and compare the results to 5-fold CV. Discuss the bias-variance tradeoff in the choice of $K$.

??? success "Solution to Exercise 3"

    ```python
    cv5 = [cv_mse(np.column_stack([np.ones(n), X_raw[:, selected[:s]]]), y, folds=5)
           for s in range(1, p_total + 1)]
    cv10 = [cv_mse(np.column_stack([np.ones(n), X_raw[:, selected[:s]]]), y, folds=10)
            for s in range(1, p_total + 1)]
    ```

    Larger $K$ means each training set is closer to size $n$ (less bias), but the folds overlap more (higher variance). $K = 5$ has slightly more bias but lower variance; $K = 10$ has less bias but more variance. The extreme case $K = n$ (LOOCV) is nearly unbiased but can have high variance. $\square$

---

**Exercise 4.** Derive the BIC penalty $k\ln(n)$ from a Bayesian model comparison perspective. Why does the penalty depend on $n$?

??? success "Solution to Exercise 4"

    In Bayesian model selection, we compute the marginal likelihood $p(\mathbf{y} \mid M)$ by integrating over the parameter space. Using a Laplace approximation for the integral yields

    $$
    \ln p(\mathbf{y} \mid M) \approx \ln p(\mathbf{y} \mid \hat{\boldsymbol{\theta}}, M) - \frac{k}{2}\ln(n) + O(1).
    $$

    Multiplying by $-2$ gives $\mathrm{BIC} = -2\ln L + k\ln(n)$. The penalty depends on $n$ because with more data, the posterior concentrates more tightly, and the "volume" cost of additional parameters scales logarithmically with $n$. $\square$

---

**Exercise 5.** Suppose the true model has 3 relevant predictors. Prove that BIC is model-selection consistent, i.e., $P(\text{BIC selects the true model}) \to 1$ as $n \to \infty$, while AIC is not.

??? success "Solution to Exercise 5"

    For AIC, the penalty for adding one extra parameter is $2$, independent of $n$. As $n \to \infty$, the reduction in $n\ln(\mathrm{RSS}/n)$ from adding an irrelevant predictor converges to a $\chi^2_1$ random variable (which has mean 1), so there is a nonzero probability of exceeding 2. Hence AIC overfits asymptotically.

    For BIC, the penalty is $\ln(n) \to \infty$, while the improvement from adding an irrelevant predictor remains bounded (converges to $\chi^2_1$). Therefore, for large enough $n$, the penalty dominates, and irrelevant predictors are excluded with probability approaching 1. This proves BIC consistency. $\square$
