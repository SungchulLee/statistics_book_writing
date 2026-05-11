# Cross-Validation Methods Comparison

## Overview

This page compares three cross-validation strategies for model selection: the validation set approach, Leave-One-Out Cross-Validation (LOOCV), and $k$-fold cross-validation. We apply each method to polynomial regression of increasing degree on synthetic data, demonstrating the bias--variance tradeoff in CV estimates, computational cost differences, and the stability of the selected model across methods.

---

## Setup and Synthetic Data

We generate $n = 200$ observations from a nonlinear model:

$$
y_i = \sin(x_i) + 0.3\,x_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, 0.5^2), \quad x_i \sim \text{Uniform}(-3, 3)
$$

For model selection, we fit polynomial regression models of degree $d = 1, 2, \ldots, 10$ and compare the estimated test MSE from each CV method.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

np.random.seed(42)

def generate_data(n=200):
    x = np.random.uniform(-3, 3, n)
    y = np.sin(x) + 0.3 * x + np.random.normal(0, 0.5, n)
    return x.reshape(-1, 1), y

def poly_pipeline(degree):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lr", LinearRegression()),
    ])

X, y = generate_data(n=200)
```

---

## Method 1: Validation Set Approach

The simplest strategy splits the data into two halves: one for training, one for testing. The estimated test MSE is:

$$
\widehat{\text{MSE}} = \frac{1}{n_{\text{test}}} \sum_{i \in \text{test}} (y_i - \hat{y}_i)^2
$$

**Drawback**: The estimate has high variance because it depends on which observations land in the training versus test sets. Repeating with different random splits produces different answers.

```python
from sklearn.model_selection import cross_val_score

def validation_set_mse(X, y, degrees, n_splits=10):
    n = len(y)
    n_train = int(0.5 * n)
    all_mses = {d: [] for d in degrees}
    for _ in range(n_splits):
        perm = np.random.permutation(n)
        train_idx, test_idx = perm[:n_train], perm[n_train:]
        for d in degrees:
            model = poly_pipeline(d).fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            mse = np.mean((y[test_idx] - pred) ** 2)
            all_mses[d].append(mse)
    return all_mses

degrees = range(1, 11)
val_mses = validation_set_mse(X, y, degrees, n_splits=10)
val_means = [np.mean(val_mses[d]) for d in degrees]
best_val = int(np.argmin(val_means)) + 1
print(f"Best degree (validation set): {best_val}")
```

---

## Method 2: Leave-One-Out Cross-Validation (LOOCV)

LOOCV trains on $n - 1$ observations and tests on the remaining one, repeating for each observation:

$$
\text{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i^{(-i)})^2
$$

where $\hat{y}_i^{(-i)}$ is the prediction for observation $i$ from the model trained without observation $i$.

**Pros**: Nearly unbiased (training set is almost the full dataset). Deterministic (no randomness in the split).

**Cons**: Computationally expensive ($n$ model fits). High variance because the $n$ training sets overlap almost completely.

```python
from sklearn.model_selection import LeaveOneOut
import time

loo = LeaveOneOut()
loocv_mses = []
t0 = time.time()
for d in degrees:
    scores = cross_val_score(poly_pipeline(d), X, y,
                             cv=loo, scoring="neg_mean_squared_error")
    loocv_mses.append(-scores.mean())
loo_time = time.time() - t0
best_loo = int(np.argmin(loocv_mses)) + 1
print(f"Best degree (LOOCV): {best_loo}, time: {loo_time:.2f}s")
```

---

## Method 3: k-Fold Cross-Validation

The data are partitioned into $k$ equally sized folds. Each fold serves as the test set once:

$$
\text{CV}_{(k)} = \frac{1}{k}\sum_{j=1}^{k}\text{MSE}_j
$$

This provides a tradeoff between the validation set approach ($k = 2$) and LOOCV ($k = n$). Common choices are $k = 5$ and $k = 10$.

**Bias--variance tradeoff**: Smaller $k$ yields higher bias (less training data per fold) but lower variance. Larger $k$ yields lower bias but higher variance (more overlap between training sets).

```python
from sklearn.model_selection import KFold

for k in [5, 10]:
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    kf_mses = []
    for d in degrees:
        scores = cross_val_score(poly_pipeline(d), X, y,
                                 cv=kf, scoring="neg_mean_squared_error")
        kf_mses.append(-scores.mean())
    best_kf = int(np.argmin(kf_mses)) + 1
    print(f"Best degree ({k}-fold): {best_kf}")
```

---

## Interpretation

- All three methods typically agree on the optimal polynomial degree for this data (usually $d = 3$ or $d = 4$), since the true function $\sin(x) + 0.3x$ is well-approximated by a low-degree polynomial on $[-3, 3]$.
- The validation set approach shows the most variability across splits: the selected degree may change from one random split to the next.
- LOOCV is deterministic and nearly unbiased but computationally costly: it requires $n \times 10 = 2000$ model fits.
- 10-fold CV is a practical compromise: it requires only $10 \times 10 = 100$ fits and produces estimates close to LOOCV.
- For polynomial degrees well above the optimum ($d = 8, 9, 10$), the test MSE increases due to overfitting, consistent with the bias--variance tradeoff.

---

## Exercises

**Exercise 1.** For simple linear regression, LOOCV has a shortcut formula. Show that the LOOCV mean squared error equals:

$$
\text{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2
$$

where $h_{ii}$ is the $i$-th diagonal element of the hat matrix $H = X(X^\top X)^{-1}X^\top$. Why does this avoid refitting the model $n$ times?

??? success "Solution to Exercise 1"

    When observation $i$ is removed, the OLS fit changes. By the Sherman--Morrison--Woodbury formula, the leave-one-out residual is:

    $$
    y_i - \hat{y}_i^{(-i)} = \frac{y_i - \hat{y}_i}{1 - h_{ii}} = \frac{e_i}{1 - h_{ii}}
    $$

    where $e_i = y_i - \hat{y}_i$ is the ordinary residual and $h_{ii}$ is the leverage of observation $i$. This identity holds because removing one observation from the regression adjusts the prediction by exactly the factor $1/(1 - h_{ii})$.

    Squaring and averaging gives the LOOCV formula. The key advantage is that we only need to fit the model **once** on all $n$ observations and compute the hat matrix, rather than refitting $n$ times. This reduces the complexity from $O(n \cdot p^2 n)$ to $O(p^2 n)$ for linear models. $\square$

---

**Exercise 2.** Run a simulation study: generate 100 datasets from the same model. For each dataset, select the best polynomial degree using 5-fold CV, 10-fold CV, and LOOCV. Report the distribution of selected degrees for each method. Which method is most stable?

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    def poly_pipeline(d):
        return Pipeline([
            ("poly", PolynomialFeatures(degree=d, include_bias=False)),
            ("lr", LinearRegression()),
        ])

    degrees = range(1, 11)
    results = {"5-fold": [], "10-fold": [], "LOOCV": []}

    for trial in range(100):
        np.random.seed(trial)
        x = np.random.uniform(-3, 3, 200).reshape(-1, 1)
        y = np.sin(x.ravel()) + 0.3 * x.ravel() + np.random.normal(0, 0.5, 200)

        for name, cv in [("5-fold", KFold(5, shuffle=True, random_state=0)),
                         ("10-fold", KFold(10, shuffle=True, random_state=0)),
                         ("LOOCV", LeaveOneOut())]:
            mses = [-cross_val_score(poly_pipeline(d), x, y,
                     cv=cv, scoring="neg_mean_squared_error").mean()
                    for d in degrees]
            results[name].append(int(np.argmin(mses)) + 1)

    for name, degs in results.items():
        print(f"{name}: mode={max(set(degs),key=degs.count)}, "
              f"mean={np.mean(degs):.1f}, std={np.std(degs):.2f}")
    ```

    LOOCV tends to have the highest variance in the selected degree because the $n$ training sets overlap heavily, making the individual fold errors highly correlated. 10-fold CV is typically the most stable, with 5-fold close behind. $\square$

---

**Exercise 3.** Prove that the expected value of the LOOCV estimator is approximately unbiased for the true expected test MSE. Specifically, show that $E[\text{CV}_{(n)}] \approx E[\text{MSE}_{\text{test}}]$ where the test MSE is evaluated on an independent observation from the same distribution.

??? success "Solution to Exercise 3"

    Each term in the LOOCV sum trains on $n - 1$ observations and tests on 1:

    $$
    \text{CV}_{(n)} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i^{(-i)})^2
    $$

    Taking expectations:

    $$
    E[\text{CV}_{(n)}] = \frac{1}{n}\sum_{i=1}^{n} E\!\left[(y_i - \hat{y}_i^{(-i)})^2\right]
    $$

    Each term $E[(y_i - \hat{y}_i^{(-i)})^2]$ is the expected test error of a model trained on $n - 1$ observations and evaluated on an independent observation $i$ (which was held out). By symmetry (all observations are identically distributed), each term equals $\text{Err}_{n-1}$, the expected test error of a model trained on $n - 1$ points. Therefore:

    $$
    E[\text{CV}_{(n)}] = \text{Err}_{n-1}
    $$

    Since $\text{Err}_{n-1} \approx \text{Err}_n$ for large $n$, LOOCV is approximately unbiased for the test error of a model trained on $n$ observations. The small upward bias (because $n - 1 < n$) is negligible for moderate to large $n$. $\square$

---

**Exercise 4.** Implement repeated $k$-fold CV: run 10-fold CV five times with different random shuffles and average the MSE curves. Compare the stability of the selected degree to a single 10-fold CV run. Plot the averaged MSE curve with error bars.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    import matplotlib.pyplot as plt

    def poly_pipeline(d):
        return Pipeline([
            ("poly", PolynomialFeatures(degree=d, include_bias=False)),
            ("lr", LinearRegression()),
        ])

    np.random.seed(42)
    x = np.random.uniform(-3, 3, 200).reshape(-1, 1)
    y = np.sin(x.ravel()) + 0.3 * x.ravel() + np.random.normal(0, 0.5, 200)

    degrees = range(1, 11)
    all_mses = {d: [] for d in degrees}

    for rep in range(5):
        kf = KFold(10, shuffle=True, random_state=rep)
        for d in degrees:
            score = -cross_val_score(poly_pipeline(d), x, y,
                                     cv=kf, scoring="neg_mean_squared_error").mean()
            all_mses[d].append(score)

    means = [np.mean(all_mses[d]) for d in degrees]
    stds = [np.std(all_mses[d]) for d in degrees]
    best = int(np.argmin(means)) + 1
    print(f"Best degree (repeated 10-fold): {best}")

    plt.errorbar(list(degrees), means, yerr=stds, fmt='o-', capsize=4)
    plt.xlabel("Polynomial Degree")
    plt.ylabel("CV MSE")
    plt.title("Repeated 10-fold CV")
    plt.show()
    ```

    Repeated $k$-fold CV reduces the variance of the MSE estimate by averaging over multiple random partitions. The error bars show the variability across repetitions. Compared to a single 10-fold run, the averaged curve is smoother and the selected degree is more reliable. $\square$

---

**Exercise 5.** Discuss the computational cost of each CV method in terms of model fits. For polynomial regression of degree $d$ on $n$ observations with $k$-fold CV, express the total number of model fits as a function of the number of candidate degrees $D$, and compare the three methods.

??? success "Solution to Exercise 5"

    Let $D$ be the number of candidate degrees to evaluate.

    - **Validation set** (single split): $D$ model fits. With $m$ repeated splits: $mD$ fits. For the example, $m = 10$, $D = 10$, giving 100 fits.

    - **$k$-fold CV**: $kD$ model fits. For 10-fold with $D = 10$: $100$ fits. For 5-fold: $50$ fits.

    - **LOOCV**: $nD$ model fits. For $n = 200$, $D = 10$: $2000$ fits.

    The cost per model fit is $O(nd^2)$ for polynomial regression (forming the polynomial features and solving the least-squares system). Therefore the total costs are:

    | Method | Total fits | Total cost |
    |---|---|---|
    | Validation set ($m$ splits) | $mD$ | $O(mDnd^2)$ |
    | $k$-fold CV | $kD$ | $O(kDnd^2)$ |
    | LOOCV | $nD$ | $O(nDnd^2) = O(n^2Dd^2)$ |

    LOOCV is $n/k$ times more expensive than $k$-fold CV. For $n = 200$ and $k = 10$, LOOCV costs 20 times more. For linear models, the shortcut formula (Exercise 1) reduces LOOCV to a single fit, but for nonlinear or non-least-squares models, the full $n$ refits are required. In practice, 10-fold CV offers the best balance of accuracy and computational efficiency. $\square$
