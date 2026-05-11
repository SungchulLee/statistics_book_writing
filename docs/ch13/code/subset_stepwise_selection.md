# Subset and Stepwise Selection

## Overview

This page demonstrates three feature selection strategies for linear regression: best subset selection, forward stepwise selection, and backward stepwise selection. Using synthetic data with 8 predictors (4 truly relevant, 4 noise), we compare these methods on training RSS, validation RSS, and selected feature sets to illustrate the tradeoffs between exhaustive search and greedy algorithms.

## Mathematical Background

### Best Subset Selection

For each model size $k = 1, \ldots, p$, best subset selection evaluates all $\binom{p}{k}$ possible $k$-variable models and selects the one with the lowest training RSS. The total number of models evaluated is $\sum_{k=1}^p \binom{p}{k} = 2^p - 1$, making this approach computationally feasible only for small $p$ (typically $p \leq 20$).

### Forward Stepwise Selection

Starting from the null model (intercept only), forward selection greedily adds the predictor that produces the greatest reduction in RSS:

1. Begin with $\mathcal{S} = \emptyset$
2. For $k = 1, \ldots, p$: find $j^* = \arg\min_{j \notin \mathcal{S}} \mathrm{RSS}(\mathcal{S} \cup \{j\})$ and set $\mathcal{S} \leftarrow \mathcal{S} \cup \{j^*\}$

This evaluates $p + (p-1) + \cdots + 1 = p(p+1)/2$ models, far fewer than $2^p$.

### Backward Stepwise Selection

Starting from the full model, backward selection greedily removes the predictor whose removal increases RSS the least:

1. Begin with $\mathcal{S} = \{1, \ldots, p\}$
2. For $k = p-1, \ldots, 1$: find $j^* = \arg\min_{j \in \mathcal{S}} \mathrm{RSS}(\mathcal{S} \setminus \{j\})$ and set $\mathcal{S} \leftarrow \mathcal{S} \setminus \{j^*\}$

### Selecting the Optimal Size

The optimal $k$ is chosen by minimizing a criterion on held-out data (validation RSS, CV, AIC, or BIC), since training RSS always decreases with $k$.

## Code

### Data Generation

```python
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n, p = 200, 8
X = np.random.randn(n, p)
true_beta = np.array([3.0, 1.5, -2.0, 0.8, 0, 0, 0, 0])
y = X @ true_beta + np.random.normal(0, 2, n)
names = [f"x{i+1}" for i in range(p)]
```

### Best Subset Selection

```python
from itertools import combinations

def best_subset(X, y, max_k=None):
    n, p = X.shape
    if max_k is None:
        max_k = p
    results = {}
    for k in range(1, max_k + 1):
        best_rss, best_features = np.inf, None
        for combo in combinations(range(p), k):
            model = LinearRegression().fit(X[:, combo], y)
            rss = np.sum((y - model.predict(X[:, combo])) ** 2)
            if rss < best_rss:
                best_rss, best_features = rss, combo
        results[k] = {"features": best_features, "rss": best_rss}
    return results
```

### Forward Stepwise Selection

```python
def forward_stepwise(X, y):
    n, p = X.shape
    selected, remaining = [], list(range(p))
    results = {}
    for k in range(1, p + 1):
        best_rss, best_feature = np.inf, None
        for f in remaining:
            trial = selected + [f]
            model = LinearRegression().fit(X[:, trial], y)
            rss = np.sum((y - model.predict(X[:, trial])) ** 2)
            if rss < best_rss:
                best_rss, best_feature = rss, f
        selected.append(best_feature)
        remaining.remove(best_feature)
        results[k] = {"features": tuple(selected), "rss": best_rss}
    return results
```

### Backward Stepwise Selection

```python
def backward_stepwise(X, y):
    n, p = X.shape
    current = list(range(p))
    results = {}
    model = LinearRegression().fit(X, y)
    results[p] = {"features": tuple(current),
                  "rss": np.sum((y - model.predict(X)) ** 2)}
    for k in range(p - 1, 0, -1):
        best_rss, best_remove = np.inf, None
        for f in current:
            trial = [x for x in current if x != f]
            model = LinearRegression().fit(X[:, trial], y)
            rss = np.sum((y - model.predict(X[:, trial])) ** 2)
            if rss < best_rss:
                best_rss, best_remove = rss, f
        current.remove(best_remove)
        results[k] = {"features": tuple(current), "rss": best_rss}
    return results
```

## Interpretation

- **Best subset** is guaranteed to find the globally best model of each size but is computationally intractable for $p > 20$ (exponential growth).
- **Forward stepwise** is a greedy approximation that may miss the globally optimal model but runs in $O(p^2)$ time. It cannot revisit a predictor once included.
- **Backward stepwise** starts from the full model and may produce different solutions from forward selection. It requires $n > p$ to fit the initial full model.
- All three methods agree on the optimal $k$ when the signal is strong and the true model is nested in the search path.
- **Validation RSS** is essential: training RSS always decreases with $k$, so it cannot be used to select model size.

## Exercises

**Exercise 1.** Run all three methods and compare the features selected at the optimal $k$ (determined by validation RSS). Do they all identify the same 4 true predictors?

??? success "Solution to Exercise 1"

    ```python
    n_train = 140
    X_tr, X_val = X[:n_train], X[n_train:]
    y_tr, y_val = y[:n_train], y[n_train:]

    best = best_subset(X_tr, y_tr)
    fwd = forward_stepwise(X_tr, y_tr)
    bwd = backward_stepwise(X_tr, y_tr)

    # Find optimal k by validation RSS for each method
    for method_name, res in [("Best", best), ("Fwd", fwd), ("Bwd", bwd)]:
        val_rss = [np.sum((y_val - LinearRegression().fit(X_tr[:, res[k]["features"]],
                   y_tr).predict(X_val[:, res[k]["features"]])) ** 2)
                   for k in range(1, 9)]
        opt_k = np.argmin(val_rss) + 1
        print(f"{method_name}: k={opt_k}, features={res[opt_k]['features']}")
    ```

    With a strong signal, all three methods typically identify $k = 4$ with features $\{x_1, x_2, x_3, x_4\}$. $\square$

---

**Exercise 2.** Increase $p$ from 8 to 20 with the same 4 true predictors. Can best subset still be computed? How does forward selection perform?

??? success "Solution to Exercise 2"

    With $p = 20$, best subset requires evaluating $2^{20} - 1 = 1{,}048{,}575$ models, which is computationally expensive but still feasible. With $p = 30$ or more, it becomes impractical. Forward selection remains fast ($O(p^2)$ models) and still identifies the true predictors if their effects are sufficiently strong. The greedy nature of forward selection means it may select a noise predictor early if it happens to correlate with the response, but this is unlikely with strong true effects. $\square$

---

**Exercise 3.** Construct an example where forward and backward stepwise select different features at the same $k$. What property of the data causes this discrepancy?

??? success "Solution to Exercise 3"

    This occurs when predictors are correlated. For instance, if $x_5$ is moderately correlated with both $x_1$ and $x_2$, forward selection might add $x_5$ early (before $x_2$) because $x_5$ captures some signal. Backward selection starts with all predictors and may remove $x_5$ before $x_2$ because $x_2$ is more useful given the full model. The discrepancy arises from the greedy, path-dependent nature of both algorithms: the order of addition/removal depends on what other predictors are already in the model. $\square$

---

**Exercise 4.** Prove that best subset selection with $k$ predictors has training RSS less than or equal to forward stepwise selection with $k$ predictors.

??? success "Solution to Exercise 4"

    Best subset searches over all $\binom{p}{k}$ subsets and picks the one with the lowest RSS. Forward selection produces one specific $k$-variable model via a greedy path. Since the forward selection model is one of the $\binom{p}{k}$ subsets considered by best subset, best subset's RSS is at most as large:

    $$
    \mathrm{RSS}_{\text{best}}(k) = \min_{\mathcal{S}: |\mathcal{S}|=k} \mathrm{RSS}(\mathcal{S}) \leq \mathrm{RSS}(\mathcal{S}_{\text{fwd}}(k)) = \mathrm{RSS}_{\text{fwd}}(k).
    $$

    Equality holds when the greedy path happens to find the global optimum. $\square$

---

**Exercise 5.** Implement model selection using AIC instead of validation RSS. Determine the optimal $k$ using AIC and compare to the validation approach.

??? success "Solution to Exercise 5"

    ```python
    def aic(n, rss, k):
        return n * np.log(rss / n) + 2 * (k + 1)  # +1 for intercept

    fwd = forward_stepwise(X_tr, y_tr)
    aic_vals = []
    for k in range(1, 9):
        aic_vals.append(aic(n_train, fwd[k]['rss'], k))
    opt_k_aic = np.argmin(aic_vals) + 1
    print(f"Optimal k (AIC): {opt_k_aic}")
    ```

    AIC typically selects $k = 4$ (the true model size), though it may occasionally favor $k = 5$ due to the mild penalty. Validation RSS is less biased but more variable (depends on the specific split). Both methods generally agree when the signal is clear. $\square$
