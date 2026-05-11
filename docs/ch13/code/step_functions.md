# Step Functions

## Overview

This page demonstrates step function regression (piecewise constant fitting) as a nonlinear modeling technique. Using the King County housing dataset, we create indicator variables from binned predictors, fit step function models, compare different numbers of bins, and contrast step functions with linear, polynomial, and spline alternatives. We also show how step functions apply to logistic regression for classification.

## Mathematical Background

A step function partitions the range of a predictor $x$ into $K$ bins using cutpoints $c_1 < c_2 < \cdots < c_{K-1}$, and creates indicator variables:

$$
C_k(x) = \mathbf{1}(c_{k-1} < x \leq c_k), \qquad k = 1, \ldots, K.
$$

The step function regression model is

$$
y_i = \beta_0 + \beta_1 C_1(x_i) + \beta_2 C_2(x_i) + \cdots + \beta_K C_K(x_i) + \varepsilon_i.
$$

This is equivalent to fitting a separate constant (the bin mean) within each interval. The fitted value for any $x$ in bin $k$ is simply $\bar{y}_k$ (the average response in that bin when all indicators are included without dropping one).

### Choosing Breakpoints

Breakpoints can be set based on:

- **Equal-width bins**: divide the range into $K$ equal intervals
- **Quantile bins** (`pd.qcut`): each bin contains approximately the same number of observations
- **Domain knowledge**: place cuts at meaningful thresholds

## Code

### Creating Step Functions with pd.cut

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

age = 2024 - house['YrBuilt'].values
price = house['AdjSalePrice'].values
df = pd.DataFrame({'age': age, 'price': price})

# Define breakpoints
knots = [0, 20, 40, 60, 80, 150]
df['age_bin'] = pd.cut(df['age'], bins=knots, include_lowest=True)

# Create dummy variables
df_dummies = pd.get_dummies(df['age_bin'], drop_first=False)

# Fit step function regression
X_step = df_dummies.values
step_model = LinearRegression()
step_model.fit(X_step, df['price'])
```

### Comparing Different Numbers of Bins

```python
results = []
for n_bins in [3, 4, 5, 6, 8, 10]:
    df['bin_temp'] = pd.qcut(df['age'], q=n_bins, duplicates='drop')
    X_temp = pd.get_dummies(df['bin_temp'], drop_first=False).values
    model = LinearRegression().fit(X_temp, df['price'])
    pred = model.predict(X_temp)
    r2 = r2_score(df['price'], pred)
    rmse = np.sqrt(mean_squared_error(df['price'], pred))
    results.append({'n_bins': n_bins, 'R2': r2, 'RMSE': rmse})
```

### Comparison with Other Methods

```python
# Linear
linear_model = LinearRegression()
linear_model.fit(df[['age']].values, df['price'])

# Polynomial (degree 3)
X_poly = np.column_stack([df['age'] ** i for i in range(1, 4)])
poly_model = LinearRegression().fit(X_poly, df['price'])
```

## Interpretation

- **Step functions** are piecewise constant: the predicted value is the same for all observations within a bin. This produces "staircase" fitted values with discontinuities at bin boundaries.
- **Advantages**: Simple to interpret (the prediction is just the bin average), easy to implement, and can capture regime changes.
- **Disadvantages**: Discontinuous at boundaries (no smooth transitions), wasteful of degrees of freedom (each bin uses one parameter for a single constant), and sensitive to bin placement.
- **Comparison**: Step functions typically have lower $R^2$ than polynomial or spline models with the same number of parameters, because splines and polynomials use smoothness to share information across neighboring regions.
- **Classification**: Step functions work well for logistic regression when the log-odds change abruptly at certain predictor values.

## Exercises

**Exercise 1.** Fit a step function with quantile-based bins ($K = 5$) and compare the bin means to those from equal-width bins. Which approach gives more uniform prediction quality?

??? success "Solution to Exercise 1"

    ```python
    # Quantile bins
    df['q_bin'] = pd.qcut(df['age'], q=5)
    q_means = df.groupby('q_bin')['price'].agg(['mean', 'count'])

    # Equal-width bins
    df['w_bin'] = pd.cut(df['age'], bins=5)
    w_means = df.groupby('w_bin')['price'].agg(['mean', 'count'])
    ```

    Quantile bins ensure each bin has roughly equal counts, providing more uniform standard errors for the bin means. Equal-width bins may have very few observations in extreme bins (e.g., very old houses), leading to unreliable estimates in those bins. $\square$

---

**Exercise 2.** Explain why increasing the number of bins always increases (or does not decrease) training $R^2$, but may increase test error.

??? success "Solution to Exercise 2"

    Each bin adds one parameter to the model. With more bins, the model can fit finer patterns in the training data, so training RSS decreases and $R^2$ increases. However, with too many bins (especially bins with few observations), the bin means become noisy estimates of the true conditional expectation. Out-of-sample, these noisy estimates increase prediction error. In the extreme, $K = n$ bins gives $R^2 = 1$ on training data but terrible out-of-sample performance. $\square$

---

**Exercise 3.** Implement a step function using `np.digitize` instead of `pd.cut`. Verify the results match.

??? success "Solution to Exercise 3"

    ```python
    bin_edges = [0, 20, 40, 60, 80, 150]
    bin_indices = np.digitize(df['age'], bin_edges)
    # Create dummy matrix
    K = len(bin_edges) - 1
    X_digit = np.zeros((len(df), K))
    for k in range(K):
        X_digit[:, k] = (bin_indices == k + 1).astype(float)
    model_digit = LinearRegression().fit(X_digit, df['price'])
    ```

    The predictions match `pd.cut` + `pd.get_dummies` because both create the same indicator matrix. $\square$

---

**Exercise 4.** Show mathematically that the OLS estimate for each bin coefficient in a step function model (with all indicators and no intercept) equals the sample mean of the response within that bin.

??? success "Solution to Exercise 4"

    Let $\mathbf{X} = [C_1 \mid C_2 \mid \cdots \mid C_K]$ where $C_k$ is the indicator vector for bin $k$. Then $\mathbf{X}^\top\mathbf{X} = \mathrm{diag}(n_1, n_2, \ldots, n_K)$ (since the bins are non-overlapping) and $\mathbf{X}^\top\mathbf{y} = (\sum_{i \in B_1} y_i, \ldots, \sum_{i \in B_K} y_i)^\top$. Therefore

    $$
    \hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} = \left(\frac{\sum_{i \in B_1} y_i}{n_1}, \ldots, \frac{\sum_{i \in B_K} y_i}{n_K}\right)^\top = (\bar{y}_1, \ldots, \bar{y}_K)^\top.
    $$

    Each coefficient is simply the sample mean within its bin. $\square$

---

**Exercise 5.** Compare the step function logistic regression for predicting "expensive" houses with a logistic regression using the continuous age predictor. Which has better classification accuracy, and why?

??? success "Solution to Exercise 5"

    ```python
    from sklearn.linear_model import LogisticRegression

    # Continuous predictor
    logit_cont = LogisticRegression().fit(df[['age']], df['expensive'])
    acc_cont = logit_cont.score(df[['age']], df['expensive'])

    # Step function predictor
    X_class = pd.get_dummies(pd.cut(df['age'], bins=5), drop_first=True).values
    logit_step = LogisticRegression().fit(X_class, df['expensive'])
    acc_step = logit_step.score(X_class, df['expensive'])
    ```

    If the true log-odds are approximately linear in age, the continuous model may perform comparably or better with fewer parameters. If the log-odds change abruptly (e.g., houses over 60 years old are much less likely to be expensive), the step function captures this better. The relative performance depends on the true underlying relationship. $\square$
