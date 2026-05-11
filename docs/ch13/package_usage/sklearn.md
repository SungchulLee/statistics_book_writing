# sklearn LinearRegression Interface

While `statsmodels` is designed for statistical inference, scikit-learn's `LinearRegression` is designed for prediction. It follows scikit-learn's consistent estimator API — `fit`, `predict`, `score` — and integrates seamlessly with the library's preprocessing, pipeline, and cross-validation tools. The tradeoff is that `sklearn` does not provide p-values, confidence intervals, or diagnostic tests out of the box.

---

## 1. Basic Usage

The `LinearRegression` class fits the model $Y = \mathbf{X}\boldsymbol{\beta} + \varepsilon$ by ordinary least squares. Unlike `statsmodels`, it adds the intercept automatically by default.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate example data
np.random.seed(42)
n = 100
X = np.random.randn(n, 2)
beta_true = np.array([3.0, 1.5])
y = X @ beta_true + 2.0 + np.random.randn(n) * 0.5

# Fit model
model = LinearRegression()
model.fit(X, y)

# Access coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
```

The fitted model stores the intercept in `model.intercept_` and the slope coefficients in `model.coef_`. Note that `X` does not need a column of ones — the intercept is handled internally when `fit_intercept=True` (the default).

---

## 2. Prediction

The `predict` method computes fitted values for new data:

```python
# Predict on training data
y_pred_train = model.predict(X)

# Predict on new data
X_new = np.array([[1.0, 0.5], [-0.5, 2.0]])
y_pred_new = model.predict(X_new)
print("Predictions:", y_pred_new)
```

The input to `predict` must have the same number of columns as the training data. Each row is a new observation, and the output is the vector of predicted values $\hat{y} = \hat{\beta}_0 + \mathbf{X}_{\text{new}} \hat{\boldsymbol{\beta}}$.

---

## 3. Evaluating Model Performance

The `score` method returns $R^2$ on a given dataset:

```python
r2_train = model.score(X, y)
print(f"R-squared (training): {r2_train:.4f}")
```

For other metrics, use `sklearn.metrics`:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = root_mean_squared_error(y, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
```

---

## 4. Train-Test Split

To estimate out-of-sample performance, split the data before fitting:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

r2_test = model.score(X_test, y_test)
print(f"R-squared (test): {r2_test:.4f}")
```

The test $R^2$ is a more honest estimate of predictive performance than the training $R^2$ because the model has not seen the test data during fitting.

---

## 5. Cross-Validation

scikit-learn provides cross-validation utilities that automate the repeated train-test splitting:

```python
from sklearn.model_selection import cross_val_score

model = LinearRegression()
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"CV R-squared scores: {cv_scores}")
print(f"Mean CV R-squared: {cv_scores.mean():.4f}")
print(f"Std CV R-squared: {cv_scores.std():.4f}")
```

The `scoring` parameter accepts any scikit-learn scorer. Common choices for regression include `'r2'`, `'neg_mean_squared_error'`, and `'neg_mean_absolute_error'`. The "neg" prefix is used because scikit-learn's convention is that higher scores are better, so error metrics are negated.

---

## 6. Pipelines

Pipelines chain preprocessing and modeling steps into a single object, ensuring that transformations are applied consistently during training and prediction:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('regression', LinearRegression())
])

pipeline.fit(X_train, y_train)
r2_pipeline = pipeline.score(X_test, y_test)
print(f"Pipeline R-squared (test): {r2_pipeline:.4f}")
```

This pipeline first standardizes each predictor to zero mean and unit variance, then creates polynomial features (including interaction terms), and finally fits a linear regression. The entire pipeline can be passed to `cross_val_score` for cross-validated evaluation.

!!! note "Why pipelines matter"
    Without a pipeline, it is easy to accidentally standardize or transform the test data using training statistics that were computed on the full dataset (data leakage). Pipelines prevent this by applying each transformation step only to the data that is available at that point in the workflow.

---

## 7. When to Use sklearn vs statsmodels

| Task | Recommended library |
|---|---|
| Hypothesis testing on coefficients | `statsmodels` |
| Confidence intervals for $\beta_j$ | `statsmodels` |
| Residual diagnostics (normality, heteroscedasticity) | `statsmodels` |
| Model comparison via AIC/BIC | `statsmodels` |
| Prediction on new data | `sklearn` |
| Cross-validated model evaluation | `sklearn` |
| Integration with preprocessing pipelines | `sklearn` |
| Regularized regression (ridge, lasso) | `sklearn` |
| High-dimensional data ($p > n$) | `sklearn` |

!!! tip "They are complementary, not competing"
    A common workflow is to use `statsmodels` during the model-building phase to test hypotheses, examine diagnostics, and select predictors, then refit the final model in `sklearn` for deployment in a prediction pipeline. The coefficient estimates are identical (both use OLS), so the choice is about which tools you need around the model.

## Exercises

**Exercise 1.**
Write Python code using scikit-learn to fit a linear regression on synthetic data, compute predictions, and print the $R^2$ score.

??? success "Solution to Exercise 1"
    ```python
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 3))
    y = 2 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.5, 200)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    print(f"Coefficients: {model.coef_.round(3)}")
    print(f"Intercept: {model.intercept_:.3f}")
    print(f"R^2 (test): {model.score(X_test, y_test):.4f}")
    ```

---

**Exercise 2.**
Explain the difference between `model.score()` and manually computing $R^2$ from predictions. Are they equivalent?

??? success "Solution to Exercise 2"
    `model.score(X, y)` computes:

    $$
    R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
    $$

    where $\hat{y}_i$ = `model.predict(X)` and $\bar{y}$ is the mean of the provided `y`. This is equivalent to manually computing:

    ```python
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    ```

    They are mathematically identical. Note: when applied to test data, $\bar{y}$ is the test set mean (not the training set mean), which can make test $R^2$ negative if the model fits poorly.

---

**Exercise 3.**
Why does scikit-learn's `LinearRegression` not provide p-values or confidence intervals for coefficients? How can you obtain them?

??? success "Solution to Exercise 3"
    Scikit-learn is designed primarily for prediction, not statistical inference. `LinearRegression` implements OLS as a machine learning algorithm and focuses on `.fit()`, `.predict()`, and `.score()`. It does not compute standard errors, t-statistics, p-values, or confidence intervals.

    To obtain inferential statistics, use:

    1. **statsmodels:** `import statsmodels.api as sm; model = sm.OLS(y, sm.add_constant(X)).fit(); print(model.summary())` provides a full regression table with p-values, CIs, and diagnostic statistics.
    2. **Manual computation:** Compute $\hat{\boldsymbol{\beta}}$, then $s^2 = \text{SSE}/(n-p)$, $\text{SE}(\hat{\beta}_j) = s\sqrt{[(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}}$, and $t_j = \hat{\beta}_j/\text{SE}(\hat{\beta}_j)$.

    The separation between prediction-focused (sklearn) and inference-focused (statsmodels) tools reflects the models-vs-algorithms distinction.

---

**Exercise 4.**
Describe how to use `sklearn.model_selection.cross_val_score` to estimate the generalization performance of a linear regression model.

??? success "Solution to Exercise 4"
    ```python
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    import numpy as np

    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    print(f"CV R^2 scores: {scores.round(4)}")
    print(f"Mean R^2: {scores.mean():.4f} (+/- {scores.std():.4f})")
    ```

    `cross_val_score` performs k-fold CV internally: it splits the data into `cv=5` folds, fits the model on 4 folds, evaluates on the 5th, and repeats for all folds. The `scoring` parameter specifies the metric (options include `"r2"`, `"neg_mean_squared_error"`, `"neg_mean_absolute_error"`).

    Note: sklearn uses negative MSE (`neg_mean_squared_error`) so that higher values are always better (consistent with the convention that the scorer should be maximized).
