# Prediction vs Inference

Data analysis serves two fundamentally different goals: predicting outcomes as accurately as possible, and understanding the relationships between variables. The choice between them drives every methodological decision.

## Definition

**Prediction** aims to estimate $\hat{Y} = \hat{f}(X)$ with minimal error on unseen data; the internal structure of $\hat{f}$ is secondary. **Inference** aims to understand how $X$ relates to $Y$: which variables matter, the direction and magnitude of effects, and whether relationships are causal.

## Explanation

| Aspect | Prediction | Inference |
|---|---|---|
| Goal | Minimize forecast error | Understand relationships |
| Model choice | Whichever predicts best | Interpretable model preferred |
| Evaluation | Out-of-sample MSE, AUC | p-values, CIs, effect sizes |
| Complexity | Complex models welcome | Simpler models preferred |

The distinction maps onto the **bias-variance tradeoff**: inference favors lower-variance, interpretable models (even at the cost of some bias), while prediction favors the complexity that minimizes total error, often using regularization to manage variance.

Modern methods increasingly bridge the gap: LASSO selects variables (inference-like) while optimizing prediction; SHAP values provide post-hoc interpretability for black-box models; causal machine learning (double/debiased ML, causal forests) combines flexible prediction with valid causal inference.

## Examples

```python
import numpy as np

np.random.seed(42)
n = 300
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n)
x3 = 0.8 * x1 + np.random.normal(0, 0.5, n)  # correlated with x1
y = 3 * x1 - 2 * x2 + np.random.normal(0, 1, n)  # x3 is irrelevant

# Inference: identify which variables matter
X = np.column_stack([np.ones(n), x1, x2, x3])
beta = np.linalg.lstsq(X, y, rcond=None)[0]
y_pred = X @ beta
residuals = y - y_pred
s2 = np.sum(residuals**2) / (n - 4)
se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))

print("Inference: coefficient estimates and standard errors")
for i, name in enumerate(["intercept", "x1", "x2", "x3"]):
    t_val = beta[i] / se[i]
    print(f"  {name:>10s}: beta={beta[i]:+.3f}, SE={se[i]:.3f}, t={t_val:+.2f}")

# Prediction: out-of-sample MSE
from numpy.random import default_rng
rng = default_rng(0)
train = rng.choice(n, 200, replace=False)
test = np.setdiff1d(np.arange(n), train)
beta_train = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
mse_test = np.mean((y[test] - X[test] @ beta_train)**2)
print(f"\nPrediction: test MSE = {mse_test:.3f}")
```
