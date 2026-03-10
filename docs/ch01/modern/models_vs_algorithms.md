# Statistical Models vs Learning Algorithms

The shift from parametric statistical models to flexible learning algorithms is one of the most important transitions in data analysis, reflecting different goals: understanding versus prediction.

## Definition

A **statistical model** specifies a family of probability distributions indexed by interpretable parameters (e.g., $Y = \beta_0 + \beta_1 X + \varepsilon$). A **learning algorithm** is a computational procedure that identifies patterns in data without necessarily specifying a full probabilistic model (e.g., random forests, neural networks).

## Explanation

| Aspect | Statistical Model | Learning Algorithm |
|---|---|---|
| Primary goal | Inference and understanding | Prediction and pattern discovery |
| Assumptions | Explicit (distributional, structural) | Minimal or implicit |
| Interpretability | High (parameters have meaning) | Often low (black box) |
| Data requirements | Works with small, structured data | Thrives on large, complex data |
| Overfitting risk | Lower (fewer parameters) | Higher (managed via cross-validation) |

The boundary is blurred in practice: regularized regression (LASSO, Ridge) is a model enhanced with algorithmic regularization; Bayesian neural networks combine deep learning with probabilistic uncertainty; gradient boosting can be viewed as iterative model fitting.

The choice depends on the goal: if you need to **understand why**, favor interpretable models; if you need to **predict what**, algorithms often excel.

## Examples

```python
import numpy as np

np.random.seed(42)
n = 200
x = np.random.uniform(0, 10, n)
y = 3 + 2 * x - 0.1 * x**2 + np.random.normal(0, 2, n)

# Statistical model: linear regression (interpretable)
X_lin = np.column_stack([np.ones(n), x])
beta_lin = np.linalg.lstsq(X_lin, y, rcond=None)[0]
y_pred_lin = X_lin @ beta_lin
mse_lin = np.mean((y - y_pred_lin)**2)

# More flexible model: polynomial regression
X_poly = np.column_stack([np.ones(n), x, x**2])
beta_poly = np.linalg.lstsq(X_poly, y, rcond=None)[0]
y_pred_poly = X_poly @ beta_poly
mse_poly = np.mean((y - y_pred_poly)**2)

print(f"Linear model MSE:     {mse_lin:.3f}  (coeffs: {beta_lin.round(3)})")
print(f"Polynomial model MSE: {mse_poly:.3f}  (coeffs: {beta_poly.round(3)})")
print(f"True: y = 3 + 2x - 0.1x^2 + noise")
```
