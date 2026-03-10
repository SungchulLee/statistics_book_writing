# Strengths and Limitations of Each Approach

The classical and modern approaches are complementary, not competing. Understanding their tradeoffs guides the choice of methodology for any given problem.

## Definition

| Dimension | Classical (Designed Collection) | Modern (Algorithmic Learning) |
|---|---|---|
| Starting point | Research question then design then data | Existing data then algorithm then insight |
| Primary goal | Inference and causal understanding | Prediction and pattern discovery |
| Causality | Strong (via randomization) | Weak (association only) |
| Scalability | Limited by cost | Scales to billions of observations |
| Interpretability | High | Often low (black-box models) |
| Uncertainty | Built-in (CIs, p-values) | Requires bootstrap, calibration |

## Explanation

**Favor the classical approach** when causal claims are needed (clinical trials, A/B tests, policy evaluation), regulatory standards require designed experiments, or precise uncertainty quantification with probabilistic guarantees is essential.

**Favor the modern approach** when prediction is the primary goal, data already exists in large volumes, data is high-dimensional or unstructured, or speed of iteration matters.

**Combine both** when you need causal inference at scale (double/debiased ML, causal forests), when classical design (randomization) generates data analyzed by modern algorithms, or when post-hoc interpretability tools (SHAP, LIME) make black-box predictions understandable.

## Examples

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n = 500
true_effect = 2.0

# Classical: A/B test with randomization
group = np.random.choice([0, 1], size=n)
outcome = 10 + true_effect * group + np.random.normal(0, 5, n)
t_stat, p_val = stats.ttest_ind(outcome[group == 1], outcome[group == 0])
print("=== Classical A/B Test ===")
print(f"Estimated effect: {outcome[group==1].mean() - outcome[group==0].mean():.2f}")
print(f"p-value: {p_val:.4f}")

# Modern: prediction from observational features
from numpy.polynomial.polynomial import polyfit
X = np.random.randn(n, 3)
y = 2*X[:, 0] - X[:, 1] + 0.5*X[:, 2] + np.random.randn(n)
# Simple least squares prediction
beta = np.linalg.lstsq(X, y, rcond=None)[0]
y_pred = X @ beta
mse = np.mean((y - y_pred)**2)
print("\n=== Modern Prediction ===")
print(f"Coefficients: {beta.round(3)}")
print(f"MSE: {mse:.3f}")
```
