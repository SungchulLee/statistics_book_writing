# Supervised Learning

Supervised learning trains a model on labeled input-output pairs to predict outcomes on new data. It is the most widely used paradigm, powering credit scoring, forecasting, fraud detection, and image recognition.

## Definition

Given a training set $\{(\mathbf{x}_i, y_i)\}_{i=1}^n$, supervised learning finds a function $\hat{f}$ that minimizes a loss function measuring discrepancy between predictions $\hat{y}_i = \hat{f}(\mathbf{x}_i)$ and true labels $y_i$. The two main tasks are:

- **Regression**: $y \in \mathbb{R}$ (continuous target). Loss: MSE.
- **Classification**: $y \in \{1, \ldots, K\}$ (categorical target). Loss: cross-entropy or misclassification rate.

## Explanation

The workflow is: choose a model family, train on labeled data (minimize loss), validate (tune hyperparameters via cross-validation), test on held-out data, and deploy. Common methods include linear/logistic regression, decision trees, random forests, gradient boosting, and neural networks.

Evaluation is straightforward because ground-truth labels exist: accuracy, MSE, AUC, precision, recall, and F1-score can all be computed on test data. Overfitting (fitting noise instead of signal) is managed through regularization, cross-validation, and train/test splitting.

## Examples

```python
import numpy as np
from scipy.special import expit  # logistic function

np.random.seed(42)
n = 1000

# Simulate binary classification: loan default
income = np.random.normal(60, 20, n).clip(10)
dti = np.random.normal(0.3, 0.15, n).clip(0.01, 1.0)
log_odds = -3 + 0.01 * (50 - income) + 5 * (dti - 0.3)
prob = expit(log_odds)
default = np.random.binomial(1, prob)

# Train/test split
train, test = np.arange(700), np.arange(700, n)
X = np.column_stack([np.ones(n), income, dti])

# Fit logistic regression via IRLS (simplified: use lstsq on log-odds)
from scipy.optimize import minimize
def neg_log_lik(beta):
    z = X[train] @ beta
    return -np.sum(default[train] * z - np.log(1 + np.exp(z)))

result = minimize(neg_log_lik, np.zeros(3), method='BFGS')
beta_hat = result.x
probs_test = expit(X[test] @ beta_hat)
preds = (probs_test > 0.5).astype(int)
accuracy = np.mean(preds == default[test])
print(f"Test accuracy: {accuracy:.3f}")
print(f"Default rate (test): {default[test].mean():.3f}")
print(f"Coefficients: {beta_hat.round(4)}")
```
