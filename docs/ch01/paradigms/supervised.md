# Supervised Learning (Prediction with Labels)

## Overview

In **supervised learning**, the model is trained on a **labeled dataset**—each input $X$ has a corresponding known output $Y$. The goal is for the model to learn the mapping $f: X \to Y$ and make accurate predictions on new, unseen data.

## Key Characteristics

- **Labeled data**: The training set consists of input–output pairs $\{(x_i, y_i)\}_{i=1}^{n}$.
- **Clear objective**: Minimize a loss function that measures the discrepancy between predictions $\hat{y}_i$ and true labels $y_i$.
- **Evaluation is straightforward**: Accuracy, MSE, AUC, and other metrics can be computed on held-out test data.

## Two Main Tasks

### Regression

The target variable $Y$ is **continuous**. The goal is to predict a numerical value.

$$
\hat{y} = f(x) \quad \text{where } y \in \mathbb{R}
$$

**Examples:**

- Predicting house prices from features (size, location, age).
- Forecasting next-quarter revenue from macroeconomic indicators.
- Estimating option prices from underlying asset characteristics.

**Common methods:** Linear regression, Ridge/LASSO, decision trees, random forests, gradient boosting, neural networks.

### Classification

The target variable $Y$ is **categorical**. The goal is to assign an input to one of $K$ classes.

$$
\hat{y} = f(x) \quad \text{where } y \in \{1, 2, \ldots, K\}
$$

**Examples:**

- Classifying emails as spam or not spam (binary).
- Recognizing handwritten digits 0–9 (multiclass).
- Predicting whether a borrower will default on a loan (binary).

**Common methods:** Logistic regression, support vector machines, decision trees, random forests, gradient boosting, neural networks.

## The Supervised Learning Workflow

```
Training Data {(x_i, y_i)}
        │
        ▼
  Choose Model Family
        │
        ▼
  Train (minimize loss)
        │
        ▼
  Validate (tune hyperparameters)
        │
        ▼
  Test (evaluate on held-out data)
        │
        ▼
  Deploy for Prediction
```

## Example: Predicting Default

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)

# Simulate data: income and debt-to-income ratio → default (0/1)
n = 1_000
income = np.random.normal(60, 20, n).clip(10)
dti = np.random.normal(0.3, 0.15, n).clip(0.01, 1.0)
log_odds = -3 + 0.01 * (50 - income) + 5 * (dti - 0.3)
prob = 1 / (1 + np.exp(-log_odds))
default = np.random.binomial(1, prob)

X = np.column_stack([income, dti])
y = default

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
```

## Key Takeaways

- Supervised learning requires labeled data and optimizes a well-defined loss function.
- The two main tasks are **regression** (continuous target) and **classification** (categorical target).
- Model evaluation on held-out data is essential to assess generalization.
- In finance, supervised learning powers credit scoring, algorithmic trading signals, fraud detection, and many other applications.
