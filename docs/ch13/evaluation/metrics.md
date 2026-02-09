# Confusion Matrix and ROC

## Decision Rule

Logistic regression outputs a probability $\hat{p}=\sigma(z)$.  To
produce a binary prediction we choose a threshold $t$ (commonly $0.5$):

$$
\hat{y} = \begin{cases}1 & \hat{p} \ge t \\ 0 & \hat{p} < t\end{cases}
$$

Different thresholds trade off different types of errors.

## Confusion Matrix

A confusion matrix cross-tabulates true and predicted labels:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

### Derived Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Accuracy** | $(TP+TN)/(TP+TN+FP+FN)$ | Overall correctness |
| **Precision** | $TP/(TP+FP)$ | Of predicted positives, how many are correct |
| **Recall (Sensitivity)** | $TP/(TP+FN)$ | Of actual positives, how many are found |
| **Specificity** | $TN/(TN+FP)$ | Of actual negatives, how many are correctly identified |
| **F1 Score** | $2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$ | Harmonic mean of precision and recall |

## ROC Curve

The **Receiver Operating Characteristic (ROC)** curve plots the true
positive rate (recall) against the false positive rate
($1-\text{specificity}$) as the threshold $t$ varies from 1 to 0.

The **AUC** (area under the ROC curve) summarizes discriminative
ability: AUC = 1 is a perfect classifier, AUC = 0.5 is no better than
random guessing.

## Implementation

### Visualization Utilities

```python
import matplotlib.pyplot as plt

def plot_data(x, y):
    _, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, y, 'o', ms=5, label="data")
    ax.legend()
    plt.show()

def plot_result(x, y, y_pred, y_pred_prob):
    _, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, y, 'oy', alpha=0.9, ms=10, label="data")
    ax.plot(x, y_pred, '+r', ms=10, label="pred")
    ax.plot(x, y_pred_prob, '*b', alpha=0.3, ms=10, label="pred_prob")
    ax.legend()
    plt.show()
```

### Computing the Confusion Matrix with Scikit-Learn

```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
```

### ROC Curve and AUC

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], '--k', alpha=0.3)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### End-to-End Example with Scikit-Learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

url = ('https://raw.githubusercontent.com/codebasics/py/'
       'master/ML/7_logistic_reg/insurance_data.csv')
df = pd.read_csv(url)

x = df[['age']].values
y = df.bought_insurance.values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.5, random_state=1)

model = LogisticRegression(solver='lbfgs')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

plot_result(x_test, y_test, y_pred, y_prob)
```

## Practical Considerations

**Class imbalance.** When one class is much rarer, accuracy can be
misleading (a model predicting the majority class always gets high
accuracy).  Precision, recall, and AUC are more informative in such
settings.

**Threshold selection.** The optimal threshold depends on the relative
cost of false positives vs. false negatives.  In medical screening
(high cost of missing a case) a lower threshold increases recall at
the expense of precision. In fraud detection the threshold is often
tuned to a target precision.
