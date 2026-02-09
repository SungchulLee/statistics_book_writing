# Multiclass Metrics

## Accuracy

The simplest multiclass metric counts the fraction of correct
predictions:

$$
\text{Accuracy} = \frac{1}{n}\sum_{i=1}^{n}\mathbf{1}[\hat{y}_i = y_i]
$$

Accuracy works well when classes are roughly balanced but can be
misleading otherwise.

## Confusion Matrix

The $C\times C$ confusion matrix $M$ has entry $M_{jk}$ equal to the
number of samples with true class $j$ and predicted class $k$.  A
perfect classifier produces a diagonal matrix.

### Reading the Matrix

The diagonal entries are the correct predictions for each class.
Off-diagonal entry $M_{jk}$ ($j\ne k$) shows how often class $j$ is
misclassified as class $k$.  For MNIST a common pattern is confusion
between visually similar digits (e.g. 3 vs 5, 4 vs 9).

### Per-Class Metrics

From the confusion matrix we can extract per-class precision, recall,
and F1 by treating each class as a one-vs-rest binary problem:

| Metric | Class $c$ formula |
|---|---|
| Precision$_c$ | $M_{cc} / \sum_j M_{jc}$ (column sum) |
| Recall$_c$ | $M_{cc} / \sum_k M_{ck}$ (row sum) |
| F1$_c$ | $2\cdot\text{Prec}_c\cdot\text{Rec}_c / (\text{Prec}_c+\text{Rec}_c)$ |

### Macro vs Micro Averaging

**Macro-average** computes the metric independently for each class and
then averages.  **Micro-average** pools the per-class counts and
computes a single metric.  For balanced classes both agree; for
imbalanced classes micro-average is dominated by the majority class.

## Implementation with Scikit-Learn

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

digits = load_digits()
x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=1)

model = LogisticRegression(solver='lbfgs', max_iter=10_000)
model.fit(x_train, y_train)
print(f"Test accuracy: {model.score(x_test, y_test):.4f}")

y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))
```

## Visualizing Misclassifications

Inspecting incorrectly classified images provides insight into model
limitations and guides feature engineering or architecture improvements.

```python
import matplotlib.pyplot as plt

def draw_10_wrong_preds(x_test, y_test_cls, y_pred_cls):
    _, axes = plt.subplots(1, 10, figsize=(12, 3))
    idx = 0
    for ax in axes:
        while y_test_cls[idx] == y_pred_cls[idx]:
            idx += 1
        ax.imshow(x_test[idx].reshape((28, 28)), cmap='binary')
        ax.set_title(f'True: {y_test_cls[idx]}\nPred: {y_pred_cls[idx]}',
                     fontsize=10)
        ax.axis('off')
        idx += 1
    plt.tight_layout()
    plt.show()
```

## Plotting Training Curves

```python
def draw_loss_and_accuracy(loss_trace, accuracy_trace):
    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3))
    for ax, trace, title in zip(
            (ax0, ax1), (loss_trace, accuracy_trace), ("Loss", "Accuracy")):
        ax.plot(trace)
        ax.set_title(title)
    plt.tight_layout()
    plt.show()
```
