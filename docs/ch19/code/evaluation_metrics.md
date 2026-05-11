# Receiver Operating Characteristic and Evaluation Metrics


## Overview

This page walks through the implementation and interpretation of core binary
classification evaluation metrics from scratch using NumPy.  We build a
confusion matrix, compute precision, recall, and the $F_1$ score, construct the
ROC curve by sweeping thresholds, and calculate AUC with the trapezoidal rule.

## Simulated Classification Problem

We create a two-class dataset with 120 positives and 180 negatives.  The
predicted scores are drawn from overlapping Gaussian distributions so that the
classes are not perfectly separable:

$$
s_i \sim
\begin{cases}
N(0.65,\; 0.25^2) & \text{if } y_i = 1 \\
N(0.35,\; 0.25^2) & \text{if } y_i = 0
\end{cases}
$$

```python
import numpy as np

np.random.seed(42)
n = 300
y_true = np.concatenate([np.ones(120), np.zeros(180)])
scores = np.concatenate([
    np.random.normal(0.65, 0.25, 120),
    np.random.normal(0.35, 0.25, 180)
])
scores = np.clip(scores, 0, 1)
```

## Confusion Matrix

At a given threshold $\tau$, each prediction is classified as positive
($\hat{y} = 1$) if $s_i \geq \tau$ and negative otherwise.  The four entries
of the confusion matrix are:

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | TN | FP |
| **Actual Positive** | FN | TP |

```python
def confusion_matrix(y_true, y_pred):
    """Compute 2x2 confusion matrix [TN, FP; FN, TP]."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

y_pred = (scores >= 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)
```

## Precision, Recall, and F1 Score

From the confusion matrix we derive:

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \qquad
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

$$
F_1 = \frac{2\,\text{Precision}\cdot\text{Recall}}{\text{Precision} + \text{Recall}}
$$

Accuracy is the overall fraction of correct predictions:

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{n}
$$

```python
def precision_recall_f1(y_true, y_pred):
    """Compute precision, recall, and F1 score."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1

prec, rec, f1 = precision_recall_f1(y_true, y_pred)
accuracy = np.mean(y_true == y_pred)
```

## ROC Curve from Scratch

The Receiver Operating Characteristic (ROC) curve plots the true positive rate
against the false positive rate as the threshold $\tau$ decreases from the
maximum score to the minimum:

$$
\text{TPR}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau)}, \qquad
\text{FPR}(\tau) = \frac{\text{FP}(\tau)}{\text{FP}(\tau) + \text{TN}(\tau)}
$$

```python
def roc_curve(y_true, scores):
    """Compute ROC curve (FPR, TPR) for varying thresholds."""
    thresholds = np.sort(np.unique(scores))[::-1]
    fpr_list, tpr_list = [0.0], [0.0]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tpr_list.append(tp / n_pos if n_pos > 0 else 0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0)

    return np.array(fpr_list), np.array(tpr_list), thresholds

fpr, tpr, thresholds = roc_curve(y_true, scores)
```

## AUC via the Trapezoidal Rule

The area under the ROC curve is approximated by summing the areas of
trapezoids formed by consecutive points:

$$
\text{AUC} \approx \sum_{i=1}^{m}
  \frac{\text{TPR}_i + \text{TPR}_{i-1}}{2}
  \bigl(\text{FPR}_i - \text{FPR}_{i-1}\bigr)
$$

```python
def auc_trapezoid(fpr, tpr):
    """Compute AUC via the trapezoidal rule."""
    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]
    return np.trapz(tpr_sorted, fpr_sorted)

area = auc_trapezoid(fpr, tpr)
print(f"AUC = {area:.4f}")
```

## Visualization

The script produces three panels:

1. **Confusion matrix heatmap** at threshold 0.5.
2. **Score distributions** for the positive and negative classes with the
   threshold shown as a vertical line.
3. **ROC curve** with the AUC annotated.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Confusion matrix heatmap
im = axes[0].imshow(cm, cmap='Blues', aspect='equal')
axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Neg', 'Pos'])
axes[0].set_yticklabels(['Neg', 'Pos'])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix')

# Score distributions
axes[1].hist(scores[y_true == 0], bins=25, alpha=0.6,
             label='Negative', edgecolor='k')
axes[1].hist(scores[y_true == 1], bins=25, alpha=0.6,
             label='Positive', edgecolor='k')
axes[1].axvline(0.5, color='red', linestyle='--', label='Threshold')
axes[1].set_xlabel('Predicted score'); axes[1].set_ylabel('Frequency')
axes[1].set_title('Score Distributions'); axes[1].legend(fontsize=8)

# ROC curve
axes[2].plot(fpr, tpr, linewidth=2, label=f'AUC = {area:.3f}')
axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('ROC Curve'); axes[2].legend(fontsize=9)

plt.tight_layout()
plt.show()
```

## Interpretation

- The separation between the two score distributions determines how well the
  classifier distinguishes the classes.  Greater separation yields a higher AUC.
- At threshold 0.5 the confusion matrix shows the specific trade-off between
  false positives and false negatives.
- The trapezoidal rule gives an unbiased estimate of AUC when applied to the
  empirical ROC points.
- AUC equals the probability that a randomly chosen positive example receives
  a higher score than a randomly chosen negative example.

## Exercises

**Exercise 1.**
A classifier produces the following confusion matrix at threshold 0.5:

$$
\begin{pmatrix} 90 & 10 \\ 30 & 70 \end{pmatrix}
$$

Compute accuracy, precision, recall, and $F_1$.

??? success "Solution to Exercise 1"

    Here TN = 90, FP = 10, FN = 30, TP = 70.

    $$
    \text{Accuracy} = \frac{90 + 70}{200} = 0.80
    $$

    $$
    \text{Precision} = \frac{70}{70 + 10} = \frac{70}{80} = 0.875
    $$

    $$
    \text{Recall} = \frac{70}{70 + 30} = \frac{70}{100} = 0.70
    $$

    $$
    F_1 = \frac{2 \times 0.875 \times 0.70}{0.875 + 0.70}
        = \frac{1.225}{1.575} \approx 0.778
    $$

    $\square$

---

**Exercise 2.**
Given three ROC points $(0, 0)$, $(0.2, 0.7)$, $(0.5, 0.9)$, $(1, 1)$,
compute the AUC using the trapezoidal rule.

??? success "Solution to Exercise 2"

    Apply the trapezoidal formula to each consecutive pair:

    $$
    A_1 = \frac{0 + 0.7}{2}(0.2 - 0) = 0.07
    $$

    $$
    A_2 = \frac{0.7 + 0.9}{2}(0.5 - 0.2) = 0.24
    $$

    $$
    A_3 = \frac{0.9 + 1.0}{2}(1.0 - 0.5) = 0.475
    $$

    $$
    \text{AUC} = 0.07 + 0.24 + 0.475 = 0.785
    $$

    $\square$

---

**Exercise 3.**
Prove that for a random classifier (scores independent of labels) the expected
AUC is 0.5.

??? success "Solution to Exercise 3"

    If the scores are independent of the labels, then for a randomly drawn
    positive-negative pair $(s^+, s^-)$:

    $$
    P(s^+ > s^-) = P(s^+ < s^-) = \frac{1}{2}
    $$

    by symmetry (ignoring ties, which have probability zero for continuous
    scores).  Since AUC equals the probability that a random positive is
    scored higher than a random negative, we obtain
    $\text{AUC} = 0.5$. $\square$

---

**Exercise 4.**
Implement a function that computes the **precision-recall curve** from scratch
(analogous to the `roc_curve` function above) by sweeping the threshold and
recording (recall, precision) pairs.

??? success "Solution to Exercise 4"

    ```python
    def pr_curve(y_true, scores):
        """Compute precision-recall curve for varying thresholds."""
        thresholds = np.sort(np.unique(scores))[::-1]
        precisions, recalls = [1.0], [0.0]
        n_pos = np.sum(y_true == 1)

        for thresh in thresholds:
            y_pred = (scores >= thresh).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)

        return np.array(recalls), np.array(precisions), thresholds
    ```

    The curve starts at (recall=0, precision=1) and traces the trade-off as
    the threshold decreases.  Average precision (AP) can be approximated by
    summing $\sum_k (R_k - R_{k-1}) P_k$ over the curve. $\square$

---

**Exercise 5.**
Show that for a binary classifier with $n$ data points, accuracy can be
written in terms of the confusion matrix as

$$
\text{Accuracy} = \frac{\text{trace}(C)}{n}
$$

where $C$ is the $2\times 2$ confusion matrix.

??? success "Solution to Exercise 5"

    The confusion matrix is

    $$
    C = \begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix}
    $$

    Its trace is $\text{trace}(C) = \text{TN} + \text{TP}$, the total number
    of correct predictions.  The total sample size satisfies
    $n = \text{TN} + \text{FP} + \text{FN} + \text{TP}$.  Therefore

    $$
    \text{Accuracy}
      = \frac{\text{TP} + \text{TN}}{n}
      = \frac{\text{trace}(C)}{n}
    $$

    This generalizes to the $K$-class setting: for a $K\times K$ confusion
    matrix, accuracy is still $\text{trace}(C)/n$. $\square$
