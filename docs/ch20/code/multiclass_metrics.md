# Multiclass Evaluation Metrics

## Overview

This page implements multiclass evaluation metrics from scratch in Python and compares the results with scikit-learn. We build the confusion matrix, extract per-class precision, recall, and F1-score, and compute macro- and micro-averaged summaries. The code makes the mathematical definitions concrete and provides a template for evaluating any multiclass classifier.

---

## Confusion Matrix

The $C \times C$ confusion matrix $\mathbf{M}$ has entry $M_{jk}$ equal to the number of observations with true class $j$ and predicted class $k$. A perfect classifier produces a diagonal matrix.

```python
import numpy as np

def confusion_matrix(y_true, y_pred, C):
    """Build a C x C confusion matrix.

    Parameters
    ----------
    y_true : ndarray, shape (n,)
        True class labels in {0, ..., C-1}.
    y_pred : ndarray, shape (n,)
        Predicted class labels in {0, ..., C-1}.
    C : int
        Number of classes.

    Returns
    -------
    ndarray, shape (C, C)
        M[j, k] = number of samples with true class j and predicted class k.
    """
    M = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M
```

Example with a small 3-class problem:

```python
y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
y_pred = np.array([0, 0, 1, 1, 1, 2, 2, 2, 0])

M = confusion_matrix(y_true, y_pred, C=3)
print(M)
# [[2 1 0]
#  [0 2 1]
#  [1 0 2]]
```

---

## Overall Accuracy

Accuracy is the fraction of correct predictions, which equals the trace of the confusion matrix divided by the total number of observations:

$$
\text{Accuracy} = \frac{\operatorname{tr}(\mathbf{M})}{n} = \frac{\sum_{c=0}^{C-1} M_{cc}}{\sum_{j,k} M_{jk}}
$$

```python
def accuracy(M):
    """Overall accuracy from a confusion matrix."""
    return np.trace(M) / np.sum(M)

print(f"Accuracy: {accuracy(M):.4f}")
# Accuracy: 0.6667
```

---

## Per-Class Precision, Recall, and F1

By treating each class as a one-vs-rest binary problem we extract per-class metrics from the confusion matrix.

For class $c$:

$$
\text{Precision}_c = \frac{M_{cc}}{\sum_{j=0}^{C-1} M_{jc}}, \qquad
\text{Recall}_c = \frac{M_{cc}}{\sum_{k=0}^{C-1} M_{ck}}
$$

$$
F_{1,c} = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
$$

Precision is the fraction of predictions for class $c$ that are correct (column-wise), while recall is the fraction of actual class $c$ observations that are correctly identified (row-wise).

```python
def per_class_metrics(M):
    """Compute precision, recall, and F1 for each class.

    Returns
    -------
    precisions : ndarray, shape (C,)
    recalls    : ndarray, shape (C,)
    f1s        : ndarray, shape (C,)
    """
    C = M.shape[0]
    precisions = np.zeros(C)
    recalls = np.zeros(C)
    f1s = np.zeros(C)

    for c in range(C):
        col_sum = M[:, c].sum()       # total predicted as c
        row_sum = M[c, :].sum()       # total truly in class c
        tp = M[c, c]

        precisions[c] = tp / col_sum if col_sum > 0 else 0.0
        recalls[c] = tp / row_sum if row_sum > 0 else 0.0
        if precisions[c] + recalls[c] > 0:
            f1s[c] = 2 * precisions[c] * recalls[c] / (precisions[c] + recalls[c])

    return precisions, recalls, f1s

prec, rec, f1 = per_class_metrics(M)
for c in range(3):
    print(f"Class {c}: Prec={prec[c]:.3f}  Rec={rec[c]:.3f}  F1={f1[c]:.3f}")
```

---

## Macro and Micro Averaging

**Macro-averaging** computes the metric for each class independently, then takes the unweighted mean:

$$
F_1^{\text{macro}} = \frac{1}{C}\sum_{c=0}^{C-1} F_{1,c}
$$

**Micro-averaging** pools all true positives, false positives, and false negatives across classes before computing a single metric:

$$
\text{Precision}_{\text{micro}} = \frac{\sum_c \text{TP}_c}{\sum_c \text{TP}_c + \sum_c \text{FP}_c}, \qquad
\text{Recall}_{\text{micro}} = \frac{\sum_c \text{TP}_c}{\sum_c \text{TP}_c + \sum_c \text{FN}_c}
$$

For a single-label problem, micro-averaged precision equals micro-averaged recall equals accuracy.

```python
def macro_f1(M):
    """Macro-averaged F1-score."""
    _, _, f1s = per_class_metrics(M)
    return np.mean(f1s)

def micro_f1(M):
    """Micro-averaged F1-score."""
    C = M.shape[0]
    tp_total = fp_total = fn_total = 0
    for c in range(C):
        tp = M[c, c]
        fp = M[:, c].sum() - tp
        fn = M[c, :].sum() - tp
        tp_total += tp
        fp_total += fp
        fn_total += fn
    prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

print(f"Macro F1: {macro_f1(M):.4f}")
print(f"Micro F1: {micro_f1(M):.4f}")
```

---

## Verification Against scikit-learn

We train a softmax regression model on the Iris dataset and compare our from-scratch metrics with scikit-learn's `classification_report`.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42)

clf = LogisticRegression(multi_class='multinomial', solver='lbfgs',
                         max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

C = 3
M_ours = confusion_matrix(y_test, y_pred, C)
M_sklearn = sk_confusion_matrix(y_test, y_pred)

print("Our confusion matrix:")
print(M_ours)
print("\nscikit-learn confusion matrix:")
print(M_sklearn)
print("\nscikit-learn classification report:")
print(classification_report(y_test, y_pred, digits=4))
print(f"Our Macro F1:  {macro_f1(M_ours):.4f}")
print(f"Our Micro F1:  {micro_f1(M_ours):.4f}")
```

The outputs should match, confirming the correctness of our implementation.

---

## Visualizing the Confusion Matrix

A heatmap makes it easy to spot systematic misclassification patterns.

```python
import matplotlib.pyplot as plt

def plot_confusion_matrix(M, class_names=None):
    """Display confusion matrix as a heatmap."""
    C = M.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(C)]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(M, cmap='Blues')
    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(C):
        for j in range(C):
            ax.text(j, i, str(M[i, j]), ha='center', va='center',
                    color='white' if M[i, j] > M.max() / 2 else 'black')

    fig.colorbar(im)
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(M_ours, class_names=iris.target_names)
```

---

## Interpretation

The from-scratch implementation highlights several key points:

1. **Accuracy can be misleading.** When classes are imbalanced, a model that always predicts the majority class achieves high accuracy but zero recall on minority classes. Per-class metrics and macro-averaging reveal this failure.
2. **Macro vs micro.** Macro-averaging treats all classes equally regardless of their size, making it sensitive to poor performance on rare classes. Micro-averaging is dominated by the majority class, which makes it equivalent to accuracy in single-label problems.
3. **Confusion matrix is the master summary.** Every scalar metric (accuracy, precision, recall, F1) can be derived from the confusion matrix. The matrix itself carries strictly more information than any single number.
4. **Off-diagonal patterns are diagnostic.** Systematically large off-diagonal entries (e.g., class 4 predicted as class 9 in digit recognition) point to specific modeling deficiencies that can guide feature engineering or data augmentation.

---

## Exercises

**Exercise 1.**
A 4-class classifier produces the following confusion matrix:

|  | Pred 0 | Pred 1 | Pred 2 | Pred 3 |
|:---:|:---:|:---:|:---:|:---:|
| **True 0** | 40 | 5 | 3 | 2 |
| **True 1** | 2 | 35 | 8 | 5 |
| **True 2** | 1 | 4 | 42 | 3 |
| **True 3** | 0 | 6 | 2 | 42 |

Compute the overall accuracy, per-class precision and recall, and the macro-averaged F1-score by hand.

??? success "Solution to Exercise 1"
    The total number of observations is $40+5+3+2+2+35+8+5+1+4+42+3+0+6+2+42 = 200$.

    **Overall accuracy:**

    $$
    \text{Accuracy} = \frac{40 + 35 + 42 + 42}{200} = \frac{159}{200} = 0.795
    $$

    **Per-class metrics:**

    | Class | TP | Col sum | Row sum | Prec | Rec | F1 |
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    | 0 | 40 | 43 | 50 | 40/43 = 0.930 | 40/50 = 0.800 | 0.860 |
    | 1 | 35 | 50 | 50 | 35/50 = 0.700 | 35/50 = 0.700 | 0.700 |
    | 2 | 42 | 55 | 50 | 42/55 = 0.764 | 42/50 = 0.840 | 0.800 |
    | 3 | 42 | 52 | 50 | 42/52 = 0.808 | 42/50 = 0.840 | 0.824 |

    **Macro F1:**

    $$
    F_1^{\text{macro}} = \frac{0.860 + 0.700 + 0.800 + 0.824}{4} = \frac{3.184}{4} = 0.796
    $$

---

**Exercise 2.**
Show that for a single-label multiclass problem, micro-averaged precision equals micro-averaged recall equals overall accuracy. (Hint: relate $\sum_c \text{FP}_c$ and $\sum_c \text{FN}_c$ to the off-diagonal entries of $\mathbf{M}$.)

??? success "Solution to Exercise 2"
    For each class $c$:

    - $\text{TP}_c = M_{cc}$
    - $\text{FP}_c = \sum_{j \neq c} M_{jc}$ (other rows in column $c$)
    - $\text{FN}_c = \sum_{k \neq c} M_{ck}$ (other columns in row $c$)

    Summing over all classes:

    $$
    \sum_c \text{TP}_c = \sum_c M_{cc} = \operatorname{tr}(\mathbf{M})
    $$

    $$
    \sum_c \text{FP}_c = \sum_c \sum_{j \neq c} M_{jc} = \sum_{j,c} M_{jc} - \sum_c M_{cc} = n - \operatorname{tr}(\mathbf{M})
    $$

    Similarly:

    $$
    \sum_c \text{FN}_c = \sum_c \sum_{k \neq c} M_{ck} = n - \operatorname{tr}(\mathbf{M})
    $$

    Therefore $\sum_c \text{FP}_c = \sum_c \text{FN}_c$, and:

    $$
    \text{Precision}_{\text{micro}} = \frac{\operatorname{tr}(\mathbf{M})}{\operatorname{tr}(\mathbf{M}) + (n - \operatorname{tr}(\mathbf{M}))} = \frac{\operatorname{tr}(\mathbf{M})}{n} = \text{Accuracy}
    $$

    The same calculation applies to recall. Since micro-precision equals micro-recall, the micro-F1 also equals accuracy. $\square$

---

**Exercise 3.**
A medical screening test classifies patients into 3 categories: healthy (0), condition A (1), and condition B (2). The class distribution is 900 healthy, 70 with condition A, and 30 with condition B out of 1000 patients. A classifier that always predicts "healthy" achieves 90% accuracy. Compute the macro-averaged F1 for this trivial classifier and explain why macro-F1 is a better evaluation criterion than accuracy for this problem.

??? success "Solution to Exercise 3"
    The confusion matrix for the "always predict 0" classifier is:

    |  | Pred 0 | Pred 1 | Pred 2 |
    |:---:|:---:|:---:|:---:|
    | **True 0** | 900 | 0 | 0 |
    | **True 1** | 70 | 0 | 0 |
    | **True 2** | 30 | 0 | 0 |

    Per-class metrics:

    - **Class 0:** Precision $= 900/1000 = 0.900$, Recall $= 900/900 = 1.000$, $F_1 = 2(0.9)(1.0)/(0.9+1.0) = 0.947$
    - **Class 1:** Precision $= 0/0$ (undefined, set to 0), Recall $= 0/70 = 0$, $F_1 = 0$
    - **Class 2:** Precision $= 0/0$ (undefined, set to 0), Recall $= 0/30 = 0$, $F_1 = 0$

    $$
    F_1^{\text{macro}} = \frac{0.947 + 0 + 0}{3} = 0.316
    $$

    While accuracy is 90%, macro-F1 is only 0.316, correctly reflecting that the classifier is useless for the two disease classes. In medical screening, failing to detect conditions A and B has severe consequences. Macro-F1 penalizes this failure heavily because it weights all classes equally, regardless of prevalence.

---

**Exercise 4.**
Implement a `weighted_f1` function that computes the weighted-average F1, where each class's F1-score is weighted by its support (number of true instances). Show that for a balanced dataset, weighted-F1 equals macro-F1.

??? success "Solution to Exercise 4"
    ```python
    def weighted_f1(M):
        """Weighted-average F1-score."""
        _, _, f1s = per_class_metrics(M)
        supports = M.sum(axis=1)          # row sums = class sizes
        total = supports.sum()
        return np.sum(f1s * supports) / total
    ```

    For a balanced dataset, every class has the same support $s = n/C$. Then:

    $$
    F_1^{\text{weighted}} = \frac{\sum_c F_{1,c} \cdot s}{\sum_c s} = \frac{s \sum_c F_{1,c}}{C \cdot s} = \frac{1}{C}\sum_c F_{1,c} = F_1^{\text{macro}}
    $$

    The weights $s/Cs = 1/C$ become uniform, reducing to the unweighted mean. $\square$

---

**Exercise 5.**
Prove that for any confusion matrix $\mathbf{M}$ and any class $c$, the F1-score satisfies $0 \leq F_{1,c} \leq 1$, with $F_{1,c} = 1$ if and only if class $c$ has perfect precision and perfect recall.

??? success "Solution to Exercise 5"
    The F1-score is the harmonic mean of precision and recall:

    $$
    F_{1,c} = \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}
    $$

    where $P_c, R_c \in [0, 1]$.

    **Lower bound:** Since $P_c \geq 0$ and $R_c \geq 0$, the numerator $2P_c R_c \geq 0$ and the denominator $P_c + R_c \geq 0$. If either $P_c = 0$ or $R_c = 0$, then $F_{1,c} = 0$. Hence $F_{1,c} \geq 0$.

    **Upper bound:** By the AM-GM inequality, $P_c R_c \leq \left(\frac{P_c + R_c}{2}\right)^2$, so:

    $$
    F_{1,c} = \frac{2 P_c R_c}{P_c + R_c} \leq \frac{2 \cdot \frac{(P_c + R_c)^2}{4}}{P_c + R_c} = \frac{P_c + R_c}{2} \leq \frac{1 + 1}{2} = 1
    $$

    Alternatively, note directly that the harmonic mean of two numbers in $[0,1]$ is at most their arithmetic mean, which is at most 1.

    **Equality at 1:** $F_{1,c} = 1$ requires $2P_c R_c = P_c + R_c$, i.e., $2P_c R_c - P_c - R_c = 0$. Factoring: $(2P_c - 1)(2R_c - 1) = 1$. Since $P_c, R_c \in [0,1]$, the factors $(2P_c - 1)$ and $(2R_c - 1)$ are each in $[-1, 1]$. Their product equals 1 only when both equal 1, giving $P_c = R_c = 1$. Thus $F_{1,c} = 1$ if and only if precision and recall are both perfect. $\square$
