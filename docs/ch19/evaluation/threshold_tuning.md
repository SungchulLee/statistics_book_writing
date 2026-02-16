# Decision Threshold Tuning

## Overview

In binary classification, logistic regression produces predicted probabilities in $[0,1]$. To make a hard prediction (positive or negative), we must choose a **decision threshold**. The default threshold of 0.5 is not always optimal—the right choice depends on the costs of false positives and false negatives.

Adjusting the threshold directly affects the precision-recall tradeoff and confusion matrix, allowing you to align the classifier with the specific needs of your application.

## Default Threshold: 0.5

By convention, most binary classifiers use a threshold of 0.5:

$$
\hat{y} = \begin{cases}
1 & \text{if } P(Y=1|\mathbf{x}) \geq 0.5 \\
0 & \text{otherwise}
\end{cases}
$$

Under this rule, an observation is classified as positive if the predicted probability exceeds 50%.

### When is 0.5 Appropriate?

The 0.5 threshold is optimal when:
- The cost of false positives equals the cost of false negatives
- The classes are balanced (similar prevalence)
- You have no prior reason to favor one error type over the other

In many real applications, these conditions do not hold.

## Adjusting the Threshold

### Lower Threshold (e.g., 0.2)

When you lower the threshold, the classifier becomes more **lenient**—it predicts "positive" more often.

**Effect on confusion matrix:**
- Recall (sensitivity) increases: catches more true positives
- Specificity decreases: more false positives
- Precision decreases: fewer predicted positives are actually correct

**Use when:**
- The cost of missing a positive (false negative) is high
- Medical screening: minimize missed disease cases
- Fraud detection: catch more fraudulent transactions
- Loan default prediction: identify as many risky borrowers as possible

### Higher Threshold (e.g., 0.8)

When you raise the threshold, the classifier becomes more **conservative**—it predicts "positive" only when very confident.

**Effect on confusion matrix:**
- Specificity (true negative rate) increases: fewer false alarms
- Recall decreases: more false negatives
- Precision increases: most predicted positives are correct

**Use when:**
- The cost of a false positive is high
- Email spam filtering: avoid filtering legitimate emails
- Credit approval: approve only very safe borrowers
- Recommender systems: only recommend when very confident

## Example: Loan Default Prediction

### Model Predictions

Consider a logistic regression model trained on 1,000 loan observations. The distribution of predicted probabilities is:

- 400 loans have predicted probability < 0.2 (very likely to repay)
- 300 loans have predicted probability 0.2-0.5 (likely to repay)
- 200 loans have predicted probability 0.5-0.8 (likely to default)
- 100 loans have predicted probability > 0.8 (very likely to default)

### Confusion Matrices at Different Thresholds

**Threshold = 0.5 (balanced):**

```
                      Predicted
                   No Default  Default
Actual No Default       280       120
Actual Default           70       530
```

- Sensitivity (recall): 530/(530+70) = 0.883
- Specificity: 280/(280+120) = 0.700
- Precision: 530/(530+120) = 0.815

**Threshold = 0.2 (lenient):**

```
                      Predicted
                   No Default  Default
Actual No Default        150       250
Actual Default            10       590
```

- Sensitivity: 590/(590+10) = 0.983 (catch almost all defaults)
- Specificity: 150/(150+250) = 0.375 (many false alarms)
- Precision: 590/(590+250) = 0.702 (less reliable)

**Threshold = 0.8 (conservative):**

```
                      Predicted
                   No Default  Default
Actual No Default       350        50
Actual Default         250       350
```

- Sensitivity: 350/(350+250) = 0.583 (miss more defaults)
- Specificity: 350/(350+50) = 0.875 (fewer false alarms)
- Precision: 350/(350+50) = 0.875 (very reliable)

## The Precision-Recall Tradeoff

As you lower the threshold:
- **Recall** (sensitivity) ↑: Identify more positives
- **Precision** ↓: More false positives dilute the positive predictions

As you raise the threshold:
- **Precision** ↑: Fewer false positives
- **Recall** ↓: Miss more true positives

This fundamental tradeoff is visualized in the **precision-recall curve**, which plots precision (y-axis) against recall (x-axis) as the threshold varies from 1 to 0.

## Selecting the Optimal Threshold

### Cost-Sensitive Approach

If you know the cost of each error type, you can compute the optimal threshold analytically. Let:

- $C_{FP}$ = cost of a false positive
- $C_{FN}$ = cost of a false negative

The optimal threshold is approximately:

$$
t^* = \frac{C_{FN}}{C_{FP} + C_{FN}}
$$

For example, if a missed default costs \$1,000 and a false alarm costs \$50:

$$
t^* = \frac{1000}{1000 + 50} \approx 0.95
$$

This high threshold reflects the asymmetric costs.

### Performance Metrics

Common methods to select a threshold:

| Metric | Optimal When |
|--------|--------------|
| **F1 Score** | $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Equal importance of precision and recall |
| **Youden's J** | $J = \text{Sensitivity} + \text{Specificity} - 1$ | Balanced classes, no cost information |
| **Precision-Recall Curve** | Look for "elbow" or domain-specific target | Trade off precision/recall visually |
| **ROC Curve** | Youden's J maximizes TPR - FPR | Imbalanced classes |

### Domain-Specific Targets

In practice, domain knowledge often guides threshold selection:

- **Medical screening:** High recall (catch disease early), tolerate false alarms
- **Fraud detection:** High precision (avoid hassling customers), moderate recall
- **Loan approval:** High precision (avoid defaults), lower recall (accept some good borrowers)

## Implementation

### Python Example

```python
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score)

# Assume model trained and y_prob contains predicted probabilities
thresholds = [0.2, 0.3, 0.5, 0.7, 0.8]

for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Threshold {t}: Precision={precision:.3f}, "
          f"Recall={recall:.3f}, F1={f1:.3f}")
```

### ROC Curve and Youden's J

```python
from sklearn.metrics import roc_curve, roc_auc_score

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Find optimal threshold by Youden's J
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold (Youden's J): {optimal_threshold:.3f}")
```

## Key Properties of Threshold Tuning

| Aspect | Impact |
|--------|--------|
| **Computation** | Fast: just change classification rule, no retraining |
| **Visualization** | ROC and PR curves show all threshold choices |
| **Interpretability** | Directly controls false positive / false negative rate |
| **Limitation** | Cannot improve AUC (overall ranking), only adjust for specific operating point |

## Application to Linear Discriminant Analysis (LDA)

The threshold concept extends beyond logistic regression to any probabilistic classifier. For example, **Linear Discriminant Analysis (LDA)** also produces class probabilities $P(Y=1|\mathbf{x})$, and you can tune its threshold in the same way.

LDA classifier with threshold 0.5 (default):
$$\hat{y} = \begin{cases} 1 & \text{if } P_{\text{LDA}}(Y=1|\mathbf{x}) \geq 0.5 \\ 0 & \text{otherwise} \end{cases}$$

Changing to threshold 0.2 makes LDA more lenient, increasing recall at the cost of precision, just as with logistic regression.

## Summary

- **Default threshold 0.5** is only optimal when error costs are equal and classes are balanced
- **Lower thresholds** (0.2-0.3) increase recall but decrease precision: use when missing positives is costly
- **Higher thresholds** (0.7-0.8) increase precision but decrease recall: use when false positives are costly
- **ROC and PR curves** visualize the full tradeoff across thresholds
- **Youden's J** and **F1 score** provide automatic selection methods
- **Threshold tuning is fast and requires no retraining**, making it practical for operational adjustments
