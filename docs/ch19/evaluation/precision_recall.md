# Precision, Recall, and F1 Score

## Precision: Positive Predictive Value

**Precision** measures the reliability of positive predictions. Of all instances we predicted as positive, what fraction were actually positive?

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
$$

Precision is the **positive predictive value (PPV)**. It answers the question: "If the model says default, how confident can we be?"

### Interpretation

- High precision means few false positives.
- Low precision means many false alarms (predicting positive when the label is negative).
- Precision = 1 means no false positives; Precision = 0 means all positive predictions were wrong.

## Recall: Sensitivity and True Positive Rate

**Recall** (also called **sensitivity** or **true positive rate**) measures how well the model identifies positive instances:

$$
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

Recall answers the question: "Of all true defaults, what fraction did we catch?"

### Interpretation

- High recall means few false negatives; we catch most positive cases.
- Low recall means we miss many positive cases.
- Recall = 1 means no false negatives; Recall = 0 means we missed all positive cases.

## The Precision-Recall Tradeoff

There is an inherent **tradeoff** between precision and recall. By lowering the classification threshold $\tau$:

- More instances are predicted positive
- Recall (TP) increases but FP also increases
- Precision decreases

Conversely, raising the threshold:

- Fewer instances are predicted positive
- Recall decreases, precision increases

The choice between precision and recall depends on the **application**:

| Domain | Priority | Rationale |
|---|---|---|
| Fraud detection | High Recall | Missing fraud (FN) is costly; some false alarms (FP) are acceptable |
| Loan approval | High Precision | False alarms (rejecting good loans) lose customers; some FN are acceptable |
| Medical diagnosis | High Recall | Missing disease (FN) endangers lives; false alarms (FP) trigger further testing |

## Example: Loan Default Data

From our logistic regression model:

```
                      Predicted
                   Default  Paid Off
Actual Default       14,336   8,335
Actual Paid Off       8,148  14,523
```

- **Precision** = $14,336 / (14,336 + 8,148) \approx 0.6376$
  - About 63.76% of predicted defaults are correct; 36.24% are false alarms.

- **Recall** = $14,336 / (14,336 + 8,335) \approx 0.6323$
  - We identify about 63.23% of actual defaults; miss 36.77%.

## F1 Score: Harmonic Mean

When precision and recall are both important, the **F1 score** provides a single summary metric:

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

The F1 score is the **harmonic mean** of precision and recall, ranging from 0 to 1. It is often preferred when:

- Classes are imbalanced
- Both false positives and false negatives are costly
- You want a single metric to balance both concerns

For our example: $F_1 = 2 \cdot \frac{0.6376 \cdot 0.6323}{0.6376 + 0.6323} \approx 0.6349$

## Weighted Metrics for Multi-Class Problems

For multi-class classification, precision, recall, and F1 scores are computed per class and then averaged:

- **Macro-average:** Simple arithmetic mean across classes (treats all classes equally)
- **Weighted average:** Weighted by class support (accounts for class imbalance)
- **Micro-average:** Computed from pooled TP, FP, FN across all classes

## Precision-Recall Curve

The **precision-recall curve** plots precision vs. recall as the classification threshold varies. It provides a more nuanced view of model performance than accuracy alone, especially for imbalanced datasets.

Key properties:

1. Points further toward the top-right (high precision, high recall) indicate better performance.
2. The curve starts at (0, 1) when the threshold is very high (predict positive for no instances).
3. The curve ends at (1, 0) when the threshold is very low (predict positive for all instances).
4. **Average Precision (AP):** The area under the precision-recall curve; ranges from 0 to 1.

When one class is rare, the precision-recall curve often provides more insight than the ROC curve, as it focuses on the behavior in the positive class region.
