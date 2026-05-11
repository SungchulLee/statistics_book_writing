# Confusion Matrix


## Definition and Setup

The **confusion matrix** (also called a **contingency table**) summarizes the performance of a classification model by comparing predicted labels to actual labels. For binary classification, it is a 2×2 table:

$$
\begin{array}{c|cc}
& \text{Predicted Negative} & \text{Predicted Positive} \\
\hline
\text{Actual Negative} & \text{TN} & \text{FP} \\
\text{Actual Positive} & \text{FN} & \text{TP}
\end{array}
$$

### The Four Outcomes

- **True Negative (TN):** Predicted negative, actually negative. Correct prediction.
- **False Positive (FP):** Predicted positive, actually negative. Type I error.
- **False Negative (FN):** Predicted negative, actually positive. Type II error.
- **True Positive (TP):** Predicted positive, actually positive. Correct prediction.

The total number of observations is $n = \text{TN} + \text{FP} + \text{FN} + \text{TP}$.

## Example: Loan Default Prediction

For a logistic regression model trained on 45,342 loan observations with binary outcome (default vs. paid off):

```
                      Predicted
                   Default  Paid Off
Actual Default       14,336   8,335
Actual Paid Off       8,148  14,523
```

Here:

- **TN = 14,523:** Correctly predicted paid-off loans
- **FP = 8,148:** Predicted default but actually paid off (false alarm)
- **FN = 8,335:** Predicted paid-off but actually defaulted (missed default)
- **TP = 14,336:** Correctly predicted defaults

## Accuracy

The most basic performance metric is **accuracy**, the proportion of correct predictions:

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{n}
$$

For our example: $\text{Accuracy} = \frac{14,336 + 14,523}{45,342} \approx 0.6365$ (63.65%).

However, accuracy alone can be misleading, especially with **imbalanced datasets** where one class is much more common than the other. It is better to examine class-specific metrics (see next section).

## Class-Specific Rates

### Positive Class (e.g., "Default")

- **Sensitivity (Recall):** What proportion of actual positives did we identify?

  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

- **Specificity:** What proportion of actual negatives did we correctly identify?

  $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

For our example:

- Recall = $14,336 / (14,336 + 8,335) \approx 0.6323$ (63.23% of actual defaults detected)
- Specificity = $14,523 / (14,523 + 8,148) \approx 0.6406$ (64.06% of actual paid-offs identified)

## Prevalence Adjustment

When the prevalence of the positive class differs between training and test sets, the confusion matrix changes accordingly. This is important in medical and fraud detection applications where the rare event is of primary interest.

## Visualization

The confusion matrix is often visualized as a heatmap with cell values and color intensity:

```
                 Predicted
                Neg    Pos
Actual Neg  [14523]  [8148]
Actual Pos  [ 8335] [14336]
```

This visual form makes it easy to see where the model makes errors: larger off-diagonal values indicate higher misclassification rates.

## Exercises

**Exercise 1.**
Model Evaluation

A spam classifier produces the following confusion matrix on a test set of 1000 emails:

|  | Predicted Spam | Predicted Not Spam |
|---|---|---|
| **Actual Spam** | 85 | 15 |
| **Actual Not Spam** | 30 | 870 |

**(a)** Compute accuracy, precision, recall, specificity, and F1 score.

**(b)** If the cost of a missed spam is 3× the cost of a false alarm, should you lower or raise the classification threshold?

**(c)** What is the False Positive Rate?

??? success "Solution to Exercise 1"

    **(a)** TP=85, FN=15, FP=30, TN=870

    Accuracy $= (85+870)/1000 = 0.955$

    Precision $= 85/(85+30) = 0.739$

    Recall $= 85/(85+15) = 0.850$

    Specificity $= 870/(870+30) = 0.967$

    F1 $= 2 \times 0.739 \times 0.850/(0.739+0.850) = 0.791$

    **(b)** Since missed spam (FN) is costlier, we should **lower** the threshold to increase recall (catch more spam), accepting more false positives.

    **(c)** FPR $= FP/(FP+TN) = 30/900 = 0.033$
