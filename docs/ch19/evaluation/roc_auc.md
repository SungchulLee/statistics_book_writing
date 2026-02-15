# ROC Curve and AUC

## Receiver Operating Characteristic (ROC) Curve

The **ROC curve** (Receiver Operating Characteristic) is a powerful tool for evaluating binary classifiers across all possible classification thresholds. It plots:

- **x-axis:** False Positive Rate (FPR) = $1 - \text{Specificity}$
- **y-axis:** True Positive Rate (TPR) = Sensitivity / Recall

### Computing the ROC Curve

For a classifier that outputs probability scores, we vary the decision threshold $\tau$ from 0 to 1:

1. For each threshold $\tau$: classify as positive if $\hat{p} \geq \tau$.
2. Compute FPR and TPR:
   $$\text{TPR} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
3. Plot the (FPR, TPR) pair.
4. Repeat for all threshold values; connect the points to form the curve.

### Interpretation

The ROC curve visualizes the **tradeoff between sensitivity and specificity**:

- **(0, 1):** Perfect classifier (TPR = 1, FPR = 0)
- **(0, 0):** Threshold so high that we predict positive for nothing
- **(1, 1):** Threshold so low that we predict positive for everything
- **Main diagonal (y = x):** Random classifier with no discrimination ability

A classifier above the diagonal is better than random; one below is worse than random (reverse predictions).

## Example: Loan Default ROC Curve

For the logistic regression model on loan data:

```
At threshold = 0.5:
TPR = 14,336 / 22,671 ≈ 0.6323
FPR = 8,148 / 22,671 ≈ 0.3594
```

Varying the threshold from 0 to 1 produces a curve that typically starts near (0, 0) and ends near (1, 1), bulging upward for a good classifier.

## Area Under the Curve (AUC)

The **AUC** is the area under the ROC curve, ranging from 0 to 1:

$$
\text{AUC} = \int_0^1 \text{TPR}(\text{FPR}) \, d(\text{FPR})
$$

In practice, AUC is computed numerically using the **trapezoidal rule**:

$$
\text{AUC} \approx \sum_{i=1}^{m} \frac{\text{TPR}_i + \text{TPR}_{i-1}}{2} \cdot (\text{FPR}_i - \text{FPR}_{i-1})
$$

### Interpretation of AUC

| AUC Value | Interpretation |
|---|---|
| 0.5 | Random classifier; no discrimination ability |
| 0.6–0.7 | Poor to fair discrimination |
| 0.7–0.8 | Acceptable discrimination |
| 0.8–0.9 | Excellent discrimination |
| 0.9–1.0 | Outstanding discrimination |
| 1.0 | Perfect classifier |

For our loan default model: **AUC ≈ 0.6917**, indicating fair to acceptable discrimination.

## Probabilistic Interpretation of AUC

An elegant interpretation of AUC comes from **rank statistics**:

**AUC = Probability that the model ranks a random positive instance higher than a random negative instance.**

In other words, if you randomly sample one default and one non-default loan, AUC is the probability that the model assigns a higher probability to the default. An AUC of 0.5 means the rankings are random; an AUC of 1.0 means the model always ranks positives higher than negatives.

## Advantages of AUC

1. **Threshold-independent:** Summarizes performance across all thresholds in a single number.
2. **Handles class imbalance well:** Unlike accuracy, AUC is not biased by imbalanced datasets.
3. **Probabilistic interpretation:** Has a clear statistical meaning.
4. **Useful for ranking tasks:** Directly applicable to scoring and ranking problems.

## Comparison: ROC vs. Precision-Recall

| Aspect | ROC Curve | PR Curve |
|---|---|---|
| **Focus** | Sensitivity vs. Specificity | Precision vs. Recall |
| **Threshold-independent** | Yes | Yes |
| **Class imbalance** | Less sensitive to imbalance | More sensitive; better for rare events |
| **Use case** | Balanced classes | Imbalanced classes (rare positives) |
| **Metric** | AUC (0 to 1) | Average Precision (0 to 1) |

For datasets with severe class imbalance (e.g., fraud detection with 1% positives), the **precision-recall curve** often reveals model behavior more clearly than the ROC curve.

## Threshold Selection

The ROC curve helps select an optimal threshold for deployment. Common criteria include:

1. **Youden's J Statistic:** $J = \text{TPR} - \text{FPR}$; choose the threshold maximizing $J$.
2. **Cost-based:** Incorporate misclassification costs: $\min_\tau [c_{FP} \cdot \text{FPR} + c_{FN} \cdot (1 - \text{TPR})]$
3. **Application-specific:** Choose based on business requirements (e.g., target a specific recall level).

For the loan default model using Youden's J, the optimal threshold might differ significantly from the default 0.5, depending on the relative costs of false positives and false negatives.
