# Confusion Matrix, ROC Curve, and Classification Metrics

## The Classification Decision

Logistic regression outputs a predicted probability $\hat{p} = P(Y = 1 \mid \mathbf{x})$. To make a binary classification decision, we apply a **threshold** $c$ (default $c = 0.5$):

$$\hat{Y} = \begin{cases} 1 & \text{if } \hat{p} \geq c \\ 0 & \text{if } \hat{p} < c \end{cases}$$

The choice of threshold affects all classification metrics.

## Confusion Matrix

The **confusion matrix** tabulates predictions against true labels:

|  | Predicted Positive ($\hat{Y} = 1$) | Predicted Negative ($\hat{Y} = 0$) |
|---|---|---|
| **Actual Positive** ($Y = 1$) | True Positive (TP) | False Negative (FN) |
| **Actual Negative** ($Y = 0$) | False Positive (FP) | True Negative (TN) |

All classification metrics are derived from these four counts.

## Primary Metrics

### Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

The proportion of all predictions that are correct. **Limitation:** misleading with imbalanced classes. If 95% of transactions are legitimate, a model that always predicts "legitimate" achieves 95% accuracy but catches zero fraud.

### Precision (Positive Predictive Value)

$$\text{Precision} = \frac{TP}{TP + FP}$$

"Of all observations predicted positive, how many actually are?" High precision means few false alarms.

### Recall (Sensitivity, True Positive Rate)

$$\text{Recall} = \frac{TP}{TP + FN}$$

"Of all actual positives, how many did we catch?" High recall means few missed positives.

### Specificity (True Negative Rate)

$$\text{Specificity} = \frac{TN}{TN + FP}$$

"Of all actual negatives, how many did we correctly identify?"

### F1 Score

The harmonic mean of precision and recall:

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2\,TP}{2\,TP + FP + FN}$$

The F1 score balances precision and recall when both matter equally.

### Precision–Recall Tradeoff

Lowering the threshold $c$ increases recall (catch more positives) but decreases precision (more false positives). The optimal threshold depends on the application:

- **Medical screening:** prioritize recall (don't miss sick patients)
- **Spam filtering:** prioritize precision (don't misclassify good emails)
- **Fraud detection:** balance depends on cost of missed fraud vs false alerts

## ROC Curve

### Definition

The **Receiver Operating Characteristic (ROC) curve** plots the True Positive Rate (Recall) against the False Positive Rate ($\text{FPR} = FP/(FP + TN) = 1 - \text{Specificity}$) for all possible threshold values $c \in [0, 1]$.

### Interpretation

- A perfect classifier hugs the top-left corner: TPR = 1, FPR = 0
- The diagonal line represents a random classifier (no discrimination)
- The ROC curve is **threshold-free** — it summarizes performance across all thresholds

### AUC (Area Under the ROC Curve)

$$\text{AUC} = \int_0^1 \text{ROC}(t)\, dt$$

| AUC | Interpretation |
|---|---|
| 1.0 | Perfect classifier |
| 0.9–1.0 | Excellent |
| 0.8–0.9 | Good |
| 0.7–0.8 | Fair |
| 0.5–0.7 | Poor |
| 0.5 | Random (no discrimination) |

**Probabilistic interpretation:** AUC equals the probability that the model ranks a randomly chosen positive instance higher than a randomly chosen negative instance:

$$\text{AUC} = P(\hat{p}_{\text{positive}} > \hat{p}_{\text{negative}})$$

## Precision–Recall Curve

For imbalanced datasets, the **PR curve** (Precision vs Recall) is more informative than the ROC curve. The **Average Precision (AP)** summarizes the PR curve:

$$\text{AP} = \sum_k (R_k - R_{k-1}) \cdot P_k$$

where $P_k$ and $R_k$ are precision and recall at the $k$-th threshold.

## Log-Loss (Cross-Entropy Loss)

The **log-loss** directly measures the quality of predicted probabilities (not just classifications):

$$\text{Log-Loss} = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\right]$$

Lower log-loss indicates better-calibrated probabilities. A perfect model has log-loss = 0.

## Choosing the Right Metric

| Scenario | Recommended Metric |
|---|---|
| Balanced classes, equal costs | Accuracy, F1 |
| Imbalanced classes | AUC, F1, Average Precision |
| Cost of FN >> cost of FP | Recall, then F1 |
| Cost of FP >> cost of FN | Precision, then F1 |
| Probability calibration matters | Log-loss, Brier score |
| Comparing models overall | AUC |

## McFadden's Pseudo-$R^2$

Unlike linear regression, logistic regression has no natural $R^2$. **McFadden's pseudo-$R^2$** provides a rough analog:

$$R^2_{\text{McFadden}} = 1 - \frac{\ell(\hat{\boldsymbol{\beta}})}{\ell(\hat{\beta}_0)}$$

where $\ell(\hat{\boldsymbol{\beta}})$ is the log-likelihood of the full model and $\ell(\hat{\beta}_0)$ is the log-likelihood of the null model (intercept only). Values of 0.2–0.4 are considered good in practice.
