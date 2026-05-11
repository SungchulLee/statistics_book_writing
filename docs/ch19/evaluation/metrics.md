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

## McFadden's Pseudo-R-squared
Unlike linear regression, logistic regression has no natural $R^2$. **McFadden's pseudo-$R^2$** provides a rough analog:

$$R^2_{\text{McFadden}} = 1 - \frac{\ell(\hat{\boldsymbol{\beta}})}{\ell(\hat{\beta}_0)}$$

where $\ell(\hat{\boldsymbol{\beta}})$ is the log-likelihood of the full model and $\ell(\hat{\beta}_0)$ is the log-likelihood of the null model (intercept only). Values of 0.2–0.4 are considered good in practice.


## Exercises

**Exercise 1.**
Describe the main concept of Confusion Matrix, ROC Curve, and Classification Metrics and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Confusion Matrix, ROC Curve, and Classification Metrics is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

---

**Exercise 2.**
State the key assumptions required by the method discussed here. How can each assumption be checked?

??? success "Solution to Exercise 2"
    The main assumptions typically include: (1) independence of observations -- verified by understanding the data collection process and checking for serial correlation; (2) distributional requirements (e.g., normality) -- checked with Q-Q plots and formal tests like Shapiro-Wilk; (3) equal variances (if applicable) -- assessed with boxplots and Levene's test. When assumptions are violated, consider robust alternatives, transformations, or nonparametric methods.

---

**Exercise 3.**
Work through a small numerical example illustrating the application of the technique from this section.

??? success "Solution to Exercise 3"
    A structured approach to applying this technique involves: (1) clearly stating the hypotheses or estimation goal; (2) verifying that the data meet the required assumptions; (3) computing the relevant test statistic, estimate, or model fit; (4) obtaining the p-value, confidence interval, or posterior distribution; (5) interpreting the result in the context of the original question. Following these steps systematically ensures a rigorous and reproducible analysis.

---

**Exercise 4.**
Compare the approach from this section with an alternative method. When would you choose each?

??? success "Solution to Exercise 4"
    The method discussed here is appropriate when its assumptions hold and the sample size is sufficient for the asymptotic approximations to be accurate. Alternative approaches include: (1) nonparametric methods -- preferred when distributional assumptions are suspect; (2) bootstrap methods -- useful when analytical reference distributions are unavailable; (3) Bayesian methods -- valuable when incorporating prior information or when direct probability statements about parameters are desired. Running multiple approaches and comparing results provides a useful robustness check.
