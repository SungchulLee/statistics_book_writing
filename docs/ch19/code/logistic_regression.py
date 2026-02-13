"""
Chapter 13: Logistic Regression — Code Examples
================================================

Demonstrates:
1. Logistic regression with sklearn
2. Logistic regression with statsmodels (inference)
3. Confusion matrix and classification report
4. ROC curve and AUC
5. Precision-recall curve
6. Threshold selection
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Generate synthetic data
# =============================================================================

np.random.seed(42)
n = 300
hours_studied = np.random.uniform(1, 10, n)
noise = np.random.normal(0, 1, n)
logit = -3 + 0.7 * hours_studied + 0.3 * noise
prob = 1 / (1 + np.exp(-logit))
passed = np.random.binomial(1, prob)

X = hours_studied.reshape(-1, 1)
y = passed

# =============================================================================
# 1. Logistic Regression with sklearn
# =============================================================================

print("=" * 60)
print("1. LOGISTIC REGRESSION WITH SKLEARN")
print("=" * 60)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Coefficient (hours): {model.coef_[0][0]:.4f}")
print(f"Train accuracy: {model.score(X_train, y_train):.3f}")
print(f"Test accuracy:  {model.score(X_test, y_test):.3f}")

# Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# =============================================================================
# 2. Logistic Regression with statsmodels (inference)
# =============================================================================

print("\n" + "=" * 60)
print("2. LOGISTIC REGRESSION WITH STATSMODELS")
print("=" * 60)

import statsmodels.api as sm

X_sm = sm.add_constant(hours_studied)
logit_model = sm.Logit(y, X_sm)
result = logit_model.fit(disp=0)
print(result.summary())

# Odds ratios
print("\nOdds Ratios:")
print(np.exp(result.params))
print("\n95% CI for Odds Ratios:")
print(np.exp(result.conf_int()))

# Likelihood ratio test (full vs null)
null_model = sm.Logit(y, sm.add_constant(np.ones(n))).fit(disp=0)
lr_stat = -2 * (null_model.llf - result.llf)
lr_pvalue = stats.chi2.sf(lr_stat, df=1)
print(f"\nLikelihood Ratio Test: χ² = {lr_stat:.4f}, p = {lr_pvalue:.6f}")

# =============================================================================
# 3. Confusion Matrix and Classification Report
# =============================================================================

print("\n" + "=" * 60)
print("3. CONFUSION MATRIX AND CLASSIFICATION REPORT")
print("=" * 60)

from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, precision_score, recall_score,
                              f1_score)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"\nTN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")

print("\nFull Classification Report:")
print(classification_report(y_test, y_pred))

# =============================================================================
# 4. ROC Curve and AUC
# =============================================================================

print("\n" + "=" * 60)
print("4. ROC CURVE AND AUC")
print("=" * 60)

from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
print(f"AUC = {auc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC curve
axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1.05])

# =============================================================================
# 5. Precision-Recall Curve
# =============================================================================

from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
print(f"Average Precision = {ap:.4f}")

axes[1].plot(recall, precision, 'r-', lw=2, label=f'PR (AP = {ap:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()

plt.tight_layout()
plt.savefig('roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# 6. Threshold Selection
# =============================================================================

print("\n" + "=" * 60)
print("6. THRESHOLD SELECTION")
print("=" * 60)

print(f"{'Threshold':>10} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
print("-" * 42)
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_t = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred_t)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    rec = recall_score(y_test, y_pred_t, zero_division=0)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    print(f"{threshold:>10.1f} {acc:>7.3f} {prec:>7.3f} {rec:>7.3f} {f1:>7.3f}")

# Youden's J statistic: optimal threshold on ROC
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"\nOptimal threshold (Youden's J): {optimal_threshold:.3f}")
print(f"  TPR = {tpr[optimal_idx]:.3f}, FPR = {fpr[optimal_idx]:.3f}")
