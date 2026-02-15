#!/usr/bin/env python3
# ======================================================================
# 19_evaluation_01_roc_confusion_precision_recall.py
# ======================================================================
# Classification evaluation metrics:
#   1. Confusion matrix.
#   2. Precision, recall, F1 score.
#   3. ROC curve and AUC (trapezoidal rule).
#
# Source:  Adapted from *Practical Statistics for Data Scientists*
#          (Chapter 5 — Classification).
# ======================================================================

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)


def confusion_matrix(y_true, y_pred):
    """Compute 2x2 confusion matrix [TN, FP; FN, TP]."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred):
    """Compute precision, recall, and F1 score."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def roc_curve(y_true, scores):
    """
    Compute ROC curve (FPR, TPR) for varying thresholds.

    Returns
    -------
    fpr : array   False positive rates.
    tpr : array   True positive rates.
    thresholds : array   Decision thresholds.
    """
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

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    return fpr, tpr, thresholds


def auc_trapezoid(fpr, tpr):
    """Compute AUC via the trapezoidal rule."""
    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]
    return np.trapz(tpr_sorted, fpr_sorted)


def main():
    print("Classification Evaluation Metrics")
    print("=" * 55)

    # ── Simulate a binary classification problem ──
    n = 300
    # True labels
    y_true = np.concatenate([np.ones(120), np.zeros(180)])
    # Predicted scores (positive class tends to have higher scores)
    scores = np.concatenate([
        np.random.normal(0.65, 0.25, 120),
        np.random.normal(0.35, 0.25, 180)
    ])
    scores = np.clip(scores, 0, 1)

    # ── 1. Confusion matrix at threshold = 0.5 ──
    y_pred = (scores >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n1. Confusion Matrix (threshold = 0.5):")
    print(f"              Predicted")
    print(f"              Neg   Pos")
    print(f"   Actual Neg  {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"   Actual Pos  {cm[1,0]:3d}   {cm[1,1]:3d}")

    # ── 2. Precision, recall, F1 ──
    prec, rec, f1 = precision_recall_f1(y_true, y_pred)
    accuracy = np.mean(y_true == y_pred)
    print(f"\n2. Metrics at threshold = 0.5:")
    print(f"   Accuracy  = {accuracy:.4f}")
    print(f"   Precision = {prec:.4f}")
    print(f"   Recall    = {rec:.4f}")
    print(f"   F1 Score  = {f1:.4f}")

    # ── 3. ROC curve and AUC ──
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    area = auc_trapezoid(fpr, tpr)
    print(f"\n3. ROC Analysis:")
    print(f"   AUC = {area:.4f}")

    # ── Plots ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Confusion matrix heatmap
    ax = axes[0]
    im = ax.imshow(cm, cmap='Blues', aspect='equal')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Neg', 'Pos'])
    ax.set_yticklabels(['Neg', 'Pos'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=16, color='white' if cm[i, j] > cm.max() / 2 else 'black')

    # Score distributions
    ax = axes[1]
    ax.hist(scores[y_true == 0], bins=25, alpha=0.6, label='Negative', edgecolor='k')
    ax.hist(scores[y_true == 1], bins=25, alpha=0.6, label='Positive', edgecolor='k')
    ax.axvline(0.5, color='red', linestyle='--', label='Threshold')
    ax.set_xlabel('Predicted score')
    ax.set_ylabel('Frequency')
    ax.set_title('Score Distributions')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ROC curve
    ax = axes[2]
    ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {area:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
