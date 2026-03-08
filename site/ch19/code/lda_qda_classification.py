"""
LDA, QDA, and Naive Bayes Classification
==========================================
Adapted from ISL (Introduction to Statistical Learning) Chapter 4 Lab.

Compares three generative classifiers with logistic regression:
1. Linear Discriminant Analysis (LDA) — shared covariance
2. Quadratic Discriminant Analysis (QDA) — class-specific covariance
3. Gaussian Naive Bayes — diagonal covariance (independence)
4. Logistic Regression — discriminative baseline

Uses synthetic 2D data with two scenarios:
  A. Classes share the same covariance (LDA should excel)
  B. Classes have different covariances (QDA should excel)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

# =============================================================================
# Main
# =============================================================================

np.random.seed(42)


def generate_shared_cov(n_per_class=200):
    """Two classes with the SAME covariance (LDA assumption holds)."""
    cov = [[1.0, 0.5], [0.5, 1.0]]
    X0 = np.random.multivariate_normal([0, 0], cov, n_per_class)
    X1 = np.random.multivariate_normal([2, 1.5], cov, n_per_class)
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def generate_diff_cov(n_per_class=200):
    """Two classes with DIFFERENT covariances (QDA should do better)."""
    cov0 = [[1.0, 0.0], [0.0, 0.3]]
    cov1 = [[0.3, 0.0], [0.0, 2.0]]
    X0 = np.random.multivariate_normal([0, 0], cov0, n_per_class)
    X1 = np.random.multivariate_normal([1.5, 1.5], cov1, n_per_class)
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def plot_decision_boundary(ax, clf, X, y, title):
    """Plot 2D decision boundary with scatter overlay."""
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="red", s=10,
               edgecolors="none", alpha=0.6, label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", s=10,
               edgecolors="none", alpha=0.6, label="Class 1")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")


def main():
    print("=" * 60)
    print("LDA, QDA, and Naive Bayes Classification Comparison")
    print("=" * 60)

    classifiers = {
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "Naive Bayes": GaussianNB(),
        "Logistic Reg": LogisticRegression(),
    }

    scenarios = {
        "Shared Covariance": generate_shared_cov(),
        "Different Covariances": generate_diff_cov(),
    }

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for row, (scenario_name, (X, y)) in enumerate(scenarios.items()):
        print(f"\n--- Scenario: {scenario_name} ---")
        print(f"  n = {len(y)}, class balance = "
              f"{np.mean(y == 0):.0%} / {np.mean(y == 1):.0%}")

        for col, (clf_name, clf) in enumerate(classifiers.items()):
            clf.fit(X, y)
            cv_acc = cross_val_score(clf, X, y, cv=10,
                                     scoring="accuracy").mean()
            train_acc = clf.score(X, y)

            print(f"  {clf_name:15s}: train acc = {train_acc:.4f}, "
                  f"10-fold CV acc = {cv_acc:.4f}")

            ax = axes[row, col]
            plot_decision_boundary(
                ax, clf, X, y,
                f"{clf_name}\nCV acc = {cv_acc:.3f}"
            )
            if col == 0:
                ax.set_ylabel(scenario_name, fontsize=11, labelpad=10)

    # Print detailed report for one scenario
    print("\n--- Confusion Matrix: QDA on Different Covariances ---")
    X_diff, y_diff = scenarios["Different Covariances"]
    qda = QuadraticDiscriminantAnalysis().fit(X_diff, y_diff)
    y_pred = qda.predict(X_diff)
    print(confusion_matrix(y_diff, y_pred))
    print(classification_report(y_diff, y_pred, target_names=["0", "1"]))

    plt.suptitle("Generative Classifiers: Decision Boundaries",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("lda_qda_classification.png", dpi=150,
                bbox_inches="tight")
    plt.show()
    print("Figure saved: lda_qda_classification.png")


if __name__ == "__main__":
    main()
