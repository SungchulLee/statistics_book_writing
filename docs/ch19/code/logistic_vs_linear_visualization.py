"""
Logistic Regression vs Linear Regression for Binary Classification
===================================================================

This script demonstrates why logistic regression is superior to linear
regression for binary outcomes. Linear regression can produce predicted
probabilities outside [0,1], while logistic regression constrains
predictions to valid probabilities via the sigmoid function.

Key visualization:
- Left panel: Linear regression fit to binary data (invalid for probabilities)
- Right panel: Logistic regression with proper S-curve (valid probabilities)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

# =============================================================================
# Generate synthetic binary classification data
# =============================================================================

np.random.seed(42)

# Simulate credit card balance vs default
n_samples = 300
balance = np.random.uniform(0, 2500, n_samples)

# Probability of default increases with balance (logistic relationship)
true_prob = 1 / (1 + np.exp(-(balance - 1250) / 300))
default = np.random.binomial(1, true_prob)

X = balance.reshape(-1, 1)
y = default

# Create test data for smooth curves
X_test = np.linspace(balance.min(), balance.max(), 300).reshape(-1, 1)

# =============================================================================
# Fit both models
# =============================================================================

# Linear regression: treats binary outcome as continuous
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X_test)

# Logistic regression: models probability via sigmoid
logistic_model = LogisticRegression(solver='lbfgs')
logistic_model.fit(X, y)
y_pred_logistic = logistic_model.predict_proba(X_test)[:, 1]

# =============================================================================
# Visualization: Side-by-side comparison
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- LEFT PANEL: Linear Regression ---
axes[0].scatter(X[y == 0], y[y == 0], alpha=0.6, s=30,
                color='steelblue', label='No Default (y=0)')
axes[0].scatter(X[y == 1], y[y == 1], alpha=0.6, s=30,
                color='coral', label='Default (y=1)')
axes[0].plot(X_test, y_pred_linear, 'g-', linewidth=2.5, label='Linear Fit')
axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0].axhline(y=1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
axes[0].axhline(y=0.5, color='red', linestyle=':', linewidth=1.5,
                alpha=0.7, label='Decision boundary (0.5)')
axes[0].set_xlabel('Credit Card Balance ($)', fontsize=11)
axes[0].set_ylabel('Predicted Probability', fontsize=11)
axes[0].set_title('Linear Regression on Binary Data', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].set_ylim(-0.5, 1.5)
axes[0].grid(True, alpha=0.3)

# Add annotation about invalid predictions
axes[0].text(150, 1.3, 'Invalid probabilities\n(outside [0,1])',
             fontsize=9, color='red', ha='left',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# --- RIGHT PANEL: Logistic Regression ---
axes[1].scatter(X[y == 0], y[y == 0], alpha=0.6, s=30,
                color='steelblue', label='No Default (y=0)')
axes[1].scatter(X[y == 1], y[y == 1], alpha=0.6, s=30,
                color='coral', label='Default (y=1)')
axes[1].plot(X_test, y_pred_logistic, 'purple', linewidth=2.5, label='Logistic Fit')
axes[1].axhline(y=0.5, color='red', linestyle=':', linewidth=1.5,
                alpha=0.7, label='Decision boundary (0.5)')
axes[1].set_xlabel('Credit Card Balance ($)', fontsize=11)
axes[1].set_ylabel('Predicted Probability', fontsize=11)
axes[1].set_title('Logistic Regression on Binary Data', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].set_ylim(-0.05, 1.05)
axes[1].grid(True, alpha=0.3)

# Add annotation about S-curve
axes[1].text(150, 0.15, 'Sigmoid S-curve\nkeeps predictions in [0,1]',
             fontsize=9, color='green', ha='left',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig('logistic_vs_linear.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# Print summary of key differences
# =============================================================================

print("=" * 70)
print("LINEAR vs LOGISTIC REGRESSION FOR BINARY CLASSIFICATION")
print("=" * 70)

print("\n1. VALID PROBABILITY RANGES:")
print(f"   Linear regression predictions (min, max): ({y_pred_linear.min():.3f}, {y_pred_linear.max():.3f})")
print(f"   Logistic regression predictions (min, max): ({y_pred_logistic.min():.3f}, {y_pred_logistic.max():.3f})")
print(f"   Valid probability range: [0, 1]")

print("\n2. INTERPRETATION:")
print(f"   Linear: Treats binary outcome as continuous → invalid for probabilities")
print(f"   Logistic: Models probability via sigmoid → always in [0, 1]")

print("\n3. MODEL COEFFICIENTS:")
print(f"   Linear intercept: {linear_model.intercept_[0]:.4f}")
print(f"   Linear slope: {linear_model.coef_[0][0]:.6f}")
print(f"   Logistic intercept: {logistic_model.intercept_[0]:.4f}")
print(f"   Logistic slope: {logistic_model.coef_[0][0]:.6f}")

print("\n4. ODDS INTERPRETATION (Logistic only):")
# For logistic: unit increase in X multiplies odds by exp(slope)
odds_ratio = np.exp(logistic_model.coef_[0][0])
print(f"   For every $100 increase in balance, odds of default multiply by {odds_ratio:.4f}")
print(f"   Or: {(odds_ratio - 1) * 100:.2f}% increase in odds per $100")

print("\n" + "=" * 70)
