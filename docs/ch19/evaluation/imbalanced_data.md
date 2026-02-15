# Handling Imbalanced Data

## The Problem: Class Imbalance

In many real-world classification problems, the classes are **not equally represented**. Examples include:

- **Loan default:** Default rate ~5-20%, most loans pay off
- **Fraud detection:** Frauds typically <1% of all transactions
- **Disease diagnosis:** Rare diseases <5% prevalence
- **Spam detection:** Spam usually <10-20% of email

A naive classifier trained on imbalanced data tends to **predict the majority class too often**, achieving high accuracy while poorly identifying the rare (positive) class. For example, a model that always predicts "paid off" on loan data with 81% paid-off rate achieves 81% accuracy but catches 0 defaults.

### The Prevalence Ratio Problem

When training on imbalanced data, the model learns class probabilities from the empirical distribution. If the training set has a different prevalence than the target population, predicted probabilities become poorly calibrated for the minority class.

## Strategy 1: Adjustment Through Weighting

One approach is to increase the **cost (weight) of minority class errors** during training:

$$
\ell_{\text{weighted}} = -\sum_{i=1}^{n} w_i \left[ y^{(i)} \log \hat{p}_i + (1 - y^{(i)}) \log(1 - \hat{p}_i) \right]
$$

where $w_i$ is a weight assigned to observation $i$.

### Class Weights

A common choice is **inverse class frequency weighting**:

$$
w_{\text{minority}} = \frac{1}{p_{\text{minority}}}, \quad w_{\text{majority}} = \frac{1}{p_{\text{majority}}}
$$

or simply: $w_{\text{minority}} = 1, w_{\text{majority}} = p_{\text{minority}} / p_{\text{majority}}$.

### Example: Loan Data

With 18.9% default rate and 81.1% paid-off rate:

| Unweighted | Weighted |
|---|---|
| Predicted defaults: 0.98% | Predicted defaults: 61.8% |
| Model undershoots defaults by 20× | Much closer to actual prevalence |

### Implementation

In scikit-learn:

```python
from sklearn.linear_model import LogisticRegression

# Option 1: Automatic balance
model = LogisticRegression(class_weight='balanced')

# Option 2: Custom weights
weights = [1.0 if y == 'paid_off' else 5.3 for y in y_train]
model.fit(X_train, y_train, sample_weight=weights)
```

**Advantages:**
- Simple to implement
- Computationally efficient
- Preserves original dataset size

**Disadvantages:**
- Requires choosing an appropriate weight ratio
- Can still bias probability estimates

## Strategy 2: Resampling

### Undersampling (Downsampling)

**Undersampling** removes majority class instances to balance the dataset:

```
Original:   81,105 paid off + 18,895 default
Undersampled: 18,895 paid off + 18,895 default
```

**Advantages:**
- Simple; balances classes perfectly
- Reduces training time

**Disadvantages:**
- **Loss of information:** Discards majority class data
- Increases variance; less stable estimates
- Biases probability estimates toward 50-50

### Oversampling (Upsampling)

**Oversampling** duplicates minority class instances:

```
Original:   81,105 paid off + 18,895 default
Oversampled: 81,105 paid off + 81,105 default (via replication)
```

**Disadvantages:**
- Creates redundant copies
- Can lead to overfitting
- Inflates dataset size

## Strategy 3: Synthetic Minority Over-Sampling (SMOTE)

**SMOTE** generates **synthetic samples** in the minority class by interpolating between neighboring minority instances. Rather than exact replication, it creates new synthetic points along the line segment connecting nearby minority examples.

### How SMOTE Works

For each minority sample $x_i$:

1. Find $k$ nearest neighbors in the minority class (typically $k=5$).
2. Randomly select one neighbor $x_{\text{neighbor}}$.
3. Generate a synthetic point:
   $$x_{\text{synthetic}} = x_i + \lambda (x_{\text{neighbor}} - x_i)$$
   where $\lambda \in [0, 1]$ is random.

### Example: Loan Data with SMOTE

```python
from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
# Result: 50-50 split of defaults and paid-offs (synthetic defaults added)

model = LogisticRegression()
model.fit(X_resampled, y_resampled)
```

**Advantages:**
- Creates realistic synthetic samples via interpolation
- Avoids exact duplication (less overfitting than naive oversampling)
- Preserves local neighborhood structure
- Widely used in practice

**Disadvantages:**
- More complex than simple resampling
- Computational overhead for finding neighbors
- Assumes continuous features (discrete/categorical features need adaptation)

### Variants of SMOTE

- **BorderlineSMOTE:** Focuses on minority samples near the decision boundary
- **ADASYN (Adaptive Synthetic Sampling):** Generates more samples for harder-to-learn minority instances
- **SVMSMOTE:** Uses SVM decision boundary to guide synthetic sample generation

```python
from imblearn.over_sampling import BorderlineSMOTE, ADASYN

# BorderlineSMOTE
X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X_train, y_train)

# ADASYN
X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)
```

## Strategy 4: Threshold Adjustment

Rather than changing the data or loss function, adjust the **decision threshold** post-hoc to match the desired **operating point**:

- For imbalanced data, the default threshold of 0.5 is often suboptimal
- Use ROC/PR curves to select a threshold that achieves desired precision/recall tradeoff
- Adjust based on business costs and constraints

### Example

If training data has 20% defaults but deployment target has 10%, lower the threshold to predict fewer positives and achieve calibration.

## Comparison and Recommendations

| Method | Simplicity | Data Efficiency | Computation | Calibration | Use Case |
|---|---|---|---|---|---|
| **No adjustment** | High | High | Low | Poor | Balanced data only |
| **Weighting** | High | High | Low | Reasonable | When computational efficiency matters |
| **Undersampling** | High | Low | Low | Poor | Small datasets; extreme imbalance |
| **Oversampling** | High | Medium | Low | Poor | Risk of overfitting |
| **SMOTE** | Medium | High | Medium | Good | Recommended for most cases |
| **Threshold tuning** | High | High | Low | Good | Combined with other methods |

## Best Practice: Combined Approach

A robust strategy often combines multiple techniques:

1. **Use SMOTE** to generate balanced training data
2. **Fit the model** on resampled data
3. **Use class weights** as regularization
4. **Tune the decision threshold** on a validation set that preserves the true class distribution
5. **Evaluate on test data** with the original imbalance using precision/recall/ROC curves

Example workflow:

```python
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# Step 1: Resample training data
X_train_resampled, y_train_resampled = SMOTE().fit_resample(X_train, y_train)

# Step 2: Train with class weights for regularization
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_resampled, y_train_resampled)

# Step 3: Tune threshold on validation set (with original imbalance)
y_val_proba = model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_val_proba)
# Select optimal threshold (e.g., via Youden's J or cost-based method)

# Step 4: Evaluate on test set using selected threshold
y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
```

## Key Takeaway

Class imbalance requires explicit handling. Simply increasing accuracy is insufficient; focus on **recall (catching rare events)** and **calibration (realistic probability estimates)** using the appropriate combination of techniques for your domain.