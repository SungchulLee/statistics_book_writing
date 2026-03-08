# Multicollinearity

## Overview

**Multicollinearity** occurs when two or more independent variables in a regression model are highly correlated. This creates challenges for estimation and inference:

- **Inflated Standard Errors**: Coefficients become unstable and have wide confidence intervals
- **Unreliable Coefficients**: Small changes in data can lead to large changes in estimated coefficients
- **Reduced Statistical Power**: Difficult to determine the individual significance of predictors
- **Model Interpretability**: Cannot reliably assess the unique contribution of each variable

Despite these issues, multicollinearity does not bias coefficient estimates and does not prevent accurate predictions on similar data. The model remains useful for forecasting even if interpretation is compromised.

---

## Detecting Multicollinearity

### Method 1: Correlation Matrix

The simplest diagnostic is to examine pairwise correlations between predictors:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Example: California Housing data
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Compute correlation matrix
corr_matrix = df.corr()

# Visualize with heatmap
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(len(housing.feature_names)), housing.feature_names, rotation=45)
plt.yticks(range(len(housing.feature_names)), housing.feature_names)
plt.title('Correlation Matrix: Housing Features')
plt.tight_layout()
plt.show()

# Print high correlations
print("High Correlations (|r| > 0.7):")
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")
```

**Limitation**: Pairwise correlations only capture two-variable relationships. Multicollinearity can exist even with modest pairwise correlations when three or more variables are involved (called **multicollinearity** to distinguish from simple bivariate correlation).

---

### Method 2: Variance Inflation Factor (VIF)

The **Variance Inflation Factor (VIF)** quantifies how much the variance of a regression coefficient is inflated due to multicollinearity with other predictors.

#### Mathematical Definition

For predictor $X_j$, the VIF is:

$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$

where $R_j^2$ is the $R^2$ from regressing $X_j$ on all other predictors.

**Interpretation**:
- **VIF = 1**: No correlation with other predictors
- **VIF < 5**: Generally acceptable (rule of thumb)
- **VIF > 5**: Concerning level of multicollinearity
- **VIF > 10**: Severe multicollinearity, often requires action

#### Using statsmodels

```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Example data
X = sm.add_constant(df[['MedInc', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude']])

# Compute VIF for each predictor (excluding constant)
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns[1:]  # Skip constant
vif_data['VIF'] = [variance_inflation_factor(X.values, i+1) for i in range(X.shape[1]-1)]

print(vif_data)
```

#### Manual VIF Calculation

Understanding how VIF is computed can provide deeper insight:

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Load data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)

features = ['MedInc', 'AveRooms', 'AveOccup', 'Latitude', 'Longitude']
X = df[features]

print("Manual VIF Calculation:")
print("=" * 60)

for j, target_feature in enumerate(features):
    # Step 1: Regress target_feature on all other features
    other_features = [f for f in features if f != target_feature]
    X_j = X[target_feature].values.reshape(-1, 1)
    X_others = X[other_features].values

    # Fit model: target_feature ~ other_features
    model = LinearRegression()
    model.fit(X_others, X_j.ravel())

    # Compute R² for this regression
    y_pred_j = model.predict(X_others)
    ss_res = np.sum((X_j.ravel() - y_pred_j) ** 2)
    ss_tot = np.sum((X_j.ravel() - X_j.mean()) ** 2)
    r2_j = 1 - (ss_res / ss_tot)

    # VIF = 1 / (1 - R²)
    vif_j = 1 / (1 - r2_j)

    print(f"{target_feature:12s}:  R² = {r2_j:.4f},  VIF = {vif_j:7.2f}")

print("=" * 60)
```

**Example Output**:
```
MedInc      :  R² = 0.0542,  VIF =    1.06
AveRooms    :  R² = 0.5621,  VIF =    2.29
AveOccup    :  R² = 0.7854,  VIF =    4.65
Latitude    :  R² = 0.9214,  VIF =   11.68
Longitude   :  R² = 0.9156,  VIF =   10.98
```

In this example, Latitude and Longitude have very high VIFs (>10), indicating severe multicollinearity. This makes sense geographically—nearby regions have similar latitude/longitude values.

---

## Addressing Multicollinearity

### Option 1: Remove Redundant Predictors

If two predictors are highly correlated, remove one:

```python
# Check which features contribute least (lowest VIF)
# or have weakest relationship with the target
# and remove those
features_reduced = ['MedInc', 'AveRooms', 'AveOccup']  # Drop Lat/Long
X_reduced = sm.add_constant(df[features_reduced])
model_reduced = sm.OLS(df['PRICE'], X_reduced).fit()
```

### Option 2: Combine Correlated Predictors

Create a composite index from correlated variables:

```python
# Combine latitude and longitude into a single "location" index
df['Location'] = (df['Latitude'] + df['Longitude']) / 2
```

### Option 3: Regularization (Ridge or Lasso Regression)

Use penalty-based methods that shrink coefficients:

```python
from sklearn.linear_model import Ridge, Lasso

# Ridge regression with alpha=1.0
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print(ridge.coef_)

# Lasso regression with alpha=0.1
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
print(lasso.coef_)
```

### Option 4: Principal Component Analysis (PCA)

Transform correlated predictors into uncorrelated principal components:

```python
from sklearn.decomposition import PCA

# Create principal components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

# Fit model with principal components
model_pca = LinearRegression()
model_pca.fit(X_pca, y)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

---

## Key Concepts

Understanding multicollinearity involves recognizing that:

1. **It's a data problem, not a modeling problem**: The issue comes from the structure of the data, not from the choice of model.

2. **Prediction vs. Inference**: Multicollinearity primarily affects inference (understanding variable effects). Predictions on similar data remain reliable.

3. **Domain Context Matters**: Sometimes keeping correlated variables is justified for interpretability or domain understanding, accepting the cost of inflated standard errors.

4. **Remedies Have Trade-offs**: Removing variables loses information. Regularization introduces bias but reduces variance. PCA is rotation-invariant but harder to interpret.

---

## Summary

Understanding multicollinearity is essential for applying statistical methods correctly in practice. Use VIF to detect it, understand the consequences for your modeling goals, and choose an appropriate remedy based on your context.
