# Multicollinearity and Influence

## Identifying Outliers and Influential Points

Outliers and influential points can substantially impact regression results. Understanding them is essential for model refinement.

- **Outliers** are observations with unusually large residuals, indicating they deviate substantially from the model's predictions. They may result from data recording errors or unique conditions not captured by the model.
- **Influential points** are observations that disproportionately affect the fitted regression model. Removing them would significantly change the estimated coefficients.

## Cook's Distance

**Cook's Distance** combines information on both the residual (how far a predicted value is from the actual value) and leverage (how far a predictor value is from the mean) to measure each observation's overall influence on the regression.

### Definition

For each observation $i$, Cook's Distance $D_i$ is:

$$
D_i = \frac{\sum_{j=1}^n (\hat{y}_{j} - \hat{y}_{j(i)})^2}{p \cdot s^2}
$$

where:

- $\hat{y}_j$ is the predicted value for the $j$-th observation using all data.
- $\hat{y}_{j(i)}$ is the predicted value for the $j$-th observation with the $i$-th observation removed.
- $p$ is the number of predictors (including the intercept).
- $s^2$ is the mean squared error of the model.

### Computing $s^2$

The residual variance is computed using $n - p$ as the denominator:

$$
s^2 = \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{n - p}
$$

This accounts for the degrees of freedom lost from estimating $p$ parameters. In simple linear regression ($p = 2$: one slope plus intercept), this gives $n - 2$. For multiple regression, use $n - p$ where $p$ includes all estimated parameters.

### Interpretation

A high Cook's Distance indicates that an observation has both a large residual and high leverage, meaning it strongly affects the fitted values. Common threshold choices:

- **$D_i > 4/n$**: A common heuristic scaled by sample size. As $n$ increases, the threshold decreases, making it easier to detect influential points in larger datasets.
- **$D_i > 1.0$**: A simpler, fixed threshold used in some references.
- **Visual inspection**: Plot Cook's Distance values and look for observations that clearly stand out.

The $4/n$ threshold is widely used as a practical starting point. It balances sensitivity to outliers with computational efficiency across different dataset sizes, but should be complemented with visual inspection and domain knowledge.

### Implementation: Removing Outliers with Cook's Distance

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Generate synthetic data
np.random.seed(0)
X, y = make_regression(n_samples=100, n_features=1, noise=10)
data = pd.DataFrame({'X': X.flatten(), 'y': y})

# Add artificial outliers
data.loc[95, 'y'] += 80
data.loc[96, 'y'] -= 80
data.loc[97, 'y'] += 60
data.loc[98, 'y'] -= 60

# Fit model with outliers
X_with_const = sm.add_constant(data['X'])
model = sm.OLS(data['y'], X_with_const).fit()

# Calculate Cook's Distance
influence = model.get_influence()
cooks_d, _ = influence.cooks_distance

# Identify influential points
n = len(data)
threshold = 4 / n
outliers = np.where(cooks_d > threshold)[0]

# Plot: with and without outliers
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Row 1: Original data with outliers
axes[0, 0].scatter(data['X'], data['y'], alpha=0.7, label='Data Points')
axes[0, 0].plot(data['X'], model.fittedvalues, color='orange', label='Regression Line')
axes[0, 0].set_title('Regression Plot (With Outliers)')
axes[0, 0].set_xlabel('Predictor (X)')
axes[0, 0].set_ylabel('Response (y)')
axes[0, 0].legend()

axes[0, 1].scatter(model.fittedvalues, model.resid, alpha=0.7)
axes[0, 1].axhline(0, color='red', linestyle='--')
axes[0, 1].set_title('Residual Plot (With Outliers)')
axes[0, 1].set_xlabel('Fitted Values')
axes[0, 1].set_ylabel('Residuals')

# Row 2: Data without outliers
data_no_outliers = data.drop(index=outliers)
X_with_const_no_outliers = sm.add_constant(data_no_outliers['X'])
model_no_outliers = sm.OLS(data_no_outliers['y'], X_with_const_no_outliers).fit()

axes[1, 0].scatter(data_no_outliers['X'], data_no_outliers['y'], alpha=0.7, label='Data Points')
axes[1, 0].plot(data_no_outliers['X'], model_no_outliers.fittedvalues, color='orange', label='Regression Line')
axes[1, 0].set_title('Regression Plot (Without Outliers)')
axes[1, 0].set_xlabel('Predictor (X)')
axes[1, 0].set_ylabel('Response (y)')
axes[1, 0].legend()

axes[1, 1].scatter(model_no_outliers.fittedvalues, model_no_outliers.resid, alpha=0.7)
axes[1, 1].axhline(0, color='red', linestyle='--')
axes[1, 1].set_title('Residual Plot (Without Outliers)')
axes[1, 1].set_xlabel('Fitted Values')
axes[1, 1].set_ylabel('Residuals')

plt.tight_layout()
plt.show()
```

## Multicollinearity

**Multicollinearity** occurs when predictor variables are highly correlated with each other. This can make coefficient estimates unstable and difficult to interpret, even if the model's overall predictive power remains high.

### Detecting Multicollinearity

- **Condition Number**: Reported in statsmodels output. Values above 30 suggest potential multicollinearity; very high values (e.g., $> 1000$) indicate severe issues. Interaction terms often inflate the condition number because they are inherently correlated with the original predictors.
- **Variance Inflation Factor (VIF)**: Measures how much the variance of a coefficient is inflated due to correlation with other predictors. A VIF above 5–10 suggests problematic multicollinearity.
- **Correlation Matrix**: Examining pairwise correlations between predictors can reveal strong linear relationships.

### Consequences of Multicollinearity

- Coefficient estimates become sensitive to small changes in the data.
- Standard errors of coefficients increase, making hypothesis tests less powerful.
- Individual predictor significance may be masked even when the overall model fits well.
- The model's predictive accuracy is generally unaffected, but interpretation of individual coefficients becomes unreliable.

### Addressing Multicollinearity

- **Remove redundant predictors**: If two predictors are highly correlated, consider keeping only one.
- **Center variables**: Subtracting the mean from predictors before creating interaction terms can substantially reduce multicollinearity.
- **Regularization**: Ridge regression (L2 penalty) directly addresses multicollinearity by shrinking coefficients toward zero.
- **Principal Component Regression**: Use PCA to create uncorrelated components from the original predictors.
