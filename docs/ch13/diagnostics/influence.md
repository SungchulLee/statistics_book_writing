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

### Computing s-squared
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
## Exercises

**Exercise 1.**
In a regression with $p = 3$ predictors and $n = 50$ observations, an observation has leverage $h_{ii} = 0.18$. Determine whether this is a high-leverage point using the standard threshold and explain what high leverage means geometrically.

??? success "Solution to Exercise 1"
    The standard threshold for high leverage is $2(p+1)/n = 2 \times 4/50 = 0.16$. Since $h_{ii} = 0.18 > 0.16$, this is a high-leverage point.

    Geometrically, leverage measures how far an observation's predictor values are from the center of the predictor space. A high-leverage point lies far from the mean of the $X$ values, giving it disproportionate influence on the regression line. The regression line is "pulled" toward high-leverage points.

---

**Exercise 2.**
Observation 17 has Cook's distance $D_{17} = 0.95$ in a regression with $n = 30$ and $p = 2$. Using the threshold $D > 4/n$, assess its influence and describe what would happen to the regression if this observation were removed.

??? success "Solution to Exercise 2"
    The threshold is $4/n = 4/30 = 0.133$. Since $D_{17} = 0.95 \gg 0.133$, this observation is highly influential.

    Removing observation 17 would substantially change the estimated regression coefficients $\hat{\beta}$. The direction and magnitude of the change depend on whether the observation has a large residual (pulling the line toward it) or lies along the current trend. The researcher should investigate whether this point is a data error, an outlier from a different population, or a valid but extreme observation.

---

**Exercise 3.**
Explain the relationship between leverage ($h_{ii}$), studentized residual ($r_i$), and Cook's distance ($D_i$). Can an observation have high Cook's distance but low leverage?

??? success "Solution to Exercise 3"
    Cook's distance combines leverage and residual size:

    $$
    D_i = \frac{r_i^2}{p+1} \cdot \frac{h_{ii}}{1 - h_{ii}}
    $$

    where $r_i$ is the internally studentized residual and $h_{ii}$ is the leverage.

    An observation can have high $D_i$ with moderate leverage if its studentized residual is very large (a clear outlier near the center of the predictor space). However, in practice, truly high Cook's distance usually involves at least moderate leverage, because observations near the center of $X$ have limited ability to shift the entire regression line even with a large residual.
