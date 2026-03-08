# Residual Analysis

Residual analysis is a critical step in evaluating how well a linear regression model fits a dataset. By examining the residuals—differences between observed values and predicted values—we gain insights into the model's accuracy and whether key assumptions hold.

## Understanding Residuals

A residual is the difference between an observed value and its corresponding predicted value:

$$
e_i = y_i - \hat{y}_i
$$

where $y_i$ is the observed value and $\hat{y}_i$ is the predicted value from the regression model.

$$\begin{array}{lll}
\text{Predict} && \hat{y}_i \\
\text{Residual} && y_i - \hat{y}_i \\\hline
\text{Actual} && y_i
\end{array}$$

Residuals reveal how well the model fits each observation. Ideally, they should be as close to zero as possible, indicating that predictions align closely with observed values.

## Key Assumptions

Residual analysis checks the following assumptions essential for reliable linear regression:

- **Linearity**: The relationship between predictors and the response should be linear. Residuals should not show patterns when plotted against predicted values.
- **Independence**: The residuals should be independent of each other, particularly important in time series data.
- **Homoscedasticity (Constant Variance)**: The variance of residuals should remain constant across all levels of the independent variables.
- **Normality**: Residuals should ideally follow a normal distribution, especially when the model is used for inference.

## Residual Plots

### Residuals vs. Fitted Values Plot

This plot helps identify issues with **linearity** and **homoscedasticity**. Ideally, residuals scatter randomly around the horizontal line at zero without forming patterns. A curved pattern suggests non-linearity; a fan or funnel shape suggests heteroscedasticity.

**Distinction from Residual Plot**: The Residuals vs. Fitted Values plot uses fitted (predicted) values on the x-axis for a global model check. A general "Residual Plot" may use individual predictors or observation indices on the x-axis to diagnose specific predictor relationships or time trends.

| Feature | Residuals vs. Fitted Values Plot | Residual Plot |
|---|---|---|
| **Primary Purpose** | Check linearity and homoscedasticity | Assess individual predictor relationships |
| **X-axis** | Fitted values (predicted values) | Predictor variable or observation index |
| **When to Use** | After fitting to check global assumptions | To diagnose specific predictors or time effects |

#### Implementation with statsmodels

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import make_regression

# Create synthetic dataset
np.random.seed(0)
X, y = make_regression(n_samples=100, n_features=1, noise=10)
data = pd.DataFrame({'X': X.flatten(), 'y': y})

# Fit OLS model
X_with_const = sm.add_constant(data['X'])
model = sm.OLS(data['y'], X_with_const).fit()

data['Fitted'] = model.fittedvalues
data['Residuals'] = model.resid

# Create plots
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3))

# Regression plot
ax0.scatter(data['X'], data['y'], alpha=0.7, label='Data Points')
ax0.plot(data['X'], data['Fitted'], color='orange', label='Regression Line')
ax0.set_title('Regression Plot')
ax0.set_xlabel('Predictor (X)')
ax0.set_ylabel('Response (y)')
ax0.legend()

# Residuals vs. Fitted Values Plot
ax1.scatter(data['Fitted'], data['Residuals'], alpha=0.7)
ax1.axhline(y=0, color='r', linestyle='--')
ax1.set_title('Residuals vs. Fitted Values Plot')
ax1.set_xlabel('Fitted Values')
ax1.set_ylabel('Residuals')

plt.tight_layout()
plt.show()
```

#### Interpreting the Plot

1. **Random Scatter**: Residuals scattered randomly around zero suggest the linearity assumption holds.
2. **Constant Spread**: Roughly constant spread across fitted values supports homoscedasticity.
3. **Patterns or Funnel Shape**: A curved pattern indicates non-linearity; a funnel shape indicates heteroscedasticity.

### Good Case: Linear Data with Linear Model

When data is truly linear and we fit a linear model, residuals scatter randomly with constant variance:

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def generate_data(n=50, noise_level=3.0, seed=0):
    np.random.seed(seed)
    x = np.random.randn(n, 1)
    x.sort(axis=0)
    noise = np.random.normal(0, 1, size=x.shape)
    y = (1 + 2 * x + noise_level * noise).reshape((-1,))
    return x, y

def perform_regression(x, y):
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    return model, y_pred

def plot_regression_and_residuals(x, y, y_pred):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3))

    ax0.plot(x, y, 'o', label="Data")
    ax0.plot(x, y_pred, '-b', label="Predicted")
    ax0.set_title('Regression Plot')
    ax0.legend()

    ax1.plot(x, y - y_pred, 'o', label="Residuals")
    ax1.set_title('Residual Plot')
    ax1.legend()

    for ax in (ax0, ax1):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position("zero")

    plt.tight_layout()
    plt.show()

x, y = generate_data()
model, y_pred = perform_regression(x, y)
plot_regression_and_residuals(x, y, y_pred)
```

### Bad Case: Polynomial Data with Linear Model

When data has a polynomial relationship but we fit only a linear model, residuals show a clear curved pattern—a sign that linearity is violated:

```python
def generate_data(n=50, noise_level=3.0, d=1, seed=0):
    np.random.seed(seed)
    x = np.random.randn(n, 1)
    x.sort(axis=0)
    noise = np.random.normal(0, 1, size=x.shape)
    y = (1 + np.sum([(k+1) * x**k for k in range(1, d+1)], axis=0) + noise_level * noise).reshape((-1,))
    return x, y

# Generate quadratic data but fit linear model
x, y = generate_data(d=2)
model, y_pred = perform_regression(x, y)
plot_regression_and_residuals(x, y, y_pred)
```

#### Linear vs. Quadratic Residual Comparison

To better diagnose model misspecification, directly comparing residuals from competing models can be insightful. Consider data that follows a quadratic relationship:

```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess

# Generate quadratic data
np.random.seed(42)
x = np.random.uniform(-3, 3, 100)
y_true = 2 + 0.5 * x - 1.5 * x**2
y = y_true + np.random.normal(0, 1, len(x))

# Fit linear model
X_linear = sm.add_constant(x)
model_linear = sm.OLS(y, X_linear).fit()
residuals_linear = model_linear.resid
y_pred_linear = model_linear.fittedvalues

# Fit quadratic model
X_quad = sm.add_constant(np.column_stack([x, x**2]))
model_quad = sm.OLS(y, X_quad).fit()
residuals_quad = model_quad.resid
y_pred_quad = model_quad.fittedvalues

# Plot residuals with LOWESS smoothing
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Linear model residuals
ax1.scatter(y_pred_linear, residuals_linear, alpha=0.6)
ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)

# Add LOWESS smooth to highlight curvature
lowess_result = lowess(residuals_linear, y_pred_linear, frac=0.3)
ax1.plot(lowess_result[:, 0], lowess_result[:, 1], 'b-', linewidth=2.5,
         label='LOWESS Trend')

ax1.set_xlabel('Fitted Values')
ax1.set_ylabel('Residuals')
ax1.set_title('Linear Model: Clear Non-linearity Pattern')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Quadratic model residuals
ax2.scatter(y_pred_quad, residuals_quad, alpha=0.6)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)

# Add LOWESS smooth
lowess_result_quad = lowess(residuals_quad, y_pred_quad, frac=0.3)
ax2.plot(lowess_result_quad[:, 0], lowess_result_quad[:, 1], 'b-', linewidth=2.5,
         label='LOWESS Trend')

ax2.set_xlabel('Fitted Values')
ax2.set_ylabel('Residuals')
ax2.set_title('Quadratic Model: Non-linearity Removed')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary comparison
print("Model Comparison:")
print(f"Linear Model R²:    {model_linear.rsquared:.4f}")
print(f"Quadratic Model R²: {model_quad.rsquared:.4f}")
print(f"Linear Model RSS:    {np.sum(residuals_linear**2):.2f}")
print(f"Quadratic Model RSS: {np.sum(residuals_quad**2):.2f}")
```

**Key Insight**: The LOWESS (Locally Weighted Scatterplot Smoothing) smooth curve through the residuals makes the violation pattern obvious. In the linear model, the curve dips below and above zero, indicating systematic underprediction and overprediction. The quadratic model's residuals scatter randomly, indicating the form of non-linearity has been captured.

### Fix: Polynomial Regression

Adding polynomial features to match the true data-generating process resolves the pattern in residuals:

```python
def perform_regression(x, y, d=1):
    x_poly = np.concatenate([x**k for k in range(1, d+1)], axis=1)
    model = LinearRegression()
    model.fit(x_poly, y)
    y_pred = model.predict(x_poly)
    return model, y_pred

x, y = generate_data(d=2)
model, y_pred = perform_regression(x, y, d=2)
plot_regression_and_residuals(x, y, y_pred)
```

!!! tip "Reference"
    [Transforming nonlinear data (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/assessing-fit-least-squares-regression/v/transforming-nonlinear-data)

## Scale-Location Plot

The Scale-Location plot checks for **homoscedasticity** by plotting the square root of absolute standardized residuals against fitted values. Consistent spread across the plot supports constant variance.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.datasets import make_regression

np.random.seed(0)
X, y = make_regression(n_samples=100, n_features=1, noise=10)
data = pd.DataFrame({'X': X.flatten(), 'y': y})

X_with_const = sm.add_constant(data['X'])
model = sm.OLS(data['y'], X_with_const).fit()

data['Fitted'] = model.fittedvalues
data['Residuals'] = model.resid
data['Standardized Residuals'] = data['Residuals'] / np.std(data['Residuals'])
data['Sqrt Abs Standardized Residuals'] = np.sqrt(np.abs(data['Standardized Residuals']))

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5))

# Regression Plot
ax0.scatter(data['X'], data['y'], alpha=0.7, label='Data Points')
ax0.plot(data['X'], data['Fitted'], color='orange', label='Regression Line')
ax0.set_title('Regression Plot')
ax0.set_xlabel('Predictor (X)')
ax0.set_ylabel('Response (y)')
ax0.legend()

# Residuals vs. Fitted Values
ax1.scatter(data['Fitted'], data['Residuals'], alpha=0.7)
ax1.axhline(y=0, color='r', linestyle='--')
ax1.set_title('Residuals vs. Fitted Values Plot')
ax1.set_xlabel('Fitted Values')
ax1.set_ylabel('Residuals')

# Scale-Location Plot
ax2.scatter(data['Fitted'], data['Sqrt Abs Standardized Residuals'], alpha=0.7)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_title('Scale-Location Plot')
ax2.set_xlabel('Fitted Values')
ax2.set_ylabel(r'$\sqrt{|\text{Standardized Residuals}|}$')

plt.tight_layout()
plt.show()
```

### Why Use Square Root?

Taking the square root of absolute standardized residuals is conventional in Scale-Location plots for several reasons:

- **Stabilizes Variability**: The square root transformation compresses larger values, making trends in the spread easier to identify.
- **Visual Clarity**: Reduces the impact of outliers, producing a more visually balanced plot.
- **Statistical Tradition**: Consistent with standard diagnostic outputs from statistical software.

Using just absolute values (without the square root) is also valid, especially in introductory settings where simplicity is desired. It directly shows deviations from the fitted line without additional transformation.

## Addressing Assumption Violations

If residual analysis reveals violations:

- **Transformations**: Apply log or square-root transformations to the dependent or independent variables to address non-linearity and heteroscedasticity.
- **Weighted Least Squares (WLS)**: Assign different weights to observations based on variance, handling non-constant variance directly.
- **Robust Regression**: Minimize the influence of outliers for greater resilience to assumption deviations.
- **Polynomial Features**: Add polynomial terms when residual plots suggest non-linearity.
