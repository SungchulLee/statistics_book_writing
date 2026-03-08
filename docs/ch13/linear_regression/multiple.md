# Multiple Linear Regression

## The Model

Multiple linear regression extends simple linear regression to include multiple predictor variables:

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \varepsilon_i
$$

In matrix notation, this can be written compactly as:

$$
\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon}
$$

where $X$ is the $n \times (p+1)$ design matrix (with a column of ones for the intercept), $\boldsymbol{\beta}$ is the $(p+1) \times 1$ coefficient vector, and $\boldsymbol{\varepsilon}$ is the $n \times 1$ error vector.

## Interaction Terms

In many applications, the effect of one predictor on the response depends on the level of another predictor. An **interaction term** captures this combined effect. For example, with predictors TV and Radio:

$$
\hat{y} = \beta_0 + \beta_1 \cdot \text{TV} + \beta_2 \cdot \text{Radio} + \beta_3 \cdot (\text{TV} \times \text{Radio})
$$

A positive and significant interaction coefficient $\beta_3$ indicates a **synergistic effect**: the combined impact of both predictors is greater than the sum of their individual effects.

## Implementation with scikit-learn

### Random Train-Test Split

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the Advertising dataset
url = 'https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv'
df = pd.read_csv(url, usecols=[1, 2, 3, 4])
print(df.head(), end="\n\n")

# Add an interaction term between TV and Radio
df['TV:Radio'] = df['TV'] * df['Radio']
print(df.head(), end="\n\n")

# Define test size ratio
test_size_ratio = 0.3

# Split data into features (X) and target (y)
X = df[['TV', 'Radio', 'TV:Radio']]
y = df['Sales']

# Perform the train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42)
print("x_train.head()")
print(x_train.head(), end="\n\n")
print("y_train.head()")
print(y_train.head(), end="\n\n")

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Make predictions on both the training and test data
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Print model coefficients and intercept
print(f"Model Intercept: {model.intercept_:.4f}")
print(f"Model Coefficients: {np.round(model.coef_, 4)}\n")

# Visualize predictions
fig, axes = plt.subplots(1, 2, figsize=(12, 3))

for ax, title, y_actual, y_pred in zip(axes, ("Train Set", "Test Set"), (y_train, y_test), (y_train_pred, y_test_pred)):
    ax.set_title(f"{title}: Actual vs Predicted Sales")
    ax.plot(y_actual, y_pred, '.', label="Predicted Sales")
    ax.plot(y_actual, y_actual, '-r', alpha=0.5, label="Actual Sales (Target)")
    ax.set_xlabel('Actual Sales')
    ax.set_ylabel('Predicted Sales')
    ax.legend()

plt.tight_layout()
plt.show()
```

### Deterministic Train-Test Split

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
url = 'https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv'
df = pd.read_csv(url, usecols=[1, 2, 3, 4])

# Add interaction term
df['TV:Radio'] = df['TV'] * df['Radio']

# Deterministic split
num_total_observations = df.shape[0]
test_ratio = 0.3
num_train_observations = int(num_total_observations * (1 - test_ratio))

train_data = df.iloc[:num_train_observations]
test_data = df.iloc[num_train_observations:]

x_train = train_data[['TV', 'Radio', 'TV:Radio']]
y_train = train_data['Sales']
x_test = test_data[['TV', 'Radio', 'TV:Radio']]
y_test = test_data['Sales']

# Fit model
model = LinearRegression()
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

print(f"Model Intercept: {model.intercept_:.4f}")
print(f"Model Coefficients: {np.round(model.coef_, 4)}\n")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 3))

for ax, title, y_actual, y_pred in zip(axes, ("Train Set", "Test Set"), (y_train, y_test), (y_train_pred, y_test_pred)):
    ax.set_title(f"{title}: Actual vs Predicted Sales")
    ax.plot(y_actual, y_pred, '.', label="Predicted Sales")
    ax.plot(y_actual, y_actual, '-r', alpha=0.5, label="Actual Sales (Target)")
    ax.set_xlabel('Actual Sales')
    ax.set_ylabel('Predicted Sales')
    ax.legend()

plt.tight_layout()
plt.show()
```

## Implementation with statsmodels

The `statsmodels` library provides extensive statistical output including p-values, confidence intervals, and diagnostic tests. See [Package Usage Comparison](../package_usage/comparison.md) for a detailed comparison with scikit-learn.

### Sales ~ TV + Radio + Newspaper

```python
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv'
data = pd.read_csv(url, usecols=[1, 2, 3, 4])

num_total_observations = data.shape[0]
test_ratio = 0.3
num_train_observations = int(num_total_observations * (1 - test_ratio))

train_data = data.iloc[:num_train_observations]
test_data = data.iloc[num_train_observations:]

# Fit model with TV, Radio, and Newspaper
model = sm.ols('Sales ~ TV + Radio + Newspaper', train_data).fit()
print("Model with TV, Radio, and Newspaper as predictors:")
print(model.summary(), end="\n\n")

train_predictions = model.predict(train_data)
test_predictions = model.predict(test_data)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].set_title("Training Data: Actual vs Predicted Sales")
axes[0].plot(train_data['Sales'], train_predictions, '.', label="Predicted Sales")
axes[0].plot(train_data['Sales'], train_data['Sales'], '-r', alpha=0.5, label="Actual Sales (Target)")
axes[0].set_xlabel('Actual Sales')
axes[0].set_ylabel('Predicted Sales')
axes[0].legend()

axes[1].set_title("Test Data: Actual vs Predicted Sales")
axes[1].plot(test_data['Sales'], test_predictions, '.', label="Predicted Sales")
axes[1].plot(test_data['Sales'], test_data['Sales'], '-r', alpha=0.5, label="Actual Sales (Target)")
axes[1].set_xlabel('Actual Sales')
axes[1].set_ylabel('Predicted Sales')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Sales ~ TV + Radio

```python
model = sm.ols('Sales ~ TV + Radio', train_data).fit()
print("Model with TV and Radio as predictors:")
print(model.summary(), end="\n\n")
```

### Sales ~ TV + Radio + TV:Radio

```python
model = sm.ols('Sales ~ TV + Radio + TV:Radio', train_data).fit()
print("Model with TV, Radio, and TV:Radio as predictors:")
print(model.summary(), end="\n\n")
```

## Interpreting statsmodels Output

The `model.summary()` output contains several important sections:

**Model Summary Section**

- **R-squared**: Proportion of variance in the dependent variable explained by the model.
- **Adj. R-squared**: R-squared adjusted for the number of predictors; penalizes unnecessary complexity.
- **F-statistic and Prob (F-statistic)**: Tests whether all coefficients are jointly zero. A high F-statistic with low p-value indicates the model is significant overall.
- **AIC and BIC**: Information criteria for model comparison; lower values indicate better models.

**Coefficients Table**

- **coef**: Estimated coefficient values.
- **std err**: Standard error of the estimate.
- **t**: t-statistic for testing whether the coefficient differs from zero.
- **P>|t|**: p-value for the coefficient; values below 0.05 indicate statistical significance.
- **[0.025, 0.975]**: 95% confidence interval for the coefficient.

**Diagnostic Metrics**

- **Omnibus and Jarque-Bera**: Tests for normality of residuals.
- **Durbin-Watson**: Tests for autocorrelation in residuals (values near 2 suggest no autocorrelation).
- **Skew and Kurtosis**: Describe the shape of the residual distribution.
- **Condition Number**: Measures multicollinearity; values above 30 suggest potential issues.

## Model Comparison Example

Comparing three models on the Advertising dataset:

| Metric | TV, Radio, Newspaper | TV, Radio | TV, Radio, TV:Radio |
|---|---|---|---|
| **R-squared** | 0.894 | 0.894 | **0.965** |
| **Adj. R-squared** | 0.891 | 0.892 | **0.964** |
| **F-statistic** | 381.2 | 575.1 | **1256** |
| **AIC** | 555.8 | 554.0 | **399.6** |
| **BIC** | 567.5 | 562.8 | **411.4** |
| **Significant Predictors** | TV, Radio | TV, Radio | TV, Radio, TV:Radio |
| **Condition Number** | 457 | 424 | **1.84e+04** |

Key findings from this comparison:

- Removing Newspaper does not reduce $R^2$, and AIC/BIC improve slightly, confirming Newspaper is not a useful predictor.
- Adding the interaction term TV:Radio significantly improves the model ($R^2$ from 0.894 to 0.965), with substantially lower AIC and BIC.
- The interaction model has a very high condition number (1.84e+04), indicating strong multicollinearity that may affect coefficient stability. Variable centering or regularization can help address this.
- All models show residual normality violations, but the interaction model shows the most severe deviations.

The **TV, Radio, and Interaction Model** is preferred for predictive power, while the **TV and Radio Model** may be preferable when interpretability and coefficient stability are priorities.
