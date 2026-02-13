# Performance Metrics

## R-squared ($R^2$)

### Definition

$R^2$ represents the proportion of the variance in the dependent variable that is predictable from the independent variables:

$$
R^2 = 1 - \frac{SS_{\text{Residual}}}{SS_{\text{Total}}}
$$

where:

$$
\begin{array}{lll}
SS_{\text{Total}} &=& \displaystyle \sum_{i}\left(y_{i}-\bar{y}\right)^{2} \\[8pt]
SS_{\text{Residual}} &=& \displaystyle \sum_{i}\left(y_{i}-\hat{y}_{i}\right)^{2}
\end{array}
$$

$R^2$ values range from 0 to 1, with higher values indicating a better fit. However, $R^2$ always increases as more predictors are added, even if they do not improve predictive power, which can lead to overfitting.

### Decomposition of $SS_{\text{Total}}$

The total variation in $y$ decomposes cleanly into explained and unexplained components:

$$
\begin{array}{lll}
SS_{\text{Total}} &=& \displaystyle \sum_{i}\left(y_{i}-\bar{y}\right)^{2} \\[10pt]
&=& \displaystyle \sum_{i}\left(\left(y_{i}-\hat{y}_{i}\right) + \left(\hat{y}_{i}-\bar{y}\right)\right)^{2} \\[10pt]
&=& \displaystyle \sum_{i}\left(y_{i}-\hat{y}_{i}\right)^{2} + \sum_{i}\left(\hat{y}_{i}-\bar{y}\right)^{2} \\[10pt]
&=& \displaystyle SS_{\text{Residual}} + SS_{\text{Treatment}}
\end{array}
$$

where $SS_{\text{Treatment}}$ represents the variation explained by the regression model. The cross terms vanish due to the properties of OLS estimation.

### Interpretation in Simple Linear Regression

In simple linear regression, $SS_{\text{Treatment}}$ can be expressed in terms of the correlation coefficient:

$$
\begin{array}{lll}
SS_{\text{Treatment}} &=& \displaystyle \sum_{i}\left(\hat{y}_{i} - \bar{y}\right)^{2} \\[10pt]
&\approx& \displaystyle \beta^2 \sum_{i}\left(x_i - \bar{x}\right)^{2} \\[10pt]
&\approx& \displaystyle n\sigma_x^2\beta^2 \\[10pt]
&\approx& \displaystyle n\sigma_x^2\left(\rho\frac{\sigma_y}{\sigma_x}\right)^2 \\[10pt]
&=& \displaystyle n\sigma_y^2\rho^2
\end{array}
$$

Therefore:

$$
R^2 = \frac{SS_{\text{Treatment}}}{SS_{\text{Total}}} \approx \frac{n\sigma_y^2 \rho^2}{n\sigma_y^2} = \rho^2
$$

In simple linear regression, $R^2$ is approximately the square of the correlation coefficient between $x$ and $y$.

!!! tip "References"
    - [R-squared or coefficient of determination (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/assessing-fit-least-squares-regression/v/r-squared-or-coefficient-of-determination)
    - [R-squared intuition (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/assessing-fit-least-squares-regression/a/r-squared-intuition)

## Adjusted $R^2$

### Definition

Adjusted $R^2$ accounts for the number of predictors, penalizing unnecessary complexity:

$$
\text{Adjusted } R^2 = 1 - \left(1 - R^2\right) \frac{n - 1}{n - p - 1}
$$

where $n$ is the sample size and $p$ is the number of predictors (excluding the intercept).

### Derivation of the Adjustment Factor

The adjustment replaces the raw sums of squares with their unbiased estimates (divided by degrees of freedom):

$$
\text{Adjusted } R^2 = 1 - \frac{SS_{\text{Residual}} / (n - p - 1)}{SS_{\text{Total}} / (n - 1)}
$$

This ensures that adding a predictor only improves Adjusted $R^2$ if the reduction in $SS_{\text{Residual}}$ justifies the lost degree of freedom.

### Key Differences from $R^2$

- **Model Complexity**: Adjusted $R^2$ accounts for the number of predictors; $R^2$ does not.
- **Model Comparison**: Adjusted $R^2$ is better for comparing models with different numbers of predictors.
- **Direction**: Adjusted $R^2$ can decrease when adding a predictor that does not improve the model, while $R^2$ can only increase.

## Other Performance Metrics

$$
\begin{array}{lll}
\text{MAE} && \displaystyle\frac{1}{n}\sum_{i=1}^n|y_i-\hat{y}_i| \\[10pt]
\text{MSE} && \displaystyle\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2 \\[10pt]
\text{RMSE} && \displaystyle\sqrt{\frac{1}{n}\sum_{i=1}^n(y_i-\hat{y}_i)^2}
\end{array}
$$

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values. Less sensitive to outliers than MSE. Provides error in the same units as the response.
- **MSE (Mean Squared Error)**: Average squared difference. Penalizes larger errors more heavily. Used as the loss function in OLS.
- **RMSE (Root Mean Squared Error)**: Square root of MSE. Returns error to the original units of the response, making it more interpretable than MSE.

### Implementation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the dataset
url = 'https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv'
df = pd.read_csv(url, usecols=[1, 2, 3, 4])

# Add interaction term
df['TV:Radio'] = df['TV'] * df['Radio']

X = df[['TV', 'Radio', 'TV:Radio']]
y = df['Sales']

# Split the data
test_size_ratio = 0.3
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, random_state=42)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Display coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}\n")

# R-squared
print(f"Training R^2: {model.score(x_train, y_train)}")
print(f"Testing R^2: {model.score(x_test, y_test)}\n")

# MAE
print(f"Training MAE: {metrics.mean_absolute_error(y_train, y_train_pred)}")
print(f"Testing MAE: {metrics.mean_absolute_error(y_test, y_test_pred)}\n")

# MSE
print(f"Training MSE: {metrics.mean_squared_error(y_train, y_train_pred)}")
print(f"Testing MSE: {metrics.mean_squared_error(y_test, y_test_pred)}\n")

# RMSE
print(f"Training RMSE: {np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))}")
print(f"Testing RMSE: {np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))}\n")
```
