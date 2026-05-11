# Performance Metrics


## R-squared (R-squared)
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

### Decomposition of SS_Total
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

## Adjusted R-squared
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

### Key Differences from R-squared
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
## Exercises

**Exercise 1.**
A regression model has MSE = 16.0 and MAE = 3.2 on a test set. A second model has MSE = 14.5 and MAE = 3.5. Which model is better, and what does the disagreement between MSE and MAE suggest about the data?

??? success "Solution to Exercise 1"
    The choice depends on the application. Model 2 has lower MSE (14.5 vs. 16.0), while Model 1 has lower MAE (3.2 vs. 3.5).

    The disagreement suggests that Model 2 has a few **large errors** that inflate its MAE more than its MSE (since MSE squares errors, it penalizes large errors more). Model 1 may have more uniform errors (lower MAE) but a few moderate outliers, while Model 2 has better average squared error but more consistently elevated absolute errors.

    If large errors are costly, prefer Model 2 (lower MSE). If typical error magnitude matters more, prefer Model 1 (lower MAE).

---

**Exercise 2.**
A model achieves $R^2 = 0.95$ on the training data but $R^2 = 0.60$ on the test data. Diagnose the problem and suggest a remedy.

??? success "Solution to Exercise 2"
    The large gap between training $R^2$ (0.95) and test $R^2$ (0.60) indicates **overfitting**. The model has learned patterns specific to the training data (including noise) that do not generalize to new data.

    Remedies: (1) **Reduce model complexity** by removing predictors, using regularization (ridge/lasso), or reducing polynomial degree. (2) **Increase training data** if possible. (3) **Use cross-validation** during model selection to estimate out-of-sample performance more reliably.

---

**Exercise 3.**
Explain why $R^2$ can be negative on a test set, even though it is always between 0 and 1 on the training set (when an intercept is included).

??? success "Solution to Exercise 3"
    On the training set with an intercept, OLS guarantees that $\text{SSR} \leq \text{SST}$, so $R^2 = 1 - \text{SSE}/\text{SST} \geq 0$.

    On the test set, $R^2 = 1 - \sum(y_i - \hat{y}_i)^2 / \sum(y_i - \bar{y}_{\text{test}})^2$. The predictions $\hat{y}_i$ are generated by the training model and may be systematically biased for the test data. If the model's predictions are worse than simply predicting the test set mean for every observation, then $\text{SSE} > \text{SST}$ and $R^2 < 0$. This means the model is not just bad -- it is worse than no model at all.
