# Multiple Regression Diagnostics

## Overview

This page provides a comprehensive walkthrough of multiple linear regression diagnostics. Using the California Housing dataset, we fit a model, check for multicollinearity via Variance Inflation Factors (VIF), perform residual analysis, identify influential observations with Cook's distance, compare models using AIC and BIC, and explore extensions including log transformations, interaction terms, and polynomial regression.

## Mathematical Background

### Variance Inflation Factor

For predictor $j$, the VIF measures how much the variance of $\hat{\beta}_j$ is inflated due to correlation with other predictors:

$$
\mathrm{VIF}_j = \frac{1}{1 - R_j^2},
$$

where $R_j^2$ is the $R^2$ from regressing $x_j$ on all other predictors. A VIF above 5--10 indicates problematic multicollinearity.

### Residual Diagnostics

The standardized residual for observation $i$ is

$$
r_i = \frac{e_i}{s\sqrt{1 - h_{ii}}},
$$

where $h_{ii}$ is the $i$-th diagonal element of the hat matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$.

### Cook's Distance

Cook's distance measures the influence of the $i$-th observation on all fitted values:

$$
D_i = \frac{r_i^2}{k} \cdot \frac{h_{ii}}{1 - h_{ii}},
$$

where $k$ is the number of parameters. Observations with $D_i > 4/n$ are considered influential.

### Information Criteria

$$
\mathrm{AIC} = n \ln(\mathrm{RSS}/n) + 2k, \qquad \mathrm{BIC} = n \ln(\mathrm{RSS}/n) + k \ln(n).
$$

BIC penalizes model complexity more heavily than AIC for $n > e^2 \approx 7.4$.

## Code

### Fitting and VIF Computation

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['PRICE'] = housing.target

features = ['MedInc', 'AveRooms', 'AveOccup']
X = add_constant(df[features])
y = df['PRICE']

model = OLS(y, X).fit()

# Compute VIF for each feature
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i)
                    for i in range(X.shape[1])]
```

### Residual Analysis

```python
y_pred = model.predict(X)
residuals = y - y_pred

# Residuals vs Fitted
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)
axes[0].axhline(y=0, color='red', linestyle='--')
axes[0].set_xlabel('Fitted Values')
axes[0].set_ylabel('Residuals')

# Q-Q Plot
sm.qqplot(residuals, line='45', ax=axes[1])
plt.tight_layout()
plt.show()
```

### Model Selection with AIC and BIC

```python
feature_sets = {
    'Model 1': ['MedInc'],
    'Model 2': ['MedInc', 'AveRooms'],
    'Model 3': ['MedInc', 'AveRooms', 'AveOccup'],
    'Model 4': list(housing.feature_names),
}

for name, feats in feature_sets.items():
    X_temp = add_constant(df[feats])
    m = OLS(y, X_temp).fit()
    print(f"{name}: AIC={m.aic:.1f}, BIC={m.bic:.1f}, "
          f"R2={m.rsquared:.4f}")
```

## Interpretation

- **VIF**: Values close to 1 indicate minimal collinearity. As VIF increases, the standard errors of the affected coefficients inflate, making significance tests unreliable.
- **Residuals vs Fitted**: A random scatter indicates the linearity and constant variance assumptions are met. Patterns (curves, funnels) suggest model misspecification or heteroscedasticity.
- **Q-Q Plot**: Points along the diagonal indicate normally distributed residuals. Deviations in the tails suggest heavy-tailed or skewed residual distributions.
- **Cook's Distance**: Observations that are both outliers (large residuals) and high-leverage points (unusual predictor values) have large Cook's distances and may distort the regression fit.
- **AIC/BIC**: Lower values indicate better models. AIC tends to select larger models; BIC favors parsimony.

## Exercises

**Exercise 1.** Compute the VIF for each predictor in the full California Housing model (all 8 features). Identify which predictors exhibit high multicollinearity.

??? success "Solution to Exercise 1"

    ```python
    X_full = add_constant(df[list(housing.feature_names)])
    for i, col in enumerate(X_full.columns):
        vif = variance_inflation_factor(X_full.values, i)
        print(f"{col}: VIF = {vif:.2f}")
    ```

    Predictors with VIF > 5 exhibit substantial multicollinearity. Typically `AveRooms` and `AveBedrms` are highly correlated (both measure room counts), leading to elevated VIFs. One remedy is to drop one of the correlated predictors. $\square$

---

**Exercise 2.** Fit a model with the log-transformed response $\ln(\text{PRICE})$. Compare the Q-Q plot of residuals to the untransformed model and discuss which better satisfies the normality assumption.

??? success "Solution to Exercise 2"

    ```python
    y_log = np.log(df['PRICE'])
    model_log = OLS(y_log, X).fit()
    sm.qqplot(model_log.resid, line='45')
    ```

    The log-transformed model typically produces residuals that are closer to normal (points hug the diagonal more tightly in the Q-Q plot), because housing prices are right-skewed. The log transformation compresses large values and stabilizes variance. $\square$

---

**Exercise 3.** Using Cook's distance with threshold $4/n$, remove all influential observations from the 3-predictor model. Report the change in $R^2$ and the coefficient estimates.

??? success "Solution to Exercise 3"

    ```python
    influence = model.get_influence()
    cooks_d = influence.cooks_distance[0]
    mask = cooks_d < 4 / len(y)
    X_clean = X[mask]
    y_clean = y[mask]
    model_clean = OLS(y_clean, X_clean).fit()
    print(f"Original R2: {model.rsquared:.4f}")
    print(f"Cleaned R2:  {model_clean.rsquared:.4f}")
    ```

    Removing influential points typically improves $R^2$ and may substantially change coefficients, particularly if the influential points are pulling the regression line. $\square$

---

**Exercise 4.** Explain why the AIC formula $n\ln(\mathrm{RSS}/n) + 2k$ is equivalent (up to a constant) to $-2\ln L + 2k$ for a Gaussian likelihood.

??? success "Solution to Exercise 4"

    Under the Gaussian model, the maximized log-likelihood (ignoring constants) is

    $$
    \ln L = -\frac{n}{2}\ln(2\pi\hat{\sigma}^2) - \frac{n}{2},
    $$

    where $\hat{\sigma}^2 = \mathrm{RSS}/n$. Therefore

    $$
    -2\ln L = n\ln(2\pi) + n\ln(\mathrm{RSS}/n) + n.
    $$

    Adding $2k$ gives $\mathrm{AIC} = n\ln(\mathrm{RSS}/n) + 2k + \text{const}$. Since the constant $n\ln(2\pi) + n$ does not depend on the model, it can be dropped for model comparison. $\square$

---

**Exercise 5.** Add the interaction term $\text{MedInc} \times \text{AveRooms}$ and a quadratic term $\text{MedInc}^2$ to the model. Use AIC to determine whether these additions improve the model.

??? success "Solution to Exercise 5"

    ```python
    df['MedInc_x_AveRooms'] = df['MedInc'] * df['AveRooms']
    df['MedInc_sq'] = df['MedInc'] ** 2
    X_ext = add_constant(df[features + ['MedInc_x_AveRooms', 'MedInc_sq']])
    model_ext = OLS(y, X_ext).fit()
    print(f"Base AIC: {model.aic:.1f}")
    print(f"Extended AIC: {model_ext.aic:.1f}")
    ```

    If the extended model has a lower AIC, the additional terms improve predictive performance enough to justify the added complexity. The interaction captures whether the effect of income depends on room count, while the quadratic term captures diminishing returns of income on price. $\square$
