# Regression (Weed Price vs Demographics)

## Overview

This page applies linear regression to predict high-quality weed prices from state-level demographic features including population, per capita income, and racial composition. We demonstrate exploratory analysis via correlation matrices, single-variable and multi-variable OLS using both scikit-learn and statsmodels, and evaluate model performance on a held-out test set.

## Mathematical Background

The multiple linear regression model is

$$
\text{HighQ}_i = \beta_0 + \beta_1 \cdot \text{population}_i + \beta_2 \cdot \text{income}_i + \beta_3 \cdot \text{pct\_white}_i + \varepsilon_i.
$$

The OLS estimator minimizes

$$
\mathrm{RSS} = \sum_{i=1}^n \left(\text{HighQ}_i - \hat{\beta}_0 - \hat{\beta}_1 x_{i1} - \cdots - \hat{\beta}_p x_{ip}\right)^2.
$$

Model evaluation on held-out test data uses Root Mean Squared Error:

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n_{\text{test}}}\sum_{i \in \text{test}}(y_i - \hat{y}_i)^2}.
$$

### Correlation

The Pearson correlation between predictor $x_j$ and response $y$ measures linear association:

$$
r_{y,x_j} = \frac{\sum(x_{ij} - \bar{x}_j)(y_i - \bar{y})}{\sqrt{\sum(x_{ij} - \bar{x}_j)^2}\sqrt{\sum(y_i - \bar{y})^2}}.
$$

Predictors with high $|r|$ are candidates for the regression model, though correlation does not imply causation.

## Code

### Data and Train/Test Split

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

np.random.seed(42)
df = build_dataset()  # synthetic state-level data

TEST_STATES = {"iowa", "kentucky", "missouri", "nevada",
               "wyoming", "south dakota", "new jersey", "colorado_extra"}
train = df[~df['state'].isin(TEST_STATES)].copy()
test = df[df['state'].isin(TEST_STATES)].copy()
```

### Single-Variable Regression

```python
model1 = LinearRegression().fit(train[['total_population']], train['HighQ'])
pred1 = model1.predict(test[['total_population']])
rmse1 = np.sqrt(np.mean((test['HighQ'] - pred1) ** 2))
```

### Multi-Variable Regression with statsmodels

```python
formula = "HighQ ~ total_population + per_capita_income + percent_white"
sm_model = smf.ols(formula=formula, data=train).fit()
print(sm_model.summary())

pred3 = sm_model.predict(test)
rmse3 = np.sqrt(np.mean((test['HighQ'] - pred3) ** 2))
```

### Prediction Table

```python
result = pd.DataFrame({
    'state': test['state'].values,
    'actual': test['HighQ'].values,
    'predicted': np.round(pred3.values, 2),
})
result['error'] = result['actual'] - result['predicted']
```

## Interpretation

- **Correlation analysis** reveals which demographic features are linearly related to weed price. Income and percent white tend to have the strongest correlations, consistent with the data-generating process.
- **Single-variable model**: Using population alone yields poor predictions because population has weak correlation with price in this dataset.
- **Multi-variable model**: Adding income and racial composition substantially improves RMSE, as these capture more of the price variation.
- **Test RMSE** provides an honest assessment of prediction accuracy on unseen states. Comparing test RMSE across models reveals which predictors contribute meaningful information.
- **statsmodels output** provides standard errors, $t$-statistics, and $p$-values for each coefficient, enabling formal inference about which predictors are statistically significant.

## Exercises

**Exercise 1.** Compute the correlation matrix for all features and the response. Which predictor has the strongest linear relationship with HighQ?

??? success "Solution to Exercise 1"

    ```python
    features = ['total_population', 'per_capita_income',
                'percent_white', 'percent_black', 'percent_hispanic']
    corr = train[['HighQ'] + features].corr()
    print(corr['HighQ'].sort_values(ascending=False))
    ```

    The predictor with the largest absolute correlation with HighQ is the most linearly associated. Based on the data-generating process, `per_capita_income` and `percent_white` should show the strongest correlations. $\square$

---

**Exercise 2.** Add `percent_black` and `percent_hispanic` to the multi-variable model. Does the test RMSE improve? Discuss whether including more predictors always helps.

??? success "Solution to Exercise 2"

    ```python
    formula_full = ("HighQ ~ total_population + per_capita_income + "
                    "percent_white + percent_black + percent_hispanic")
    sm_full = smf.ols(formula=formula_full, data=train).fit()
    pred_full = sm_full.predict(test)
    rmse_full = np.sqrt(np.mean((test['HighQ'] - pred_full) ** 2))
    ```

    If the additional predictors are noise (zero true coefficients), including them adds variance to the estimates without reducing bias, potentially increasing test RMSE. Adding predictors always reduces training RSS but may increase test error due to overfitting, especially with small $n$. $\square$

---

**Exercise 3.** Implement a simple train/test split using 80/20 instead of the state-based split. Compare the RMSE to the original approach and discuss the pros and cons.

??? success "Solution to Exercise 3"

    ```python
    from sklearn.model_selection import train_test_split

    X_all = df[['total_population', 'per_capita_income', 'percent_white']]
    y_all = df['HighQ']
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all,
                                                test_size=0.2, random_state=42)
    model_split = LinearRegression().fit(X_tr, y_tr)
    rmse_split = np.sqrt(np.mean((y_te - model_split.predict(X_te)) ** 2))
    ```

    Random splits ensure no systematic difference between train and test, but results vary with the seed. The state-based split is more realistic if the goal is predicting prices for entirely new states. $\square$

---

**Exercise 4.** Use the statsmodels $F$-test to compare the reduced model (population only) to the full model (population + income + percent white). State the hypotheses and interpret the result.

??? success "Solution to Exercise 4"

    The $F$-test evaluates:

    $$
    H_0\colon \beta_{\text{income}} = \beta_{\text{pct\_white}} = 0 \quad \text{vs} \quad H_1\colon \text{at least one is nonzero}.
    $$

    ```python
    sm_reduced = smf.ols("HighQ ~ total_population", data=train).fit()
    f_stat = ((sm_reduced.ssr - sm_model.ssr) / 2) / sm_model.mse_resid
    from scipy import stats
    p_val = 1 - stats.f.cdf(f_stat, 2, sm_model.df_resid)
    ```

    A small $p$-value rejects $H_0$, confirming that income and percent white jointly contribute significant predictive power beyond population alone. $\square$

---

**Exercise 5.** Discuss the limitations of using observational demographic data to draw causal conclusions about what drives weed prices across states. What confounding factors might bias the regression coefficients?

??? success "Solution to Exercise 5"

    Observational regression coefficients measure associations, not causal effects. Potential confounders include: (1) state-level marijuana legislation (legalization vs. prohibition affects supply and price), (2) proximity to production regions or borders, (3) urban/rural composition (which correlates with both demographics and prices), (4) enforcement intensity, (5) cost of living. These unmeasured variables correlate with both the predictors (income, demographics) and the response (price), biasing the coefficients. A causal analysis would require methods such as instrumental variables, difference-in-differences, or randomized experiments. $\square$
