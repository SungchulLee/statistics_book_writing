# Regression Diagnostics (Housing)

## Overview

This page demonstrates a comprehensive regression diagnostics workflow using the King County housing dataset. We cover studentized residuals for outlier detection, Cook's distance and hat values for influence analysis, heteroscedasticity testing via the Breusch-Pagan test, partial residual plots, and the impact of removing influential observations on model estimates.

## Mathematical Background

### Studentized Residuals

The internally studentized residual for observation $i$ is

$$
r_i = \frac{e_i}{s\sqrt{1 - h_{ii}}},
$$

where $e_i = y_i - \hat{y}_i$ is the raw residual and $h_{ii}$ is the leverage (diagonal of the hat matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$). Observations with $|r_i| > 2.5$ are potential outliers.

### Hat Values (Leverage)

The leverage $h_{ii}$ measures how far observation $i$ is from the center of the predictor space. High leverage means $\mathbf{x}_i$ is unusual. Observations with $h_{ii} > 2k/n$ or $h_{ii} > 3\bar{h}$ (where $\bar{h} = k/n$) are high-leverage points.

### Cook's Distance

Cook's distance combines residual magnitude and leverage:

$$
D_i = \frac{1}{k}\,r_i^2\,\frac{h_{ii}}{1 - h_{ii}}.
$$

Equivalently, $D_i$ measures the change in all fitted values when observation $i$ is deleted. A common threshold is $D_i > 4/n$.

### Breusch-Pagan Test

The Breusch-Pagan test for heteroscedasticity regresses squared residuals on the predictors:

$$
e_i^2 = \gamma_0 + \gamma_1 x_{i1} + \cdots + \gamma_p x_{ip} + v_i.
$$

Under $H_0$ (homoscedasticity), the test statistic $nR^2$ from this auxiliary regression follows $\chi^2_p$.

## Code

### Baseline Model and Influence Diagnostics

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
X = house_98105[predictors].assign(const=1)
y = house_98105['AdjSalePrice']

results = sm.OLS(y, X).fit()
influence = OLSInfluence(results)

studentized_resids = influence.resid_studentized_internal
hat_values = influence.hat_matrix_diag
cooks_dist, _ = influence.cooks_distance
```

### Effect of Removing Influential Points

```python
threshold_cooks = 4 / len(y)
mask_keep = cooks_dist < threshold_cooks

X_filtered = X[mask_keep]
y_filtered = y[mask_keep]
results_filtered = sm.OLS(y_filtered, X_filtered).fit()

# Compare coefficients
comparison = pd.DataFrame({
    'Original': results.params,
    'Filtered': results_filtered.params,
})
```

### Heteroscedasticity Test

```python
from statsmodels.stats.diagnostic import het_breuschpagan

bp_stat, bp_pval, _, _ = het_breuschpagan(results.resid, X)
print(f"Breusch-Pagan p-value: {bp_pval:.4f}")
```

## Interpretation

- **Studentized residuals**: Large values indicate observations that are poorly predicted by the model. These may be data entry errors, unusual cases, or signals that the model is misspecified.
- **Leverage**: High-leverage points are not necessarily harmful; they become influential only when they also have large residuals. A high-leverage point that lies on the regression surface actually stabilizes the fit.
- **Cook's distance**: Points with high Cook's distance affect the entire regression surface. Removing them and comparing results reveals whether conclusions are sensitive to a few observations.
- **Breusch-Pagan test**: A significant result ($p < 0.05$) indicates heteroscedasticity, suggesting that OLS standard errors are unreliable. Remedies include WLS, robust standard errors, or variance-stabilizing transformations.
- **Partial residual plots** (CCPR plots) show the relationship between each predictor and the response after accounting for other predictors, revealing potential nonlinearities.

## Exercises

**Exercise 1.** Compute externally studentized residuals (using the leave-one-out variance estimate) and compare them to the internally studentized residuals. Which observations show the largest discrepancy?

??? success "Solution to Exercise 1"

    ```python
    ext_resids = influence.resid_studentized_external
    int_resids = influence.resid_studentized_internal
    discrepancy = np.abs(ext_resids - int_resids)
    top_idx = np.argsort(discrepancy)[-5:]
    ```

    The largest discrepancies occur at observations where the residual is large and $h_{ii}$ is also large. The external residual uses $s_{(i)}$ (computed without observation $i$), which differs most from $s$ when observation $i$ strongly affects the residual variance. $\square$

---

**Exercise 2.** Prove that the sum of all hat values equals $k$ (the number of parameters). What does this imply about the average leverage?

??? success "Solution to Exercise 2"

    The hat matrix is $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$. Since $\mathbf{H}$ is idempotent ($\mathbf{H}^2 = \mathbf{H}$) and symmetric:

    $$
    \sum_{i=1}^n h_{ii} = \operatorname{tr}(\mathbf{H}) = \operatorname{tr}(\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top) = \operatorname{tr}((\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{X}) = \operatorname{tr}(\mathbf{I}_k) = k.
    $$

    The average leverage is $\bar{h} = k/n$, so the threshold $3\bar{h} = 3k/n$ identifies points with leverage three times the average. $\square$

---

**Exercise 3.** Implement the Breusch-Pagan test from scratch by running the auxiliary regression of $e_i^2$ on $\mathbf{X}$ and computing $nR^2$. Verify it matches the statsmodels result.

??? success "Solution to Exercise 3"

    ```python
    resid_sq = results.resid ** 2
    aux_model = sm.OLS(resid_sq, X).fit()
    bp_stat_manual = len(y) * aux_model.rsquared
    from scipy import stats
    bp_pval_manual = 1 - stats.chi2.cdf(bp_stat_manual, X.shape[1] - 1)
    ```

    The manual computation should closely match `het_breuschpagan` (minor differences may arise from implementation details in the constant term handling). $\square$

---

**Exercise 4.** After removing influential observations, the coefficient estimates change. Discuss when it is appropriate to remove such observations versus when they should be retained.

??? success "Solution to Exercise 4"

    Removal is appropriate when influential observations are clearly erroneous (data entry mistakes, measurement failures) or come from a different population. Retention is appropriate when the observations are legitimate data points that happen to be unusual. In this case, the sensitivity analysis itself is informative: if results change drastically, the conclusions are fragile. Alternatives to removal include robust regression (e.g., Huber or bisquare weighting), which downweights influential points without entirely excluding them. $\square$

---

**Exercise 5.** Explain why the Q-Q plot tails often deviate from the diagonal even when the true errors are normal. How does sample size affect this phenomenon?

??? success "Solution to Exercise 5"

    In a Q-Q plot, the extreme order statistics (the smallest and largest residuals) have the highest sampling variability. Even when errors are exactly normal, the tails of the empirical distribution fluctuate substantially in finite samples. With $n$ observations, the expected range of a normal sample is approximately $2\sqrt{2\ln n}\,\sigma$, but the actual range varies considerably. As $n$ increases, the Q-Q plot tails become more stable (the law of large numbers stabilizes the empirical quantiles), so large-sample Q-Q plots provide more reliable evidence about tail behavior. $\square$
