# Influential Data Points

## Overview

In ANOVA, certain data points can exert a disproportionate influence on the results, leading to skewed conclusions. These influential data points may be outliers (unusual response values) or leverage points (unusual predictor values), and they can significantly affect the estimated group means, variances, and the overall F-statistic. Identifying and addressing these points is crucial to ensure the robustness of the ANOVA results.

## Cook's Distance

Cook's distance combines the residual of each observation with its leverage to assess its overall influence on the fitted model. It measures how much the fitted values change when observation $i$ is removed:

$$
D_i = \frac{r_i^2}{p} \cdot \frac{h_{ii}}{1 - h_{ii}}
$$

where $r_i$ is the standardized residual, $h_{ii}$ is the leverage, and $p$ is the number of parameters in the model (number of groups in one-way ANOVA).

```python
import numpy as np
import matplotlib.pyplot as plt

influence = model.get_influence()
cooks_d = influence.cooks_distance[0]

plt.stem(range(len(cooks_d)), cooks_d, markerfmt=",")
plt.xlabel("Observation Index")
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance")
plt.axhline(y=4/len(cooks_d), color='r', linestyle='--', label=f'Threshold = {4/len(cooks_d):.3f}')
plt.legend()
plt.show()
```

Common thresholds for identifying influential points:

- $D_i > 4/n$: A commonly used rule of thumb.
- $D_i > 1$: A more conservative threshold.
- $D_i > F_{0.50}(p, n-p)$: Based on the median of the $F$-distribution.

## Leverage

Leverage measures how far an observation's predictor values are from the mean of the predictor values. In one-way ANOVA, leverage depends on the group size:

$$
h_{ii} = \frac{1}{n_i}
$$

where $n_i$ is the size of the group to which observation $i$ belongs. Points in smaller groups have higher leverage.

High leverage points are not necessarily influential—they become influential only when coupled with a large residual.

```python
leverage = influence.hat_matrix_diag

plt.scatter(leverage, influence.resid_studentized_internal, alpha=0.6)
plt.xlabel("Leverage")
plt.ylabel("Studentized Residuals")
plt.title("Leverage vs. Studentized Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

## DFFITS

DFFITS measures the influence of each observation on its own fitted value:

$$
\text{DFFITS}_i = r_i^* \sqrt{\frac{h_{ii}}{1 - h_{ii}}}
$$

where $r_i^*$ is the externally studentized residual. A common threshold is $|\text{DFFITS}_i| > 2\sqrt{p/n}$.

## Addressing Influential Points

When influential data points are identified, several strategies can be considered:

**Investigation:**
Before taking any action, investigate why the point is influential. Is it a data entry error? A measurement anomaly? Or a genuinely unusual observation that is scientifically meaningful?

**Sensitivity Analysis:**
Run the ANOVA with and without the influential points. If the conclusions change substantially, the results are not robust to these observations, and this should be reported.

**Removal:**
Remove the influential point only if there is a substantive justification (e.g., a known data error). Never remove points simply because they are inconvenient.

**Transformation:**
Applying transformations to the data (e.g., log, square root) can reduce the influence of extreme values by compressing the scale.

**Robust ANOVA Methods:**
Methods such as trimmed means, Winsorized means, or M-estimators can downweight the influence of outliers and provide more reliable results. Bootstrap methods also offer robustness against influential points.

## Example: Complete Influence Diagnostics

```python
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Fit model
model = ols('response ~ group', data=data).fit()

# Influence diagnostics
influence = model.get_influence()
summary = influence.summary_frame()
print(summary[['hat_diag', 'cooks_d', 'dffits', 'student_resid']].describe())
```
