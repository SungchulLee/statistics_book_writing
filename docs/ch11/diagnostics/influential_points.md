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
## Exercises

**Exercise 1.**
In a one-way ANOVA with three groups ($n = 10$ each), one observation has Cook's distance $D_i = 1.2$. The commonly used threshold is $D_i > 4/N$. Determine whether this point is influential and explain what Cook's distance measures.

??? success "Solution to Exercise 1"
    The threshold is $4/N = 4/30 \approx 0.133$. Since $D_i = 1.2 \gg 0.133$, this observation is highly influential.

    Cook's distance measures the overall influence of observation $i$ on all fitted values simultaneously. It combines leverage (how unusual the observation's predictor values are) and residual size (how far the observation is from the fitted model). A large Cook's distance means that removing the observation would substantially change the estimated group means and the F-statistic.

---

**Exercise 2.**
Distinguish between an outlier, a leverage point, and an influential point in the ANOVA context. Give an example where a point has high leverage but is not influential.

??? success "Solution to Exercise 2"

    - **Outlier:** An observation with an unusually large residual (far from its group mean).
    - **Leverage point:** An observation with unusual predictor values. In ANOVA this typically means belonging to a group with very few observations, giving it more influence on the group mean.
    - **Influential point:** An observation that substantially changes the results when removed. It typically has both high leverage and a large residual.

    **Example of high leverage without influence:** In a one-way ANOVA, if one group has only $n = 3$ observations while others have $n = 30$, each observation in the small group has high leverage (large hat value). But if those observations are close to their group mean, their residuals are small and they are not influential (Cook's distance remains low).

---

**Exercise 3.**
A DFFITS value exceeds $2\sqrt{p/n}$ for an observation in group B. Explain what DFFITS measures and how it differs from Cook's distance.

??? success "Solution to Exercise 3"
    DFFITS measures the change in the **fitted value** for observation $i$ when that observation is deleted, scaled by its standard error. Formally:

    $$
    \text{DFFITS}_i = \frac{\hat{Y}_i - \hat{Y}_{i(i)}}{s_{(i)} \sqrt{h_{ii}}}
    $$

    where $\hat{Y}_{i(i)}$ is the fitted value when observation $i$ is excluded.

    The key difference from Cook's distance is that DFFITS focuses on the effect on a **single fitted value** (the observation's own prediction), while Cook's distance measures the effect on **all fitted values simultaneously**. An observation can have a large DFFITS but moderate Cook's distance if its influence is localized.
