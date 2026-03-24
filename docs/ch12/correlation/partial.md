# Partial Correlation

Two variables may appear correlated simply because they share a common influence from a third variable. For example, ice cream sales and drowning incidents are positively correlated, but both are driven by temperature. **Partial correlation** measures the linear association between two variables after removing the linear effect of one or more controlling variables, revealing whether the relationship persists once the shared influence is accounted for.

---

## Motivation

Suppose we observe a strong correlation between variables $X$ and $Y$. Before concluding that $X$ and $Y$ are directly related, we should ask: is this association driven by a third variable $Z$? Partial correlation answers this question by "partialling out" the effect of $Z$.

If the partial correlation $r_{XY \cdot Z}$ is close to zero while the marginal correlation $r_{XY}$ is large, the apparent association between $X$ and $Y$ is largely explained by $Z$. Conversely, if $r_{XY \cdot Z}$ remains large, the relationship between $X$ and $Y$ persists after accounting for $Z$.

---

## Definition: First-Order Partial Correlation

The **partial correlation** between $X$ and $Y$ controlling for a single variable $Z$ is

$$
r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ} \, r_{YZ}}{\sqrt{1 - r_{XZ}^2} \; \sqrt{1 - r_{YZ}^2}}
$$

where $r_{XY}$, $r_{XZ}$, and $r_{YZ}$ are the pairwise Pearson correlations.

This formula is called the **first-order** partial correlation because we control for a single variable. The result satisfies $-1 \le r_{XY \cdot Z} \le 1$.

---

## Geometric Interpretation

Partial correlation has an equivalent interpretation through linear regression residuals:

1. Regress $X$ on $Z$ and compute the residuals $e_X = X - \hat{X}$.
2. Regress $Y$ on $Z$ and compute the residuals $e_Y = Y - \hat{Y}$.
3. The partial correlation $r_{XY \cdot Z}$ equals the Pearson correlation between $e_X$ and $e_Y$.

The residuals $e_X$ and $e_Y$ represent the parts of $X$ and $Y$ that are not linearly explained by $Z$. The partial correlation therefore measures the linear association between the "unexplained" components of $X$ and $Y$.

---

## Higher-Order Partial Correlation

When controlling for multiple variables $Z_1, Z_2, \ldots, Z_k$, the partial correlation $r_{XY \cdot Z_1 Z_2 \cdots Z_k}$ can be computed recursively:

$$
r_{XY \cdot Z_1 Z_2 \cdots Z_k} = \frac{r_{XY \cdot Z_1 \cdots Z_{k-1}} - r_{XZ_k \cdot Z_1 \cdots Z_{k-1}} \, r_{YZ_k \cdot Z_1 \cdots Z_{k-1}}}{\sqrt{1 - r_{XZ_k \cdot Z_1 \cdots Z_{k-1}}^2} \; \sqrt{1 - r_{YZ_k \cdot Z_1 \cdots Z_{k-1}}^2}}
$$

Equivalently, one can regress $X$ and $Y$ on all the control variables $Z_1, \ldots, Z_k$, then correlate the residuals. In practice, the residual approach is simpler for higher-order partial correlations.

---

## Example: Ice Cream, Drowning, and Temperature

Consider three variables measured monthly:

- $X$: ice cream sales (thousands of units)
- $Y$: drowning incidents
- $Z$: average temperature (degrees)

Suppose the pairwise correlations are:

$$
r_{XY} = 0.85, \quad r_{XZ} = 0.92, \quad r_{YZ} = 0.88
$$

The partial correlation between ice cream sales and drowning, controlling for temperature, is

$$
r_{XY \cdot Z} = \frac{0.85 - (0.92)(0.88)}{\sqrt{1 - 0.92^2}\;\sqrt{1 - 0.88^2}} = \frac{0.85 - 0.8096}{\sqrt{0.1536}\;\sqrt{0.2256}} = \frac{0.0404}{0.3920 \times 0.4750} \approx 0.22
$$

The marginal correlation of $0.85$ drops to a partial correlation of approximately $0.22$. Most of the apparent association between ice cream sales and drowning is explained by their shared dependence on temperature.

---

## Partial vs Semi-Partial Correlation

It is important to distinguish partial correlation from **semi-partial** (or **part**) correlation:

| | Partial $r_{XY \cdot Z}$ | Semi-partial $r_{X(Y \cdot Z)}$ |
|:---|:---|:---|
| What is controlled | Effect of $Z$ removed from **both** $X$ and $Y$ | Effect of $Z$ removed from $Y$ only |
| Interpretation | Association after removing $Z$ from both | Unique contribution of $Y$ beyond $Z$ |
| Common use | General association analysis | $R^2$ decomposition in regression |

The semi-partial correlation is defined as

$$
r_{X(Y \cdot Z)} = \frac{r_{XY} - r_{XZ} \, r_{YZ}}{\sqrt{1 - r_{YZ}^2}}
$$

Note that the numerator is the same, but the denominator adjusts only for $Z$'s effect on $Y$.

---

## Computation in Python

```python
import numpy as np
from scipy import stats

# Simulated data
np.random.seed(42)
n = 100
z = np.random.normal(0, 1, n)       # temperature (confounder)
x = 2 * z + np.random.normal(0, 1, n)  # ice cream sales
y = 1.5 * z + np.random.normal(0, 1, n)  # drowning incidents

# Pairwise correlations
r_xy = stats.pearsonr(x, y)[0]
r_xz = stats.pearsonr(x, z)[0]
r_yz = stats.pearsonr(y, z)[0]

# Partial correlation using the formula
r_xy_z = (r_xy - r_xz * r_yz) / (
    np.sqrt(1 - r_xz**2) * np.sqrt(1 - r_yz**2)
)

print(f"Marginal r(X,Y)     = {r_xy:.4f}")
print(f"Partial r(X,Y | Z)  = {r_xy_z:.4f}")
```

For a dedicated function, the `pingouin` library provides `pingouin.partial_corr`, which also computes confidence intervals and p-values.

---

## Connection to Multiple Regression

Partial correlations are closely related to the coefficients and tests in multiple regression. In a regression of $Y$ on both $X$ and $Z$, the t-test for the coefficient of $X$ is equivalent to testing whether $r_{XY \cdot Z} = 0$. This connection makes partial correlation a foundational concept for understanding regression output. See [Multiple Regression](../../ch13/linear_regression/multiple.md) for details.

---

## Summary

Partial correlation measures the linear association between two variables after controlling for one or more additional variables. It is computed either through a closed-form formula involving pairwise correlations or by correlating regression residuals. A large marginal correlation that vanishes (or shrinks substantially) after controlling for a third variable signals that the apparent association is driven by a common influence rather than a direct relationship. Partial correlation is essential for disentangling confounded relationships and forms the statistical foundation for many regression diagnostics.
