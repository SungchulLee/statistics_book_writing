# Pearson Correlation Coefficient

When analyzing two quantitative variables, a natural first question is: do they tend to move together? The **Pearson correlation coefficient** provides a precise, unitless measure of the strength and direction of the *linear* relationship between two variables. It is the most widely used correlation measure in statistics.

---

## Population Correlation

For two random variables $X$ and $Y$ with finite variances, the **population Pearson correlation coefficient** is defined as

$$
\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \, \sigma_Y} = \frac{\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]}{\sqrt{\mathbb{E}[(X - \mu_X)^2]} \; \sqrt{\mathbb{E}[(Y - \mu_Y)^2]}}
$$

where $\mu_X = \mathbb{E}[X]$, $\mu_Y = \mathbb{E}[Y]$, $\sigma_X = \sqrt{\text{Var}(X)}$, and $\sigma_Y = \sqrt{\text{Var}(Y)}$.

The key property is that $\rho_{XY}$ always lies in the interval $[-1, 1]$.

---

## Sample Correlation Coefficient

Given paired observations $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$, the **sample Pearson correlation coefficient** is

$$
r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \; \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
$$

where $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ and $\bar{y} = \frac{1}{n}\sum_{i=1}^n y_i$ are the sample means. An equivalent computational formula is

$$
r = \frac{n \sum x_i y_i - (\sum x_i)(\sum y_i)}{\sqrt{[n \sum x_i^2 - (\sum x_i)^2][n \sum y_i^2 - (\sum y_i)^2]}}
$$

The sample correlation $r$ is a consistent estimator of the population correlation $\rho_{XY}$.

---

## Properties

The Pearson correlation coefficient satisfies several important properties.

1. **Boundedness.** $-1 \le r \le 1$ for any dataset, and $-1 \le \rho \le 1$ for any pair of random variables with finite variance.

2. **Symmetry.** $r_{XY} = r_{YX}$, and similarly for $\rho$.

3. **Translation invariance.** For any constants $a$ and $b$,

    $$
    r_{X+a, \, Y+b} = r_{X, Y}
    $$

    Shifting either variable does not change the correlation.

4. **Scale invariance.** For positive constants $a > 0$ and $b > 0$,

    $$
    r_{aX, \, bY} = r_{X, Y}
    $$

    Rescaling by a positive factor preserves the correlation. If $a < 0$ or $b < 0$, the sign of $r$ flips.

5. **Perfect correlation.** $|r| = 1$ if and only if all data points lie exactly on a line. Specifically, $r = 1$ when $y_i = a + bx_i$ for some $b > 0$, and $r = -1$ when $b < 0$.

---

## Interpretation

The value of $r$ describes the strength and direction of the linear association.

| Range of $r$ | Interpretation |
|:---:|:---|
| $0.7 \le r \le 1.0$ | Strong positive linear relationship |
| $0.3 \le r < 0.7$ | Moderate positive linear relationship |
| $0 < r < 0.3$ | Weak positive linear relationship |
| $r = 0$ | No linear relationship |
| $-0.3 < r < 0$ | Weak negative linear relationship |
| $-0.7 < r \le -0.3$ | Moderate negative linear relationship |
| $-1.0 \le r \le -0.7$ | Strong negative linear relationship |

These thresholds are conventional guidelines, not strict rules. Context matters: in some fields (e.g., physics), $r = 0.7$ would be considered weak, while in social science it might be considered strong.

---

## The Coefficient of Determination

The square of the Pearson correlation, $r^2$, is called the **coefficient of determination**. It represents the proportion of variance in $Y$ that is linearly explained by $X$:

$$
r^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}
$$

where $\hat{y}_i$ are the fitted values from the simple linear regression of $Y$ on $X$.

For example, if $r = 0.80$, then $r^2 = 0.64$, meaning that 64% of the variability in $Y$ is accounted for by the linear relationship with $X$.

---

## Linearity Assumption

A critical limitation is that Pearson's $r$ measures only **linear** association. Two variables can have a strong nonlinear relationship yet produce $r \approx 0$.

!!! warning "Pearson's r can be misleading for nonlinear relationships"
    Consider $X$ uniformly distributed on $[-1, 1]$ and $Y = X^2$. There is a perfect deterministic relationship, but $r_{XY} = 0$ because the relationship is symmetric and nonlinear. Always inspect a scatter plot before interpreting $r$.

Anscombe's quartet provides a famous illustration: four datasets with nearly identical values of $r$ but very different scatter plot patterns. This underscores the importance of visualization alongside numerical summaries.

---

## Sensitivity to Outliers

The Pearson correlation is sensitive to outliers because it is based on means and standard deviations, which are themselves sensitive to extreme values. A single outlier can dramatically inflate or deflate $r$.

??? example "Outlier effect on correlation"
    Consider the dataset $(1,1), (2,2), (3,3), (4,4), (5,5)$, which has $r = 1.0$. Adding the single point $(10, -5)$ changes $r$ to approximately $0.24$. One outlier reduced a perfect correlation to a weak one.

When outliers are present or the data are non-normal, consider using rank-based alternatives such as [Spearman's correlation](spearman.md) or [Kendall's tau](kendall.md).

---

## Computation in Python

```python
import numpy as np
from scipy import stats

# Sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([2.1, 3.9, 6.2, 7.8, 10.1, 12.3, 13.8, 16.1, 18.0, 20.2])

# Method 1: NumPy correlation matrix
corr_matrix = np.corrcoef(x, y)
r_numpy = corr_matrix[0, 1]
print(f"NumPy r = {r_numpy:.4f}")

# Method 2: SciPy (also returns the p-value)
r_scipy, p_value = stats.pearsonr(x, y)
print(f"SciPy r = {r_scipy:.4f}, p-value = {p_value:.6f}")
```

The `scipy.stats.pearsonr` function returns both the sample correlation and a two-sided p-value for the null hypothesis $H_0\!: \rho = 0$. For details on the hypothesis test, see [Testing Pearson's r](../correlation_test/test_pearson.md).

---

## Summary

The Pearson correlation coefficient $r$ quantifies the strength and direction of the linear relationship between two variables. It ranges from $-1$ (perfect negative) to $+1$ (perfect positive), with $0$ indicating no linear association. While powerful and widely used, $r$ captures only linear relationships and is sensitive to outliers. Always pair the numerical value of $r$ with a scatter plot to verify that the linear model is appropriate.
