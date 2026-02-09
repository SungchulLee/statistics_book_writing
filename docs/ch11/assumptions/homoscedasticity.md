# Checking Homoscedasticity (Equal Variance)

## Why Homoscedasticity Matters

The assumption of homoscedasticity (also called homogeneity of variances) states that the variance of residuals should be approximately the same across all groups. In the ANOVA framework, the F-statistic pools the within-group variances into a single estimate of the common variance $\sigma^2$. If the true variances differ across groups, this pooled estimate is a weighted average that does not accurately represent any single group, leading to:

- An inflated or deflated F-statistic depending on the pattern of unequal variances and sample sizes.
- Inaccurate p-values and increased risk of Type I or Type II errors.
- Particularly severe distortions when unequal variances are combined with unequal sample sizes.

## How to Check

### Levene's Test

Levene's test assesses the null hypothesis that the population variances are equal across groups. It is more robust to departures from normality than Bartlett's test, making it the preferred choice in practice.

The test works by computing the absolute deviations from the group medians (or means) and then performing a one-way ANOVA on these deviations.

```python
from scipy.stats import levene

group1 = data[data['group'] == 'A']['response']
group2 = data[data['group'] == 'B']['response']
group3 = data[data['group'] == 'C']['response']

stat, p_value = levene(group1, group2, group3)
print(f"Levene's Test: F = {stat:.4f}, p-value = {p_value:.4f}")
```

A significant result ($p < 0.05$) indicates that the assumption of equal variances is violated. For a detailed treatment of Levene's test and related robust variance tests, see [Robust Variance Tests](../../ch16/robust_tests/robust_tests.md).

### Bartlett's Test

Bartlett's test is another test for homogeneity of variances. It is the uniformly most powerful test when the data are truly normal, but it is highly sensitive to departures from normality, making it less practical than Levene's test for real-world data.

```python
from scipy.stats import bartlett

stat, p_value = bartlett(group1, group2, group3)
print(f"Bartlett's Test: χ² = {stat:.4f}, p-value = {p_value:.4f}")
```

For a full discussion of Bartlett's test, see [Bartlett's Test](../../ch16/bartlett_test/bartlett_test.md).

### F-Test of Equality of Variances (Two Groups)

When comparing exactly two groups, the F-test for equality of variances uses the ratio of sample variances:

$$
F_{d_1,d_2} = \frac{S_1^2}{S_2^2}
$$

where $d_1 = n_1 - 1$ and $d_2 = n_2 - 1$ are the degrees of freedom. Under the null hypothesis $\sigma_1^2 = \sigma_2^2$, this ratio follows an $F$-distribution.

This test is derived from the relationship between the chi-squared distribution and sample variance:

$$
\chi^2_{n-1} = \frac{(n-1)S^2}{\sigma^2}
\quad\Rightarrow\quad
\frac{\chi^2_{n-1}}{n-1} = \frac{S^2}{\sigma^2}
$$

$$
F_{d_1,d_2} := \frac{\chi^2_{d_1}/d_1}{\chi^2_{d_2}/d_2} = \frac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2} = \frac{S_1^2}{S_2^2} \quad \text{if } \sigma_1 = \sigma_2
$$

!!! warning "Sensitivity to Normality"
    The F-test of equality of variances is extremely sensitive to non-normality. Even approximate normality may not be sufficient to make the test valid. For this reason, Levene's test or Brown-Forsythe test are generally preferred in practice.

For a detailed treatment, see [F-Test for Comparing Two Variances](../../ch16/f_test/f_test_two_variances.md).

### Residuals vs. Fitted Values Plot

A visual diagnostic plots residuals against the fitted values (predicted group means). Under homoscedasticity, the spread of residuals should be roughly constant across all fitted values.

```python
import matplotlib.pyplot as plt

plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.show()
```

Look for:

- **Funnel shape:** A widening or narrowing pattern indicates heteroscedasticity.
- **Constant band:** Residuals spread evenly around zero across all fitted values confirms homoscedasticity.

## What to Do If Homoscedasticity Is Violated

- **Welch's ANOVA:** Does not assume equal variances and adjusts degrees of freedom accordingly. This is the recommended first alternative (see [Welch's ANOVA](../anova_welch/welch_one_way.md)).
- **Data transformations:** Log, square root, or Box-Cox transformations can stabilize variance across groups.
- **Non-parametric tests:** The Kruskal-Wallis test does not assume equal variances.
- **Robust standard errors:** Heteroscedasticity-consistent standard errors (e.g., White's estimator) can be used in the regression formulation of ANOVA.
