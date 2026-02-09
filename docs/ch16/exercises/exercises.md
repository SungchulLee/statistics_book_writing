# Exercises

## Exercise 1: F-Test for Comparing Two Variances

Two production lines manufacture similar products, but you suspect that the variability in product weights is different between the two lines. You are provided with the following sample data (in grams):

- **Production Line 1:** $[14.2, 13.8, 15.1, 14.7, 14.5, 14.6, 15.0, 14.8]$
- **Production Line 2:** $[15.2, 14.8, 15.6, 15.0, 14.9, 15.3, 15.5, 14.9]$

Use the F-test to determine whether the variances of the two production lines are significantly different at the 5% significance level.

### Solution

**Hypotheses:**

- Null Hypothesis ($H_0$): The variances are equal, i.e., $\sigma_1^2 = \sigma_2^2$.
- Alternative Hypothesis ($H_1$): The variances are not equal, i.e., $\sigma_1^2 \neq \sigma_2^2$.

**Test Statistic:**

The F-statistic is calculated as the ratio of the sample variances:

$$
F = \frac{s_1^2}{s_2^2}
$$

where $s_1^2$ and $s_2^2$ are the sample variances of the two production lines.

**Python Implementation:**

```python
import numpy as np
from scipy.stats import f

# Sample data
line1 = np.array([14.2, 13.8, 15.1, 14.7, 14.5, 14.6, 15.0, 14.8])
line2 = np.array([15.2, 14.8, 15.6, 15.0, 14.9, 15.3, 15.5, 14.9])

# Sample variances
var_line1 = np.var(line1, ddof=1)
var_line2 = np.var(line2, ddof=1)

# F-statistic
F_stat = var_line1 / var_line2

# Degrees of freedom
df1 = len(line1) - 1
df2 = len(line2) - 1

# Critical values for two-tailed test at 5% significance level
alpha = 0.05
critical_value_upper = f.ppf(1 - alpha/2, df1, df2)
critical_value_lower = f.ppf(alpha/2, df1, df2)

print(f"F-statistic: {F_stat}")
print(f"Critical values: [{critical_value_lower}, {critical_value_upper}]")
```

**Interpretation:**

If the calculated F-statistic falls outside the critical value bounds, reject the null hypothesis. Otherwise, fail to reject the null hypothesis, indicating that the variances are not significantly different.

---

## Exercise 2: Levene's Test for Equality of Variances

Three different teaching methods are applied to three groups of students. After the semester, the students' scores are recorded as follows:

- **Group 1:** $[78, 82, 85, 90, 87]$
- **Group 2:** $[65, 70, 72, 68, 74]$
- **Group 3:** $[92, 88, 94, 89, 91]$

Use Levene's test to determine if the variances in the test scores are equal across the three groups.

### Solution

**Hypotheses:**

- Null Hypothesis ($H_0$): The variances are equal across the three groups.
- Alternative Hypothesis ($H_1$): At least one group has a variance that differs from the others.

**Test Statistic:**

Levene's test calculates the absolute deviations from the group medians and tests whether the variance of these deviations differs across groups.

**Python Implementation:**

```python
from scipy.stats import levene

# Test scores for the three groups
group1 = [78, 82, 85, 90, 87]
group2 = [65, 70, 72, 68, 74]
group3 = [92, 88, 94, 89, 91]

# Perform Levene's test
statistic, p_value = levene(group1, group2, group3)

print(f"Levene's test statistic: {statistic}")
print(f"P-value: {p_value}")
```

**Interpretation:**

If the p-value is less than $0.05$, reject the null hypothesis and conclude that the variances are not equal across the groups.

---

## Exercise 3: Interpreting Conflicting Results from Bartlett's Test and Levene's Test

For two datasets that are not normally distributed, a Bartlett's test was performed, resulting in a low p-value (rejection of $H_0$: equal variances). However, a Levene's test yielded a high p-value (failure to reject $H_0$). How should these two test results be interpreted?

### Solution

- The results of Bartlett's test are **not reliable** when the assumption of normality is violated. Bartlett's test is highly sensitive to non-normality, and its rejection of the null hypothesis may be driven by the distributional shape rather than actual differences in variance.
- Levene's test is **less sensitive** to violations of normality, so it provides a more trustworthy result in this scenario. It is reasonable to conclude that the variances of the two datasets are equal.

---

## Exercise 4: Interpreting Variance and Mean Comparison Results

Samples were drawn from two populations and analyzed. A Levene's test was performed, resulting in a high p-value (failure to reject $H_0$: equal variances). However, a t-test assuming equal variances yielded a p-value smaller than 0.001. How should these two test results be interpreted?

### Solution

- The Levene's test supports the assumption that the **variances are equal** across the two populations.
- The t-test result provides **strong evidence** that the **means** of the two populations are significantly different.
- These results are not contradictory — two populations can have equal variances while having very different means. The Levene's test validates the equal-variance assumption used in the t-test, which strengthens the conclusion that the observed mean difference is genuine.
