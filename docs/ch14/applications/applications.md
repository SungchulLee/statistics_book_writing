# Applications of Normality Tests in Statistics

Normality tests are an essential part of statistical analysis because many common statistical methods rely on the assumption that the data is normally distributed.

## When to Apply a Normality Test

Normality tests are typically applied before using parametric statistical methods that assume normality. Common scenarios include:

- **$t$-tests**: Both the one-sample and two-sample $t$-tests assume that the data is normally distributed within each group.
- **ANOVA**: The Analysis of Variance test assumes that residuals are normally distributed across groups.
- **Linear Regression**: Regression analysis assumes that the residuals (errors) of the model are normally distributed.
- **Confidence Intervals**: Normality is often assumed when constructing confidence intervals for population parameters, especially with small sample sizes.

A normality test is useful to determine whether it is appropriate to use these methods or if alternative approaches (such as transformations or non-parametric tests) should be considered.

## Case Study 1: Testing Normality in $t$-Test Assumptions

A two-sample $t$-test assumes that data within each sample is normally distributed. Before performing the $t$-test, it is essential to check the normality of both groups.

```python
import numpy as np
from scipy.stats import ttest_ind, shapiro

# Generate two sample datasets
group1 = np.random.normal(0, 1, 50)
group2 = np.random.normal(0.5, 1, 50)

# Perform Shapiro-Wilk test for normality on both groups
_, p_value_group1 = shapiro(group1)
_, p_value_group2 = shapiro(group2)

if p_value_group1 > 0.05 and p_value_group2 > 0.05:
    # If both groups pass the normality test, perform a t-test
    stat, p_value = ttest_ind(group1, group2)
    print(f"Two-sample t-test: p-value={p_value}")
else:
    print("One or both groups fail the normality test. Consider using a non-parametric alternative.")
```

If one or both groups fail the normality test, a non-parametric alternative such as the **Mann-Whitney U Test** should be used.

## Case Study 2: Normality in Linear Regression Residuals

In linear regression, it is assumed that the residuals (the differences between the observed and predicted values) are normally distributed. A normality test can be applied to residuals to check whether this assumption holds.

```python
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# Generate example data
np.random.seed(0)
X = np.random.normal(0, 1, 100)
y = 2 * X + np.random.normal(0, 1, 100)

# Add a constant to X for the intercept
X = sm.add_constant(X)

# Fit the linear model
model = sm.OLS(y, X).fit()

# Get the residuals
residuals = model.resid

# Perform a Shapiro-Wilk test on the residuals
_, p_value = shapiro(residuals)

print(f"Shapiro-Wilk Test on Residuals: p-value={p_value}")

# Plot residuals
plt.hist(residuals, bins=20)
plt.title('Residuals Histogram')
plt.show()

if p_value > 0.05:
    print("Residuals are normally distributed.")
else:
    print("Residuals are not normally distributed.")
```

If the residuals are not normally distributed, the results of the regression analysis might be unreliable, and corrective measures such as transformations or alternative regression models may be necessary.

## Case Study 3: Normality in ANOVA

The **ANOVA (Analysis of Variance)** test assumes that the residuals of the data across groups are normally distributed. If this assumption is violated, the results of ANOVA may be misleading.

```python
import numpy as np
from scipy.stats import f_oneway, shapiro

# Generate sample data for three groups
group1 = np.random.normal(0, 1, 30)
group2 = np.random.normal(0.5, 1, 30)
group3 = np.random.normal(1, 1, 30)

# Perform Shapiro-Wilk test on the residuals
_, p_value_group1 = shapiro(group1)
_, p_value_group2 = shapiro(group2)
_, p_value_group3 = shapiro(group3)

# Check if the data is normally distributed
if p_value_group1 > 0.05 and p_value_group2 > 0.05 and p_value_group3 > 0.05:
    # Perform ANOVA
    stat, p_value = f_oneway(group1, group2, group3)
    print(f"ANOVA test: p-value={p_value}")
else:
    print("One or more groups fail the normality test. Consider using a non-parametric alternative.")
```

Before applying ANOVA, the Shapiro-Wilk test is used to check if the data in each group is normally distributed. If one or more groups fail the test, a non-parametric alternative like the **Kruskal-Wallis Test** might be more appropriate.

## Conclusion

Normality tests are crucial in various applications where parametric methods such as $t$-tests, ANOVA, and linear regression are used. Ensuring that the data (or residuals) follow a normal distribution allows these methods to produce valid results. When normality assumptions are violated, transformations or non-parametric alternatives can often be applied. In practice, combining normality tests with graphical assessments helps provide a clearer picture of the underlying data distribution.
