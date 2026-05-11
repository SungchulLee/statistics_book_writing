# Applications of Normality Tests in Statistics


Normality tests are an essential part of statistical analysis because many common statistical methods rely on the assumption that the data is normally distributed.

## When to Apply a Normality Test

Normality tests are typically applied before using parametric statistical methods that assume normality. Common scenarios include:

- **$t$-tests**: Both the one-sample and two-sample $t$-tests assume that the data is normally distributed within each group.
- **ANOVA**: The Analysis of Variance test assumes that residuals are normally distributed across groups.
- **Linear Regression**: Regression analysis assumes that the residuals (errors) of the model are normally distributed.
- **Confidence Intervals**: Normality is often assumed when constructing confidence intervals for population parameters, especially with small sample sizes.

A normality test is useful to determine whether it is appropriate to use these methods or if alternative approaches (such as transformations or non-parametric tests) should be considered.

## Case Study 1: Testing Normality in t-Test Assumptions
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

## Exercises

**Exercise 1.**
A researcher performs a one-sample t-test on $n = 15$ observations and obtains $t = 2.35$. Before interpreting the result, they should check for normality. Explain why, and describe what could go wrong if the data are heavily skewed.

??? success "Solution to Exercise 1"
    The one-sample t-test assumes the data come from a normal distribution (or that $n$ is large enough for the CLT). With $n = 15$, the CLT may not provide sufficient approximation if the data are heavily skewed.

    If the data are right-skewed, the sampling distribution of $\bar{X}$ is also skewed, and the t-distribution is a poor approximation. This can lead to: (1) incorrect p-values (the actual Type I error rate differs from the nominal $\alpha$), (2) confidence intervals with incorrect coverage, and (3) reduced power to detect real effects. With heavy skewness and $n = 15$, a nonparametric test (Wilcoxon signed-rank) or a bootstrap test would be more reliable.

---

**Exercise 2.**
Name three statistical methods that rely on normality assumptions and state for each how robust it is to non-normality.

??? success "Solution to Exercise 2"

    1. **t-test for a mean:** Moderately robust. With $n \geq 30$ and moderate skewness, the CLT ensures approximate validity. Not robust to heavy tails or extreme outliers in small samples.

    2. **F-test for equality of variances:** Not robust. The F-test is highly sensitive to non-normality; even mild departures can severely inflate the Type I error rate. Levene's or Brown-Forsythe tests are preferred alternatives.

    3. **Linear regression (OLS):** The OLS estimates are valid (unbiased, BLUE) without normality. However, inference (t-tests, F-tests, confidence intervals) requires normality of errors or large $n$. Prediction intervals are especially sensitive to non-normality.

---

**Exercise 3.**
Explain the practical workflow for checking normality before performing a statistical test.

??? success "Solution to Exercise 3"
    A recommended workflow:

    1. **Visual inspection first:** Create a histogram or density plot and a Q-Q plot of the data (or residuals for regression). Look for skewness, heavy tails, outliers, or multimodality.

    2. **Formal test:** Apply a normality test (Shapiro-Wilk for $n < 50$, Anderson-Darling or D'Agostino for larger samples) as a supplement to visual methods.

    3. **Interpret results together:** If the Q-Q plot shows approximate linearity and the formal test does not reject at a reasonable level, proceed with normal-theory methods. If both suggest non-normality, consider alternatives.

    4. **Choose a remedy if needed:** Apply a transformation (log, Box-Cox), use a nonparametric test, or use bootstrap methods.

    5. **Report the assessment:** State which normality checks were performed and their results, even if normality is supported.

---

**Exercise 4.**
For a large sample ($n = 5000$), a Shapiro-Wilk test rejects normality with $p < 0.001$, but the Q-Q plot looks nearly linear. How should you proceed?

??? success "Solution to Exercise 4"
    With $n = 5000$, formal tests have extremely high power and will detect trivially small departures from normality that have no practical impact on inference. A Shapiro-Wilk p-value of $< 0.001$ does not mean the data are "far" from normal -- it means the departure is statistically detectable.

    Since the Q-Q plot looks nearly linear, the departure is likely small. You should:

    1. **Proceed with normal-theory methods:** For $n = 5000$, the CLT provides strong protection, and the t-test/ANOVA/regression inference will be very accurate even with slight non-normality.
    2. **Report both findings:** Note that the formal test rejects normality but visual inspection suggests approximate normality.
    3. **Consider effect size:** Quantify the degree of non-normality using skewness and kurtosis coefficients rather than relying on a binary reject/fail-to-reject decision.
