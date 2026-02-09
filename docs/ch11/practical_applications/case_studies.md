# ANOVA Practical Applications

This section presents complete worked examples demonstrating ANOVA assumption testing and diagnostics using Python. Each case study follows the full workflow: fit the model, check assumptions, and address any violations.

## Case Study 1: Iris Species (Plant Morphology)

### Background

We use the classic Iris dataset to test whether sepal length differs significantly between two species (versicolor and virginica). This example demonstrates the complete ANOVA diagnostic workflow.

### Step 1: Load Data and Fit Model

```python
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

data = sns.load_dataset("iris")
data = data[data["species"] != "setosa"]  # Two species for simplicity

model = ols('sepal_length ~ species', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
```

### Step 2: Check Normality

```python
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# Q-Q Plot
sm.qqplot(model.resid, line='s')
plt.title("Q-Q Plot of Residuals")
plt.show()

# Shapiro-Wilk Test
stat, p_value = shapiro(model.resid)
print(f"Shapiro-Wilk Test: W = {stat:.4f}, p-value = {p_value:.4f}")
```

### Step 3: Check Homoscedasticity

```python
from scipy.stats import levene

group1 = data[data['species'] == 'versicolor']['sepal_length']
group2 = data[data['species'] == 'virginica']['sepal_length']
stat, p_value = levene(group1, group2)
print(f"Levene's Test: F = {stat:.4f}, p-value = {p_value:.4f}")
```

### Step 4: Check Independence (Residual Plot)

```python
plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.show()
```

### Interpretation

If normality and homoscedasticity are not rejected (p > 0.05 for both tests) and the residual plot shows no systematic patterns, the ANOVA results can be interpreted with confidence. Otherwise, consider Welch's ANOVA or the Kruskal-Wallis test.

---

## Case Study 2: Employee Productivity by Working Environment

### Background

A company wants to determine if employee productivity differs across three working environments: remote, office, and hybrid. This example uses a small simulated dataset.

### Step 1: Load Data and Fit Model

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

data = pd.DataFrame({
    'productivity': [68, 75, 80, 65, 85, 78, 70, 82, 90, 88, 72, 95, 67, 85, 79],
    'environment': ['remote']*5 + ['office']*5 + ['hybrid']*5
})

model = ols('productivity ~ environment', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
```

### Step 2: Check Assumptions

```python
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene

# Normality
sm.qqplot(model.resid, line='s')
plt.title("Q-Q Plot of Residuals")
plt.show()

stat, p_value = shapiro(model.resid)
print(f"Shapiro-Wilk Test: p-value = {p_value:.4f}")

# Homoscedasticity
group1 = data[data['environment'] == 'remote']['productivity']
group2 = data[data['environment'] == 'office']['productivity']
group3 = data[data['environment'] == 'hybrid']['productivity']
stat, p_value = levene(group1, group2, group3)
print(f"Levene's Test: p-value = {p_value:.4f}")

# Independence
plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.show()
```

### Note on Small Samples

With only 5 observations per group, the Shapiro-Wilk test has low power and the Q-Q plot may not be very informative. In such cases, ANOVA relies heavily on the assumption that the underlying populations are normal, and it may be prudent to supplement with a non-parametric test.

---

## Case Study 3: Customer Satisfaction Across Store Locations

### Background

A retailer analyzes customer satisfaction scores across four store locations (A, B, C, D) to determine if there are significant differences.

### Step 1: Load Data and Fit Model

```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

data = pd.DataFrame({
    'satisfaction': [4.5, 3.8, 4.7, 4.2, 4.9, 4.1, 3.5, 4.3, 4.8, 3.9,
                     4.4, 4.0, 3.7, 4.2, 4.6, 4.8, 3.6, 4.3, 4.1, 4.7],
    'location': ['A']*5 + ['B']*5 + ['C']*5 + ['D']*5
})

model = ols('satisfaction ~ location', data=data).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
```

### Step 2: Check Assumptions

```python
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene

# Normality
sm.qqplot(model.resid, line='s')
plt.title("Q-Q Plot of Residuals")
plt.show()

stat, p_value = shapiro(model.resid)
print(f"Shapiro-Wilk Test: p-value = {p_value:.4f}")

# Homoscedasticity
groups = [data[data['location'] == loc]['satisfaction'] for loc in ['A', 'B', 'C', 'D']]
stat, p_value = levene(*groups)
print(f"Levene's Test: p-value = {p_value:.4f}")

# Independence
plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted Values')
plt.show()
```

### Step 3: Post-Hoc Analysis

If ANOVA reveals a significant difference and assumptions are met, perform post-hoc pairwise comparisons:

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

tukey = pairwise_tukeyhsd(data['satisfaction'], data['location'], alpha=0.05)
print(tukey)
```

For details on Tukey's HSD, see [Tukey HSD](../post_hoc/tukey.md).

---

## Summary

These case studies demonstrate a consistent workflow for ANOVA analysis:

1. **Fit the model** using `statsmodels.formula.api.ols`.
2. **Check normality** with Q-Q plots and the Shapiro-Wilk test.
3. **Check homoscedasticity** with Levene's test.
4. **Check independence** with residual plots.
5. **Address violations** if detected, using Welch's ANOVA, non-parametric tests, or transformations.
6. **Perform post-hoc tests** if the overall ANOVA is significant.

By following this workflow, you can ensure that your ANOVA results are robust and that conclusions are well-supported by the data.
