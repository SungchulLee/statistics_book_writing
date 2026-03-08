# Test of Independence

## Overview

An **Independence Test** is a statistical technique used to determine if there is a significant relationship between two categorical variables. Essentially, it helps answer the question: "Do the occurrences of one variable affect the occurrences of another?" If the two variables are independent, changes in one variable should have no effect on the distribution of the other.

## Example Scenario

For example, let's say we want to examine whether there is an association between gender (male or female) and preference for a particular type of beverage (coffee, tea, or juice). The Independence Test helps assess whether gender influences beverage preference, or whether the preferences are independent of gender.

## Hypotheses

- **Null Hypothesis ($H_0$)**: The two variables are independent (i.e., there is no association between the variables).
- **Alternative Hypothesis ($H_A$)**: The two variables are not independent (i.e., there is an association between the variables).

## Contingency Table

A contingency table is a matrix format table that shows the frequency distribution of variables. For example, if you want to check if there is an association between gender (male, female) and preference for a product (like, dislike), the table might look like this:

|           | Like | Dislike | Total |
|-----------|------|---------|-------|
| Male      | 30   | 20      | 50    |
| Female    | 25   | 25      | 50    |
| **Total** | 55   | 45      | 100   |

## Expected Frequencies

Under the independence assumption, the expected frequency for each cell is computed as:

$$
E_{ij} = \frac{\text{(Row Total for Row } i\text{)} \times \text{(Column Total for Column } j\text{)}}{\text{Grand Total}}
$$

## Test Statistic

The test statistic for the Chi-Square Test of Independence is calculated using the formula:

$$
\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

where:

- $O_{ij}$ = Observed frequency in cell $ij$
- $E_{ij}$ = Expected frequency in cell $ij$

## Degrees of Freedom

The degrees of freedom ($\text{df}$) for the test is calculated as:

$$
\text{df} = (r - 1) \times (c - 1)
$$

where $r$ is the number of rows and $c$ is the number of columns.

## Critical Region

$$
\begin{array}{lll}
\text{Null} & \text{They are independent} \\
& \text{Observed frequencies are close to expected frequencies} \\
& O_{ij} \approx E_{ij} \quad \Rightarrow \quad \text{statistic} \approx 0 \\
\\
\text{Alternative} & \text{They are not independent} \\
& \text{Observed frequencies are quite different from expected frequencies} \\
& O_{ij} \not\approx E_{ij} \quad \Rightarrow \quad \text{statistic} \approx \text{large positive number}
\end{array}
$$

## Critical Value and p-Value

- **Critical Value**: The critical value is determined from the Chi-Square distribution table, based on the degrees of freedom and the chosen significance level (e.g., 0.05).
- **p-Value**: The p-value is calculated from the Chi-Square distribution using the test statistic and degrees of freedom. It represents the probability of observing a test statistic as extreme as, or more extreme than, the one calculated under the null hypothesis.

## Decision Rule

- If the test statistic exceeds the critical value, or if the p-value is less than the significance level, reject the null hypothesis. This suggests that the variables are not independent and there is an association between them.
- If the test statistic does not exceed the critical value, or if the p-value is greater than the significance level, fail to reject the null hypothesis. This indicates that the variables are independent.

## Assumptions and Limitations

The Chi-Square Test of Independence assumes that the observations are randomly sampled, the expected frequency in each cell is at least 5, and that the categories are mutually exclusive. If these assumptions are violated, the results may be misleading.

Other independence tests, like **Fisher's Exact Test**, may be preferred if sample sizes are small, as it does not rely on the large sample approximation used by the Chi-Square Test.

---

## Example A: Gender vs Right-Handedness

### Question

We randomly selected several people and recorded their sex and dominant hand. Here is the data.

**Observed:**

$$
\begin{array}{crr|r}
 & \text{men} & \text{women} & \text{row sum} \\ \hline
\text{right-handed} & 934 & 1{,}070 & 2{,}004 \\
\text{left-handed} & 113 & 92 & 205 \\
\text{ambidextrous} & 20 & 8 & 28 \\ \hline
\text{column sum} & 1{,}067 & 1{,}170 & 2{,}237
\end{array}
$$

Is there any relationship between sex and the dominant hand, or are these two variables independent?

### Hypotheses

$$
\begin{array}{lll}
\text{Null} & \text{They are independent} \\
\\
\text{Alternative} & \text{They are not independent}
\end{array}
$$

### Expected Frequencies

**Expected:**

$$
\begin{array}{ccc|r}
 & \text{men} & \text{women} & \text{row sum} \\ \hline
\text{right-handed} & 956 & 1{,}048 & 2{,}004 \\
\text{left-handed} & 98 & 107 & 205 \\
\text{ambidextrous} & 13 & 15 & 28 \\ \hline
\text{column sum} & 1{,}067 & 1{,}170 & 2{,}237
\end{array}
$$

**How to compute expected frequencies**: If they are independent,

$$
P(\text{men}) = \frac{1067}{2237}, \quad P(\text{right-handed}) = \frac{2004}{2237}
$$

$$
\Rightarrow P(\text{men}, \text{right-handed}) = \frac{1067}{2237} \times \frac{2004}{2237}
$$

$$
\Rightarrow \text{expected frequency}(\text{men}, \text{right-handed}) = \frac{1067}{2237} \times \frac{2004}{2237} \times 2237 \approx 956
$$

### p-value

$$
\text{p-value} = P\left(\sum_{i=1}^{r}\sum_{j=1}^{c}\frac{(O_{ij}-E_{ij})^2}{E_{ij}} \ge \text{statistic} \;\middle|\; H_0\right)
$$

### Conclusion

$$\text{They are not independent.}$$

### Python Implementation (Without `scipy.stats.chi2_contingency`)

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def compute_expected(observed_counts):
    """
    Computes the expected frequency table based on observed counts for a chi-squared test.

    Parameters:
    observed_counts (numpy array): 2D array of observed counts in contingency table format.

    Returns:
    numpy array: 2D array of expected counts based on the marginal probabilities.
    """
    row_totals = observed_counts.sum(axis=1)
    col_totals = observed_counts.sum(axis=0)

    row_pmf = row_totals.reshape((-1, 1)) / row_totals.sum()
    col_pmf = col_totals.reshape((1, -1)) / col_totals.sum()

    joint_pmf = row_pmf * col_pmf
    expected_counts = joint_pmf * row_totals.sum()

    return expected_counts

# Observed counts in contingency table format
observed_counts = np.array([[934, 1070], [113, 92], [20, 8]])
expected_counts = compute_expected(observed_counts)

# Degrees of freedom for chi-squared test
degrees_of_freedom = (observed_counts.shape[0] - 1) * (observed_counts.shape[1] - 1)

# Chi-squared statistic calculation
chi_squared_statistic = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
p_value = stats.chi2(degrees_of_freedom).sf(chi_squared_statistic)

# Print the chi-squared statistic and p-value
print(f"chi_squared_statistic = {chi_squared_statistic:.02f}")
print(f"p_value = {p_value:.02%}")

# Plot chi-squared distribution and highlight observed statistic
fig, ax = plt.subplots(figsize=(12, 4))

x_values = np.linspace(0, chi_squared_statistic, 100)
y_values = stats.chi2(degrees_of_freedom).pdf(x_values)
ax.plot(x_values, y_values, color='b', linewidth=3)

x_fill_left = np.concatenate([[0], x_values, [chi_squared_statistic], [0]])
y_fill_left = np.concatenate([[0], y_values, [0], [0]])
ax.fill(x_fill_left, y_fill_left, color='b', alpha=0.1)

x_values_right = np.linspace(chi_squared_statistic, 20, 100)
y_values_right = stats.chi2(degrees_of_freedom).pdf(x_values_right)
ax.plot(x_values_right, y_values_right, color='r', linewidth=3)

x_fill_right = np.concatenate([[chi_squared_statistic], x_values_right, [20], [chi_squared_statistic]])
y_fill_right = np.concatenate([[0], y_values_right, [0], [0]])
ax.fill(x_fill_right, y_fill_right, color='r', alpha=0.1)

ax.annotate(f'p_value = {p_value:.02%}', xy=(12.5, 0.01), xytext=(16.5, 0.10),
            fontsize=15, arrowprops=dict(color='k', width=0.2, headwidth=8))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position("zero")
ax.spines['left'].set_position("zero")

plt.show()
```

### Python Implementation (With `scipy.stats.chi2_contingency`)

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Observed counts in a contingency table format
observed_counts = np.array([[934, 1070], [113, 92], [20, 8]])

# Perform chi-squared test of independence
chi_squared_statistic, p_value, degrees_of_freedom, expected_counts = stats.chi2_contingency(observed_counts)

# Print the chi-squared statistic and p-value
print(f"chi_squared_statistic = {chi_squared_statistic:.02f}")
print(f"p_value = {p_value:.02%}", end="\n\n")

print("expected_counts")
print(expected_counts, end="\n\n")

# Plot
fig, ax = plt.subplots(figsize=(12, 4))

x_values = np.linspace(0, chi_squared_statistic, 100)
y_values = stats.chi2(degrees_of_freedom).pdf(x_values)
ax.plot(x_values, y_values, color='b', linewidth=3)

x_fill_left = np.concatenate([[0], x_values, [chi_squared_statistic], [0]])
y_fill_left = np.concatenate([[0], y_values, [0], [0]])
ax.fill(x_fill_left, y_fill_left, color='b', alpha=0.1)

x_values_right = np.linspace(chi_squared_statistic, 20, 100)
y_values_right = stats.chi2(degrees_of_freedom).pdf(x_values_right)
ax.plot(x_values_right, y_values_right, color='r', linewidth=3)

x_fill_right = np.concatenate([[chi_squared_statistic], x_values_right, [20], [chi_squared_statistic]])
y_fill_right = np.concatenate([[0], y_values_right, [0], [0]])
ax.fill(x_fill_right, y_fill_right, color='r', alpha=0.1)

ax.annotate(f'p_value = {p_value:.02%}', xy=(12.5, 0.01), xytext=(16.5, 0.10),
            fontsize=15, arrowprops=dict(color='k', width=0.2, headwidth=8))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position("zero")
ax.spines['left'].set_position("zero")

plt.show()
```

---

## Example B: Longer Hand vs. Longer Foot

> **Source**: [Khan Academy — Chi-Square Test Association Independence](https://www.khanacademy.org/math/ap-statistics/chi-square-tests/chi-square-tests-two-way-tables/v/chi-square-test-association-independence)

We suspect there might be a relationship between foot length and hand length. The null hypothesis assumes no association or independence between the variables. The alternative hypothesis expresses our suspicion that there is indeed an association between foot and hand lengths, suggesting that they are not independent.

We randomly sample 100 individuals. For each individual, we determine whether their right hand is longer, their left hand is longer, or both hands are of equal length. We repeat the same process for foot length.

|                   | Right Foot Longer | Left Foot Longer | Both Feet Same |
|:-----------------:|:-----------------:|:----------------:|:--------------:|
| Right Hand Longer | 11                | 3                | 8              |
| Left Hand Longer  | 2                 | 9                | 14             |
| Both Hands Same   | 12                | 13               | 28             |

### Solution

**Step 1: Hypotheses**

- $H_0$: Foot length and hand length are independent.
- $H_1$: Foot length and hand length are not independent.

**Step 2: Observed Frequencies with Totals**

$$
\begin{array}{c|c|c|c|c}
 & \text{Right Foot} & \text{Left Foot} & \text{Both Same} & \text{Row Total} \\
\hline
\text{Right Hand} & 11 & 3 & 8 & 22 \\
\text{Left Hand} & 2 & 9 & 14 & 25 \\
\text{Both Same} & 12 & 13 & 28 & 53 \\
\hline
\text{Col Total} & 25 & 25 & 50 & 100
\end{array}
$$

**Step 3: Expected Frequencies**

$$
E_{ij} = \frac{\text{(Row Total for Row } i\text{)} \times \text{(Column Total for Column } j\text{)}}{\text{Grand Total}}
$$

$$
\begin{array}{c|c|c|c}
 & \text{Right Foot} & \text{Left Foot} & \text{Both Same} \\
\hline
\text{Right Hand} & 5.5 & 5.5 & 11 \\
\text{Left Hand} & 6.25 & 6.25 & 12.5 \\
\text{Both Same} & 13.25 & 13.25 & 26.5
\end{array}
$$

**Step 4: Test Statistic**

Computing each term $(O_{ij} - E_{ij})^2 / E_{ij}$ and summing:

$$
\chi^2 \approx 5.5 + 1.136 + 0.818 + 2.89 + 1.21 + 0.18 + 0.118 + 0.005 + 0.085 \approx 11.94
$$

**Step 5: Degrees of Freedom**

$$
\text{df} = (3 - 1)(3 - 1) = 4
$$

**Step 6: p-value**

The p-value for $\chi^2 = 11.94$ with $\text{df} = 4$ is approximately **0.018**.

**Conclusion**: Since the p-value (0.018) is below the conventional significance level of 0.05, we reject the null hypothesis. This suggests there is evidence of an association between foot length and hand length.

### Python Implementation

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def main():
    observed = np.array([[11, 3, 8], [2, 9, 14], [12, 13, 28]])

    statistic, p_value, df, expected = stats.chi2_contingency(observed)
    print(f"{statistic = :.02f}")
    print(f"{p_value   = :.02%}")

    _, ax = plt.subplots(figsize=(12, 4))

    x = np.linspace(0, statistic)
    y = stats.chi2(df).pdf(x)
    ax.plot(x, y, color='b', linewidth=3)

    x = np.concatenate([[0], x, [statistic], [0]])
    y = np.concatenate([[0], y, [0], [0]])
    ax.fill(x, y, color='b', alpha=0.1)

    x = np.linspace(statistic, 20, 100)
    y = stats.chi2(df).pdf(x)
    ax.plot(x, y, color='r', linewidth=3)

    x = np.concatenate([[statistic], x, [20], [statistic]])
    y = np.concatenate([[0], y, [0], [0]])
    ax.fill(x, y, color='r', alpha=0.1)

    xy = (15.0, 0.01)
    xytext = (16.5, 0.10)
    arrowprops = dict(color='k', width=0.2, headwidth=8)
    ax.annotate(f'{p_value = :.02%}', xy, xytext=xytext, fontsize=15, arrowprops=arrowprops)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position("zero")
    ax.spines['left'].set_position("zero")

    plt.show()

if __name__ == "__main__":
    main()
```

---

## Example C: Detailed Expected Frequency Computation

This example demonstrates a full step-by-step expected frequency calculation for a larger contingency table.

```python
"""
Expected Frequency Calculator for Contingency Table
Computes expected frequencies under independence assumption
"""

import numpy as np
import pandas as pd
from scipy import stats

# Observed frequencies
observed = np.array([
    [10, 20, 30, 40, 20],
    [5, 15, 40, 50, 10],
    [10, 10, 20, 30, 19]
])

alpha = 0.05

# Calculate marginal totals
row_totals = observed.sum(axis=1)
col_totals = observed.sum(axis=0)
grand_total = observed.sum()

print("=" * 70)
print("OBSERVED FREQUENCIES")
print("=" * 70)
obs_df = pd.DataFrame(observed,
                      index=['Row 1 (20)', 'Row 2 (30)', 'Row 3 (40)'],
                      columns=['Col 1', 'Col 2', 'Col 3', 'Col 4', 'Col 5'])
obs_df['Row Total'] = row_totals
print(obs_df)
print(f"\nColumn Totals: {col_totals}")
print(f"Grand Total: {grand_total}")

print("\n" + "=" * 70)
print("EXPECTED FREQUENCIES")
print("=" * 70)
print("Formula: E_ij = (Row_i_total × Column_j_total) / Grand_total\n")

# Calculate expected frequencies
expected = np.zeros_like(observed, dtype=float)
for i in range(observed.shape[0]):
    for j in range(observed.shape[1]):
        expected[i, j] = (row_totals[i] * col_totals[j]) / grand_total

# Display expected frequencies
exp_df = pd.DataFrame(expected,
                      index=['Row 1 (20)', 'Row 2 (30)', 'Row 3 (40)'],
                      columns=['Col 1', 'Col 2', 'Col 3', 'Col 4', 'Col 5'])
exp_df['Row Total'] = exp_df.sum(axis=1)
print(exp_df)
print(f"\nColumn Totals: {expected.sum(axis=0)}")
print(f"Grand Total: {expected.sum()}")

print("\n" + "=" * 70)
print("DETAILED EXPECTED FREQUENCY CALCULATIONS")
print("=" * 70)
for i in range(observed.shape[0]):
    print(f"\nRow {i+1} (Row Total = {row_totals[i]}):")
    for j in range(observed.shape[1]):
        calculation = f"E[{i+1},{j+1}] = ({row_totals[i]} × {col_totals[j]}) / {grand_total}"
        result = f"= {row_totals[i] * col_totals[j]} / {grand_total} = {expected[i,j]:.4f}"
        print(f"  {calculation} {result}")

print("\n" + "=" * 70)
print("CHI-SQUARE CONTRIBUTIONS")
print("=" * 70)
print("Formula: (Observed - Expected)² / Expected\n")

chi_sq_contrib = (observed - expected)**2 / expected
chi_sq_df = pd.DataFrame(chi_sq_contrib,
                         index=['Row 1 (20)', 'Row 2 (30)', 'Row 3 (40)'],
                         columns=['Col 1', 'Col 2', 'Col 3', 'Col 4', 'Col 5'])
print(chi_sq_df)
print(f"\nChi-square statistic: {chi_sq_contrib.sum():.4f}")
print(f"Degrees of freedom: {(observed.shape[0]-1) * (observed.shape[1]-1)}")

p_value = stats.chi2(df=(observed.shape[0]-1) * (observed.shape[1]-1)).sf(chi_sq_contrib.sum())
print(f"{p_value = :.4f}")
if p_value < alpha:
    print("We have enough evidence to reject the null hypothesis that X and Y are independent.")
else:
    print("We do not have enough evidence to reject the null hypothesis that X and Y are independent.")
```

---

## 4. Resampling-Based Chi-Square Test

For situations with small sample sizes, low expected cell frequencies, or when you want a distribution-free approach, permutation/resampling-based chi-square tests provide an alternative to the asymptotic chi-square distribution.

### Algorithm

The resampling approach tests independence by:

1. **Calculate observed chi-square statistic** from the actual contingency table
2. **Generate expected cell probabilities** under independence
3. **Simulate B contingency tables** by randomly allocating observations according to independence assumption
4. **Calculate chi-square for each simulated table**
5. **Compute p-value**: proportion of simulated chi-squares as extreme as or more extreme than observed

### Example: Headline Click Rates (A/B Testing)

Three headlines are tested with users; we measure whether they clicked or not. This is typical in digital marketing A/B testing.

**Observed Data:**

```python
import numpy as np
import pandas as pd
import random
from scipy import stats

# Click rates for three headlines
headlines = pd.DataFrame({
    'Click': [14, 8, 12],
    'No-click': [986, 992, 988],
    'Headline': ['Headline A', 'Headline B', 'Headline C']
})

# Create contingency table
click_rate = headlines.copy()
clicks = click_rate.set_index('Headline')[['Click', 'No-click']]

print("Observed Contingency Table:")
print(clicks)
print(f"\nTotal: {clicks.values.sum()}")
```

### Resampling Approach (Without Replacement)

```python
def chi2_stat(observed, expected):
    """Calculate chi-square statistic."""
    pearson_residuals = []
    for row, expect in zip(observed, expected):
        pearson_residuals.append([(observe - expect) ** 2 / expect
                                  for observe in row])
    return np.sum(pearson_residuals)

# Observed chi-square
row_average = clicks.mean(axis=1)
expected = np.array([[row_average['Click'], row_average['Click'], row_average['Click']],
                     [row_average['No-click'], row_average['No-click'], row_average['No-click']]])

chi2_obs = chi2_stat(clicks.values, row_average.values)
print(f"Observed chi-square: {chi2_obs:.4f}")

# Resampling approach
def perm_fun_chisq(box):
    """
    Generate permuted contingency table by random allocation.

    Parameters:
    -----------
    box : list
        Binary response (1 = click, 0 = no-click) for all users

    Returns:
    --------
    float : Chi-square statistic for permuted table
    """
    random.shuffle(box)
    # Allocate first 1000 to Headline A, next 1000 to B, last 1000 to C
    sample_clicks = [sum(box[0:1000]),
                     sum(box[1000:2000]),
                     sum(box[2000:3000])]
    sample_noclicks = [1000 - n for n in sample_clicks]
    return chi2_stat([sample_clicks, sample_noclicks], row_average.values)

# Create box: 1 for each click, 0 for each non-click
box = [1] * 34 + [0] * 2966

# Run permutation test
random.seed(42)
perm_chi2 = [perm_fun_chisq(box) for _ in range(2000)]

p_value_resamp = sum(np.array(perm_chi2) >= chi2_obs) / len(perm_chi2)
print(f"Resampling p-value: {p_value_resamp:.4f}")
```

### Resampling Approach (With Replacement)

Alternatively, sample with replacement from the box:

```python
def sample_with_replacement(box):
    """
    Generate permuted contingency table by sampling with replacement.
    """
    sample_clicks = [sum(random.sample(box, 1000)),
                     sum(random.sample(box, 1000)),
                     sum(random.sample(box, 1000))]
    sample_noclicks = [1000 - n for n in sample_clicks]
    return chi2_stat([sample_clicks, sample_noclicks], row_average.values)

# Run with-replacement resampling
random.seed(42)
perm_chi2_wr = [sample_with_replacement(box) for _ in range(2000)]

p_value_wr = sum(np.array(perm_chi2_wr) >= chi2_obs) / len(perm_chi2_wr)
print(f"Resampling (with replacement) p-value: {p_value_wr:.4f}")
```

### Comparison: Resampling vs. Parametric

```python
# Parametric chi-square test
chi2_param, p_param, df, expected_param = stats.chi2_contingency(clicks.values)

print(f"\nComparison:")
print(f"Parametric chi-square: {chi2_param:.4f}, p-value: {p_param:.4f}")
print(f"Resampling (without repl): p-value: {p_value_resamp:.4f}")
print(f"Resampling (with repl): p-value: {p_value_wr:.4f}")
```

### Visualization

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Resampling distribution (without replacement)
ax1.hist(perm_chi2, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
ax1.axvline(chi2_obs, color='red', linewidth=2, label=f'Observed = {chi2_obs:.2f}')
ax1.set_xlabel('Chi-square Statistic')
ax1.set_ylabel('Frequency')
ax1.set_title(f'Resampling Distribution (without replacement)\np-value = {p_value_resamp:.4f}')
ax1.legend()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Resampling distribution (with replacement)
ax2.hist(perm_chi2_wr, bins=40, alpha=0.7, color='forestgreen', edgecolor='black')
ax2.axvline(chi2_obs, color='red', linewidth=2, label=f'Observed = {chi2_obs:.2f}')
ax2.set_xlabel('Chi-square Statistic')
ax2.set_ylabel('Frequency')
ax2.set_title(f'Resampling Distribution (with replacement)\np-value = {p_value_wr:.4f}')
ax2.legend()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

### Advantages of Resampling Chi-Square

1. **No distributional assumptions**: Does not rely on chi-square approximation
2. **Small cell counts**: Works even when expected cell counts < 5
3. **Exact**: p-value is exact (not approximate)
4. **Flexible**: Can be applied to any contingency table size

### When to Use Resampling

- **Small expected frequencies**: Any expected cell count < 5
- **Small sample sizes**: n < 20-30
- **Robustness check**: Compare against parametric chi-square
- **Pedagogical value**: Directly tests the null hypothesis through randomization

### Computational Considerations

- Use 2,000-5,000 permutations for most applications
- Without-replacement is more conservative; with-replacement is more liberal
- Both approaches typically give similar p-values for moderate sample sizes
