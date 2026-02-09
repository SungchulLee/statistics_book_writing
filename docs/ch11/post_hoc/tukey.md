# Post-Hoc Comparisons: Tukey HSD

## 1. Post-Hoc Tests in One-Way ANOVA

When conducting a **one-way ANOVA**, we may find that there is a significant difference between the group means. However, a significant result from the ANOVA test does not tell us which specific groups are different from each other. **Post-hoc tests** are used in this case to identify the specific pairs of groups that differ significantly. These tests help control for Type I error (false positives) when making multiple comparisons.

### A. Understanding Post-Hoc Tests for One-Way ANOVA

In a one-way ANOVA, post-hoc tests are conducted when:

1. The one-way ANOVA indicates a statistically significant difference between group means.
2. We want to know which specific groups differ from each other.

### B. Types of Post-Hoc Tests for One-Way ANOVA

The most common post-hoc tests for one-way ANOVA include:

- **Tukey's Honest Significant Difference (HSD)**: A widely used method that controls the family-wise error rate. Tukey's HSD is appropriate for equal group sizes but can also be used with slight inequalities.
- **Bonferroni Correction**: A conservative approach that adjusts the significance level by dividing it by the number of comparisons. Suitable when the number of comparisons is low or strict error control is needed.
- **Scheffé's Test**: A flexible and conservative test that is particularly useful when testing complex comparisons beyond just pairwise comparisons.
- **Dunnett's Test**: Specifically used when comparing multiple treatment groups against a single control group.

### C. Performing Post-Hoc Tests in Python

#### Step 1: Conduct One-Way ANOVA

Let's assume we have a dataset with three or more groups, and we want to test if there's a significant difference among their means.

```python
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Load sample data
url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/PlantGrowth.csv'
df = pd.read_csv(url, usecols=[1, 2])

# Perform one-way ANOVA
model = ols('weight ~ C(group)', data=df).fit()
anova_results = anova_lm(model)
print("One-Way ANOVA Results:")
print(anova_results)
```

#### Step 2: Post-Hoc Tests Using Tukey's HSD

If the one-way ANOVA is significant, we can use Tukey's HSD to find which pairs of groups are significantly different.

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Perform Tukey's HSD test
tukey_result = pairwise_tukeyhsd(endog=df['weight'], groups=df['group'], alpha=0.05)
print("Tukey's HSD Test Results:")
print(tukey_result)
```

This output will show pairwise comparisons between each pair of groups, including:

- **meandiff**: The difference in means between the groups.
- **p-adj**: The adjusted p-value for each pairwise comparison.
- **reject**: A boolean indicating whether the null hypothesis (no difference) was rejected for each pair.

#### Step 3: Bonferroni Correction (Alternative to Tukey's HSD)

For a more conservative approach, use the Bonferroni correction, which divides the alpha level by the number of comparisons.

```python
from statsmodels.stats.multitest import multipletests
from itertools import combinations
from scipy.stats import ttest_ind

# Define groups
groups = df['group'].unique()

# Perform pairwise t-tests and apply Bonferroni correction
p_values = []
comparisons = []

for group1, group2 in combinations(groups, 2):
    data1 = df[df['group'] == group1]['weight']
    data2 = df[df['group'] == group2]['weight']
    stat, p_val = ttest_ind(data1, data2)
    p_values.append(p_val)
    comparisons.append(f"{group1} vs {group2}")

# Apply Bonferroni correction
_, p_values_corrected, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

# Display Bonferroni-corrected results
print("Bonferroni-Corrected Pairwise Comparisons:")
for comparison, p_val, p_val_corr in zip(comparisons, p_values, p_values_corrected):
    print(f"{comparison}: p-value = {p_val:.4f}, Bonferroni-corrected p-value = {p_val_corr:.4f}")
```

#### Step 4: Scheffé's Test (For Complex Comparisons)

Scheffé's test is suitable for testing non-pairwise comparisons or contrasts, but it is complex and less commonly used in basic pairwise comparisons. In `statsmodels`, Scheffé's test isn't directly available, but you can manually construct contrasts if needed, especially for more advanced comparisons.

## 2. scipy.stats.tukey_hsd

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd

def load_data():
    """
    Load and preprocess plant growth data for ANOVA and posthoc testing.
    """
    url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/PlantGrowth.csv'
    df = pd.read_csv(url, usecols=[1, 2])

    grouped_data = df.groupby('group')
    data_ctrl = grouped_data.get_group('ctrl').weight
    data_trt1 = grouped_data.get_group('trt1').weight
    data_trt2 = grouped_data.get_group('trt2').weight
    data = (data_ctrl, data_trt1, data_trt2)

    total_samples = data_ctrl.shape[0] + data_trt1.shape[0] + data_trt2.shape[0]
    num_groups = len(data)
    df1 = num_groups - 1
    df2 = total_samples - num_groups

    return df, data, df1, df2

def perform_anova(data_ctrl, data_trt1, data_trt2):
    """
    Perform one-way ANOVA on the data groups.
    """
    statistic, p_value = stats.f_oneway(data_ctrl, data_trt1, data_trt2)
    print("\nOne-way ANOVA Results:")
    print(f"F-statistic = {statistic:.4f}")
    print(f"P-value = {p_value:.4f}\n")
    return statistic, p_value

def perform_tukey_hsd(data_ctrl, data_trt1, data_trt2, confidence_level=0.95):
    """
    Perform Tukey's HSD posthoc test and display confidence intervals.
    """
    result = stats.tukey_hsd(data_ctrl, data_trt1, data_trt2)
    print(result)

    print(f"\nTukey's HSD Pairwise Group Comparisons ({confidence_level:.0%} Confidence Interval)")
    print("Comparison    Lower CI   Upper CI")
    confidence_interval = result.confidence_interval(confidence_level=confidence_level)
    for ((i, j), low) in np.ndenumerate(confidence_interval.low):
        if i < j:
            high = confidence_interval.high[i, j]
            print(f" ({i} - {j})   {low:>10.3f}   {high:>9.3f}")
    print()

# Load data, perform ANOVA, and conduct Tukey's HSD posthoc tests
df, (data_ctrl, data_trt1, data_trt2), df1, df2 = load_data()

# Conduct one-way ANOVA
statistic, p_value = perform_anova(data_ctrl, data_trt1, data_trt2)

# Conduct Tukey's HSD posthoc tests at default 95% confidence level
perform_tukey_hsd(data_ctrl, data_trt1, data_trt2)

# Conduct Tukey's HSD posthoc tests at a 99% confidence level
perform_tukey_hsd(data_ctrl, data_trt1, data_trt2, confidence_level=0.99)
```

### Output Interpretation

The table shows the results of Tukey's HSD (Honestly Significant Difference) pairwise comparisons for three groups (referred to as groups 0, 1, and 2), with a 95% confidence interval.

**Columns Explained:**

1. **Comparison**: The pair of groups being compared, denoted by their index numbers. For example, "(0 - 1)" represents the comparison between group 0 and group 1.

2. **Statistic**: The mean difference between the two groups for that particular comparison. A positive value means that the mean of the first group is higher than the mean of the second group, while a negative value indicates the opposite.

3. **p-value**: The probability value associated with the comparison. This value indicates whether the difference in means is statistically significant. A low p-value (typically less than 0.05) suggests a statistically significant difference between the groups.

4. **Lower CI**: The lower bound of the 95% confidence interval for the mean difference. If this interval does not include 0, the difference is considered statistically significant.

5. **Upper CI**: The upper bound of the 95% confidence interval for the mean difference.

**Interpretation of Each Comparison:**

- **(0 - 1) and (1 - 0)**: The mean difference between group 0 and group 1 is 0.371, with a p-value of 0.391. This high p-value indicates that the difference is not statistically significant, as the confidence interval (-0.320 to 1.062) includes 0.

- **(0 - 2) and (2 - 0)**: The mean difference between group 0 and group 2 is -0.494, with a p-value of 0.198. Again, this p-value is not significant, and the confidence interval (-1.185 to 0.197) includes 0, suggesting no significant difference.

- **(1 - 2) and (2 - 1)**: The mean difference between group 1 and group 2 is -0.865, with a p-value of 0.012. This p-value is below 0.05, indicating a statistically significant difference. The confidence interval (-1.556 to -0.174) does not include 0, which further confirms that the difference is significant.

**Summary:**

The Tukey's HSD test results suggest that there is a statistically significant difference between groups 1 and 2 at the 95% confidence level, as indicated by the low p-value and a confidence interval that does not contain 0. There are no significant differences between groups 0 and 1 or between groups 0 and 2.

### Positional Interpretation

1. **Group 1 (Left) and Group 2 (Right)**: Groups 1 and 2 show a statistically significant difference, as indicated by a low p-value (0.012) and a confidence interval that does not contain 0 (from -1.556 to -0.174). This suggests that these two groups are quite distinct in terms of their mean values.

2. **Group 0 (Middle) and Group 1 (Left)**: There is no statistically significant difference between Group 0 and Group 1. The p-value is 0.391, and the confidence interval (-0.320 to 1.062) includes 0.

3. **Group 0 (Middle) and Group 2 (Right)**: Similarly, there is no significant difference between Group 0 and Group 2, as the p-value is 0.198, and the confidence interval (-1.185 to 0.197) also includes 0.

**Interpretation:**

- **Group 0 (Middle) appears to be intermediate** between Groups 1 and 2, as it does not significantly differ from either.
- **Group 1 (Left) and Group 2 (Right) are significantly different from each other**, implying that they represent distinct levels or conditions compared to each other.
- Group 0 serves as an intermediate or transitional group, not significantly different from either Group 1 or Group 2.

### Does scipy.stats.tukey_hsd Run Pairwise t-Test For Each Pair?

1. **Purpose and Context of Use**: Tukey's HSD test is specifically a **post-hoc** test, meaning it is applied after an ANOVA has determined that there is a statistically significant difference among group means. It answers the question: "Which group pairs have significantly different means?" This is different from t-tests, which can be conducted independently and without reference to ANOVA.

2. **Studentized Range Distribution**: The reliance on the **Studentized range distribution** is a core distinguishing feature. This distribution accounts for the range of means in all groups simultaneously, rather than evaluating individual pairs in isolation (as t-tests do). It considers the number of groups being compared and controls the **family-wise error rate** (FWER) across all possible comparisons.

3. **Family-Wise Error Rate**: The adjustment made by Tukey's HSD ensures that the probability of making at least one Type I error across all comparisons does not exceed the pre-specified alpha level (e.g., 0.05). In contrast, conducting multiple t-tests increases the likelihood of Type I errors because the error rate compounds with the number of comparisons.

    For example, with $m$ groups, the number of pairwise comparisons is $\binom{m}{2} = \frac{m(m-1)}{2}$. If $m = 5$, there are 10 comparisons. Conducting these independently at a 5% significance level leads to an overall Type I error rate that can exceed 40%. Tukey's HSD prevents this inflation by incorporating multiple testing corrections.

4. **Critical Value and Interpretation**: Tukey's test computes a single critical difference value (HSD) that applies uniformly to all group comparisons. If the absolute difference in means between two groups exceeds the HSD value, the difference is deemed significant. This uniform threshold simplifies interpretation and avoids the variability introduced by running separate t-tests, each with its own critical value.

5. **Conservativeness**: By controlling for family-wise error, Tukey's HSD is generally more **conservative** than individual t-tests. While this reduces the risk of Type I errors, it may slightly increase the risk of Type II errors (failing to detect true differences). However, this trade-off is often acceptable in studies where controlling for false positives is a priority.

**Practical Implication**: Tukey's HSD is ideal for balanced designs (equal group sizes) but can still be applied to unbalanced designs with some adjustment. In such cases, other post-hoc tests like the Games-Howell test might be preferred for greater accuracy.

## 3. Post-Hoc Tests in Two-Way ANOVA

Post-hoc tests for a two-way ANOVA are essential to explore significant main and interaction effects in detail.

### A. Understanding When to Use Post-Hoc Tests in Two-Way ANOVA

In two-way ANOVA, post-hoc tests are generally used to:

1. **Investigate Main Effects**: If one or both main effects (e.g., factor A or factor B) are significant, post-hoc tests can identify which levels of the factor are significantly different from each other.
2. **Examine Interaction Effects**: If there is a significant interaction between factors A and B, post-hoc tests can help determine which specific combinations of factor levels show significant differences.

### B. Types of Post-Hoc Tests for Two-Way ANOVA

- **Tukey's Honest Significant Difference (HSD)**: Widely used for pairwise comparisons in ANOVA because it controls the family-wise error rate.
- **Bonferroni Correction**: A more conservative method that adjusts the significance level by dividing it by the number of comparisons.
- **Simple Effects Analysis**: If there's a significant interaction effect, simple effects analysis can be used to examine the effect of one factor at each level of the other factor.

### C. Performing Post-Hoc Tests in Python

#### Step 1: Conduct the Two-Way ANOVA

```python
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# Load the dataset
url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/ToothGrowth.csv'
df = pd.read_csv(url, usecols=[1, 2, 3])

# Define and fit the two-way ANOVA model
model = ols('len ~ C(supp) + C(dose) + C(supp):C(dose)', data=df).fit()
anova_results = anova_lm(model)
print(anova_results)
```

#### Step 2: Post-Hoc Tests for Main Effects

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Tukey's HSD for the main effect of dose
tukey_dose = pairwise_tukeyhsd(endog=df['len'], groups=df['dose'], alpha=0.05)
print("Post-Hoc Test for Dose:")
print(tukey_dose)

# Tukey's HSD for the main effect of supplement
tukey_supp = pairwise_tukeyhsd(endog=df['len'], groups=df['supp'], alpha=0.05)
print("Post-Hoc Test for Supplement:")
print(tukey_supp)
```

#### Step 3: Post-Hoc Tests for Interaction Effect

```python
# Create a combined factor for interaction analysis
df['supp_dose'] = df['supp'].astype(str) + "_" + df['dose'].astype(str)

# Perform Tukey's HSD on the interaction between supplement and dose
tukey_interaction = pairwise_tukeyhsd(endog=df['len'], groups=df['supp_dose'], alpha=0.05)
print("Post-Hoc Test for Interaction (Supplement x Dose):")
print(tukey_interaction)
```

#### Step 4: Simple Effects Analysis (Alternative to Interaction Post-Hoc)

If the interaction effect is significant, a **simple effects analysis** can provide a detailed breakdown by examining the effect of one factor at each level of the other factor.

```python
# Separate data by supplement type
oj_data = df[df['supp'] == 'OJ']
vc_data = df[df['supp'] == 'VC']

# Perform one-way ANOVA on dose within each supplement type
oj_model = ols('len ~ C(dose)', data=oj_data).fit()
vc_model = ols('len ~ C(dose)', data=vc_data).fit()

# Print ANOVA results for each supplement type
print("ANOVA for Dose within Supplement OJ:")
print(anova_lm(oj_model))

print("ANOVA for Dose within Supplement VC:")
print(anova_lm(vc_model))

# Tukey's HSD for dose within each supplement type
print("Tukey HSD for Dose within Supplement OJ:")
print(pairwise_tukeyhsd(endog=oj_data['len'], groups=oj_data['dose'], alpha=0.05))

print("Tukey HSD for Dose within Supplement VC:")
print(pairwise_tukeyhsd(endog=vc_data['len'], groups=vc_data['dose'], alpha=0.05))
```

### D. Summary of Steps

1. **Run Two-Way ANOVA**: Identify significant main and interaction effects.
2. **Post-Hoc for Main Effects**: Use Tukey's HSD or another pairwise test for each significant main effect.
3. **Post-Hoc for Interaction**: If the interaction effect is significant, use Tukey's HSD on combined factor levels or perform simple effects analysis.
