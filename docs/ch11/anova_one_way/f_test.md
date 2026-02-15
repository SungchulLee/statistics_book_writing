# One-Way ANOVA: F-Test Procedure

## 1. Procedure for Conducting One-Way ANOVA

[eng|](https://www.youtube.com/watch?v=Lp2aV_4LF48&t=146s)

### Step 1: Formulate the Hypotheses

- **Null Hypothesis** ($H_0$): All group means are equal.

$$
H_0: \mu_1 = \mu_2 = \mu_3 = \dots = \mu_k
$$

where $\mu_1, \mu_2, \dots, \mu_k$ represent the population means for each of the $k$ groups.

- **Alternative Hypothesis** ($H_A$): At least one group mean is different.

### Step 2: Calculate the Overall Mean and Group Means

Compute the mean of all the dependent variable:

$$ \bar{y}_{\cdot\cdot} = \frac{1}{\sum_{i=1}^kn_i}\sum_{i=1}^k \sum_{j=1}^{n_i} y_{ij} $$

For each group, compute the mean of the dependent variable:

$$ \bar{y}_{i\cdot} = \frac{1}{n_i} \sum_{j=1}^{n_i} y_{ij} $$

### Step 3: Calculate Total Variation or Total Sum of Squares SST

Calculate the total variation $SST$, which is given by

$$
SST
=\displaystyle
\sum_{i=1}^{k}\sum_{j=1}^{n_i} \left( y_{ij} - \bar{y}_{\cdot\cdot} \right)^2
$$

### Step 4: Calculate Within-Group Variation SSW

Calculate the within-group variation $SSW$, which is given by

$$
SSW = \sum_{i=1}^{k} \sum_{j=1}^{n_i} \left( y_{ij} - \bar{y}_{i\cdot} \right)^2
$$

### Step 5: Calculate Between-Group Variation SSB

Calculate the between-group variation $SSB$, which is given by

$$
SSB
= \sum_{i=1}^{k} \sum_{j=1}^{n_i} \left( \bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot} \right)^2
= \sum_{i=1}^{k} n_i \left( \bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot} \right)^2
$$

### Step 6: Check the Partition of the Total Variation

The total sum of squares can be written as:

$$\begin{array}{lllllll}
SST
&=&\displaystyle
\sum_{i=1}^{k}\sum_{j=1}^{n_i} \left( y_{ij} - \bar{y}_{\cdot\cdot} \right)^2\\
&=&\displaystyle
\sum_{i=1}^{k} \sum_{j=1}^{n_i} \left[ \left( y_{ij} - \bar{y}_{i\cdot} \right) + \left( \bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot} \right) \right]^2\\
&=&\displaystyle
\sum_{i=1}^{k}\sum_{j=1}^{n_i} \left( y_{ij} - \bar{y}_{i\cdot} \right)^2 + \sum_{i=1}^{k} n_i \left( \bar{y}_{i\cdot} - \bar{y}_{\cdot\cdot} \right)^2
&=&
SSW + SSB
\end{array}
$$

### Step 7: Calculate the F-Statistic

$$\begin{array}{cccccccccc}
\text{Factor}&\text{df}&SS&MS&F&H_0&\text{Sampling Distribution of }F\text{ under $H_0$}\\
\hline
\text{Treatment}&k-1&SSB&\displaystyle MSB=\frac{SSB}{k-1}&\displaystyle F=\frac{MSB}{MSW}&\text{all }\beta_{i}=0&F\sim F_{k-1,N-k}\\
\text{Error}&N-k&SSW&\displaystyle MSW=\frac{SSW}{N-k}&\\
\hline
\text{Total}&N-1&SST&&\\
\end{array}$$

The test statistic for ANOVA is the F-statistic, calculated as the ratio of the mean square between groups (MSB) to the mean square within groups (MSW):

$$
F = \frac{\text{MSB}}{\text{MSW}} = \frac{\text{SSB}/(k-1)}{\text{SSW}/(N-k)}
$$

Where:

- $k$ is the number of groups
- $N$ is the total number of observations across all groups
- $\text{MSB} = \frac{\text{SSB}}{k - 1}$ is the **mean square between groups**
- $\text{MSW} = \frac{\text{SSW}}{N - k}$ is the **mean square within groups**

### Step 8: Determine the Critical Value or P-Value

The sampling distribution of $F$ under $H_0$ is

$$
F\sim F_{k-1,N-k}
$$

- Compare the calculated F-statistic with the critical value from the F-distribution table (based on $k-1$ and $N-k$ degrees of freedom), or
- Use the p-value approach. If the p-value is less than the significance level (e.g., $\alpha = 0.05$), reject the null hypothesis.

### Step 9: Make a Decision

$$ \text{statistic} > F_{\text{critical}} \quad\Rightarrow\quad\text{Choose $H_1$} $$

## 2. One-Way ANOVA Packages

### A. Scipy.Stats

#### scipy.stats.f_oneway

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def load_data():
    """
    Load and preprocess plant growth data from the given URL.

    Returns:
        df (pd.DataFrame): The full DataFrame of plant growth data.
        data (tuple): A tuple containing weights for each group ('ctrl', 'trt1', 'trt2').
        df1 (int): Degrees of freedom between groups.
        df2 (int): Degrees of freedom within groups.
    """
    url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/PlantGrowth.csv'
    df = pd.read_csv(url, usecols=[1, 2])

    group_data = df.groupby('group')
    data_ctrl = group_data.get_group('ctrl').weight
    data_trt1 = group_data.get_group('trt1').weight
    data_trt2 = group_data.get_group('trt2').weight
    data = (data_ctrl, data_trt1, data_trt2)

    total_samples = data_ctrl.shape[0] + data_trt1.shape[0] + data_trt2.shape[0]
    num_groups = len(data)
    df1 = num_groups - 1
    df2 = total_samples - num_groups

    return df, data, df1, df2

def perform_anova(data_ctrl, data_trt1, data_trt2):
    """
    Perform one-way ANOVA on the given data.

    Returns:
        statistic (float): F-statistic of the ANOVA test.
        p_value (float): P-value of the ANOVA test.
    """
    statistic, p_value = stats.f_oneway(data_ctrl, data_trt1, data_trt2)
    print(f"\nANOVA Results:\nF-Statistic = {statistic:.4f}\nP-Value = {p_value:.4f}")
    return statistic, p_value

def plot_data(data_ctrl, data_trt1, data_trt2, df1, df2, statistic, p_value):
    """
    Plot boxplot of weights for each group and F-distribution with critical region.
    """
    fig, (ax_box, ax_pdf) = plt.subplots(1, 2, figsize=(12, 4))

    ax_box.boxplot([data_ctrl, data_trt1, data_trt2], labels=['ctrl', 'trt1', 'trt2'])
    ax_box.set_ylim(3, 7)
    ax_box.set_xlabel('Group')
    ax_box.set_ylabel('Weight')

    x_vals = np.linspace(0, 6, 100)
    pdf_vals = stats.f(df1, df2).pdf(x_vals)
    ax_pdf.plot(x_vals, pdf_vals, label='F-distribution PDF')
    ax_pdf.fill_between(x_vals[x_vals >= statistic], pdf_vals[x_vals >= statistic], color='red', alpha=0.3)

    ax_pdf.spines[['top','right']].set_visible(False)
    ax_pdf.spines[['bottom','left']].set_position("zero")
    ax_pdf.set_title("F-distribution and Critical Region")
    ax_pdf.legend()

    ax_pdf.annotate(f'P-Value = {p_value:.2%}', xy=(5.0, 0.1), xytext=(5.0, 0.8),
                    arrowprops=dict(color='k', width=0.2, headwidth=8), fontsize=12)

    plt.tight_layout()
    plt.show()

# Load data, perform ANOVA, and plot results
_, (data_ctrl, data_trt1, data_trt2), df1, df2 = load_data()
statistic, p_value = perform_anova(data_ctrl, data_trt1, data_trt2)
plot_data(data_ctrl, data_trt1, data_trt2, df1, df2, statistic, p_value)
```

### B. Statsmodels

#### statsmodels.formula.api.ols and statsmodels.stats.anova.anova_lm

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def load_data():
    """
    Load and preprocess plant growth data for ANOVA.
    """
    url = 'https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/PlantGrowth.csv'
    df = pd.read_csv(url, usecols=[1, 2])

    group_data = df.groupby('group')
    data_ctrl = group_data.get_group('ctrl').weight
    data_trt1 = group_data.get_group('trt1').weight
    data_trt2 = group_data.get_group('trt2').weight
    data = (data_ctrl, data_trt1, data_trt2)

    total_samples = data_ctrl.shape[0] + data_trt1.shape[0] + data_trt2.shape[0]
    num_groups = len(data)
    df1 = num_groups - 1
    df2 = total_samples - num_groups

    return df, data, df1, df2

def perform_anova(df):
    """
    Perform one-way ANOVA using statsmodels.
    """
    model = ols('weight ~ C(group)', data=df).fit()
    anova_results = anova_lm(model)
    statistic = anova_results['F'].iloc[0]
    p_value = anova_results['PR(>F)'].iloc[0]

    print("\nANOVA Results:\n", anova_results)
    return statistic, p_value

def plot_data(data_ctrl, data_trt1, data_trt2, df1, df2, statistic, p_value):
    """
    Plot boxplot of weights for each group and F-distribution with critical region.
    """
    fig, (ax_box, ax_pdf) = plt.subplots(1, 2, figsize=(12, 4))

    ax_box.boxplot([data_ctrl, data_trt1, data_trt2], labels=['ctrl', 'trt1', 'trt2'])
    ax_box.set_ylim(3, 7)
    ax_box.set_xlabel('Group')
    ax_box.set_ylabel('Weight')

    x_vals = np.linspace(0, 6, 100)
    pdf_vals = stats.f(df1, df2).pdf(x_vals)
    ax_pdf.plot(x_vals, pdf_vals, label='F-distribution PDF')
    ax_pdf.fill_between(x_vals[x_vals >= statistic], pdf_vals[x_vals >= statistic], color='red', alpha=0.3)

    ax_pdf.spines['top'].set_visible(False)
    ax_pdf.spines['right'].set_visible(False)
    ax_pdf.set_title("F-distribution and Critical Region")
    ax_pdf.legend()

    ax_pdf.annotate(f'P-Value = {p_value:.2%}', xy=(5.0, 0.1), xytext=(5.0, 0.8),
                    arrowprops=dict(color='k', width=0.2, headwidth=8), fontsize=12)

    plt.tight_layout()
    plt.show()

# Load data, perform ANOVA, and plot results
df, (data_ctrl, data_trt1, data_trt2), df1, df2 = load_data()
statistic, p_value = perform_anova(df)
plot_data(data_ctrl, data_trt1, data_trt2, df1, df2, statistic, p_value)
```

## 3. Example: Reaction Times After Consuming Different Types Of Drinks

### Question

Suppose we have three different study groups measuring reaction times (in milliseconds) after consuming different types of drinks: **Water**, **Energy Drink**, and **Coffee**. We want to determine if there is a statistically significant difference in reaction times between these groups.

The data for each group is as follows:

- **Water Group**: [19, 18, 17, 18, 20]
- **Energy Drink Group**: [20, 22, 19, 21, 20]
- **Coffee Group**: [18, 17, 16, 19, 20]

We want to perform a One-Way ANOVA test to determine if there is a difference in the average reaction times between the groups.

### Step 1: Formulate the Hypotheses

**Null Hypothesis ($H_0$)**: The mean reaction times for the three groups are equal.

**Alternative Hypothesis ($H_a$)**: At least one group has a different mean reaction time.

### Step 2: Calculate the Overall Mean and Group Means

1. **Overall Mean ($\bar{X}$)**:

$$ \bar{X} = \frac{19 + 18 + 17 + 18 + 20 + 20 + 22 + 19 + 21 + 20 + 18 + 17 + 16 + 19 + 20}{15} = 19.0 $$

2. **Group Means**:
   - **Water Group**: $(19 + 18 + 17 + 18 + 20) / 5 = 18.4$
   - **Energy Drink Group**: $(20 + 22 + 19 + 21 + 20) / 5 = 20.4$
   - **Coffee Group**: $(18 + 17 + 16 + 19 + 20) / 5 = 18.0$

### Step 3: Calculate Total Variation or Total Sum of Squares SST

The Total Sum of Squares (SST) measures the total variability in the data relative to the overall mean.

$$ SST = \sum_{i=1}^{k}\sum_{j=1}^{n_i} (X_{ij} - \bar{X}_{\cdot\cdot})^2 $$

For each value in the dataset:

**Water Group**: $(19 - 19)^2 = 0$, $(18 - 19)^2 = 1$, $(17 - 19)^2 = 4$, $(18 - 19)^2 = 1$, $(20 - 19)^2 = 1$

**Energy Drink Group**: $(20 - 19)^2 = 1$, $(22 - 19)^2 = 9$, $(19 - 19)^2 = 0$, $(21 - 19)^2 = 4$, $(20 - 19)^2 = 1$

**Coffee Group**: $(18 - 19)^2 = 1$, $(17 - 19)^2 = 4$, $(16 - 19)^2 = 9$, $(19 - 19)^2 = 0$, $(20 - 19)^2 = 1$

**Summing all these values**:

$$
SST = 0 + 1 + 4 + 1 + 1 + 1 + 9 + 0 + 4 + 1 + 1 + 4 + 9 + 0 + 1 = 37
$$

### Step 4: Calculate Within-Group Variation SSW

The Within-Group Sum of Squares (SSW) measures the variability within each group.

$$ SSW = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_{i\cdot})^2 $$

**Water Group**:

$$ SSW_{\text{Water}} = 0.36 + 0.16 + 1.96 + 0.16 + 2.56 = 5.20 $$

**Energy Drink Group**:

$$ SSW_{\text{Energy Drink}} = 0.16 + 2.56 + 1.96 + 0.36 + 0.16 = 5.20 $$

**Coffee Group**:

$$ SSW_{\text{Coffee}} = 0 + 1 + 4 + 1 + 4 = 10.00 $$

**Summing these values**:

$$
SSW = 5.20 + 5.20 + 10.00 = 20.40
$$

### Step 5: Calculate Between-Group Variation SSB

The Between-Group Sum of Squares (SSB) measures the variability between the group means and the overall mean.

$$ SSB = \sum_{i=1}^{k} n_i (\bar{X}_{i\cdot} - \bar{X}_{\cdot\cdot})^2 $$

**Water Group**: $SSB_{\text{Water}} = 5 \times (18.4 - 19)^2 = 5 \times 0.36 = 1.8$

**Energy Drink Group**: $SSB_{\text{Energy Drink}} = 5 \times (20.4 - 19)^2 = 5 \times 1.96 = 9.8$

**Coffee Group**: $SSB_{\text{Coffee}} = 5 \times (18.0 - 19)^2 = 5 \times 1.0 = 5.0$

**Summing these values**:

$$
SSB = 1.8 + 9.8 + 5.0 = 16.6
$$

### Step 6: Check the Partition of the Total Variation

Now, let's verify if:

$$SST = SSB + SSW$$

- **SST** = 37
- **SSB + SSW** = 16.6 + 20.40 = 37

The values match, confirming our calculations are consistent.

### Step 7: Calculate the F-Statistic

The **F-statistic** is calculated as:

$$F = \frac{MSB}{MSW}$$

Where:

- **Mean Square Between (MSB)**: $MSB = \frac{SSB}{k - 1} = \frac{16.6}{3 - 1} = 8.3$
- **Mean Square Within (MSW)**: $MSW = \frac{SSW}{N - k} = \frac{20.40}{15 - 3} = 1.70$

Thus:

$$F = \frac{8.3}{1.70} = 4.88$$

### Step 8: Determine the Critical Value or P-Value

To determine the p-value, we compare the F-statistic to the critical value from the F-distribution with $df_1 = 2$ (between groups) and $df_2 = 12$ (within groups).

The **p-value for $F = 4.88$** with these degrees of freedom is approximately **0.03**.

### Step 9: Make a Decision

Since the p-value is **0.03**, which is **less than the typical significance level of $\alpha = 0.05$**, we **reject the null hypothesis**. This indicates that there is a significant difference between the means of the three groups.

### Step 10: Post-Hoc Tests

If the result were closer to significance, you might want to increase the sample size or use a different significance level to further test the hypothesis. Alternatively, conducting **post-hoc** tests could help in understanding more subtle differences between the groups.

```python
import scipy.stats as stats

# Data for the three groups
water_group = [19, 18, 17, 18, 20]
energy_drink_group = [20, 22, 19, 21, 20]
coffee_group = [18, 17, 16, 19, 20]

# Perform One-Way ANOVA
f_statistic, p_value = stats.f_oneway(water_group, energy_drink_group, coffee_group)

# Print results
print(f"F-statistic: {f_statistic:.2f}")
print(f"P-value: {p_value:.4f}")
```

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Given F-statistic and degrees of freedom
df_between = 3 - 1
df_within = 15 - 3
f_statistic = 4.86

# Calculate p-value
p_value = stats.f.sf(f_statistic, df_between, df_within)
print(f"{f_statistic = :.04f}")
print(f"{p_value = :.04f}")

# Create the figure and axis using OOP style
fig, ax = plt.subplots(figsize=(12, 4))

# Generate F-distribution values for the left of the statistic
x = np.linspace(0, f_statistic, 500)
y = stats.f.pdf(x, df_between, df_within)
ax.plot(x, y, color='blue', linewidth=3)

# Fill the left region under the curve
x = np.concatenate([[0], x, [f_statistic], [0]])
y = np.concatenate([[0], y, [0], [0]])
ax.fill(x, y, color='blue', alpha=0.1)

# Generate F-distribution values for the right of the statistic
x = np.linspace(f_statistic, 20, 500)
y = stats.f.pdf(x, df_between, df_within)
ax.plot(x, y, color='red', linewidth=3)

# Fill the right region under the curve (p-value region)
x = np.concatenate([[f_statistic], x, [20], [f_statistic]])
y = np.concatenate([[0], y, [0], [0]])
ax.fill(x, y, color='red', alpha=0.1)

# Annotate the p-value region
xy = ((f_statistic + 15.0) / 2, 0.01)
xytext = (f_statistic + 3, 0.5)
arrowprops = dict(color='black', width=0.2, headwidth=8)
ax.annotate(f'{p_value = :.02%}', xy, xytext=xytext, fontsize=15, arrowprops=arrowprops)

# Customize plot appearance
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')

ax.set_xlabel('F-value')
ax.set_ylabel('Probability Density')
ax.set_title('F-distribution with Highlighted p-value Region')

plt.show()
```

---

## 4. Permutation-Based One-Way ANOVA

Permutation tests for ANOVA provide a non-parametric alternative that requires no distributional assumptions. Two approaches are commonly used: testing the variance of group means or computing the F-statistic directly.

### Approach 1: Permutation Test Using Variance of Group Means

This approach tests whether the variance among group means is unusually large under the null hypothesis.

#### Algorithm

1. **Calculate observed variance of group means**:
   $$\text{Var}(\bar{y}_{1\cdot}, \bar{y}_{2\cdot}, \ldots, \bar{y}_{k\cdot})$$

2. **Permute data B times**:
   - Pool all observations
   - Randomly reassign observations to groups (maintaining group sizes)
   - Calculate variance of group means for each permutation

3. **Calculate p-value**:
   $$p\text{-value} = \frac{\#\{\text{Perm Var} \geq \text{Obs Var}\}}{B}$$

#### Example: Four Web Pages

Suppose we test session times for four different web pages:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Session time data for four pages
four_sessions = pd.DataFrame({
    'Time': [164, 178, 175, 155, 172, 182, 180, 179, 165, 166,
             172, 161, 171, 173, 158, 161, 179, 159, 167, 162],
    'Page': ['Page 1']*5 + ['Page 2']*5 + ['Page 3']*5 + ['Page 4']*5
})

# Observed variance of group means
obs_variance = four_sessions.groupby('Page')['Time'].mean().var()
print(f"Observed variance of means: {obs_variance:.2f}")

# Permutation test
def perm_test_anova(df, group_col='Page', value_col='Time', n_perms=3000):
    """
    Permutation test for ANOVA using variance of group means.

    Parameters:
    -----------
    df : DataFrame
        Data with groups and values
    group_col : str
        Name of column with group labels
    value_col : str
        Name of column with values
    n_perms : int
        Number of permutations

    Returns:
    --------
    p_value : float
        Permutation test p-value
    perm_vars : array
        Permuted variances
    """
    groups = df[group_col].unique()
    group_sizes = {g: (df[group_col] == g).sum() for g in groups}

    obs_var = df.groupby(group_col)[value_col].mean().var()

    perm_vars = np.zeros(n_perms)
    for i in range(n_perms):
        # Shuffle values and reassign to groups
        shuffled_values = np.random.permutation(df[value_col].values)
        perm_df = df.copy()
        perm_df[value_col] = shuffled_values

        # Calculate variance of group means
        perm_vars[i] = perm_df.groupby(group_col)[value_col].mean().var()

    p_value = np.mean(perm_vars >= obs_var)
    return p_value, perm_vars, obs_var

# Run permutation test
np.random.seed(42)
p_val, perm_vars, obs_var = perm_test_anova(four_sessions)

print(f"Permutation test p-value: {p_val:.4f}")
print(f"Conclusion: {'Reject H₀' if p_val < 0.05 else 'Fail to reject H₀'}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(perm_vars, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
ax.axvline(obs_var, color='red', linewidth=2, label=f'Observed = {obs_var:.2f}')
ax.set_xlabel('Variance of Group Means')
ax.set_ylabel('Frequency')
ax.set_title(f'Permutation Distribution of Group Mean Variance (p={p_val:.3f})')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()
```

### Approach 2: Permutation Test Using F-Statistic

While the variance-based approach is intuitive, we can also use the F-statistic as our test statistic for a more direct comparison with the parametric ANOVA.

#### Algorithm

1. **Calculate observed F-statistic**:
   $$F_{\text{obs}} = \frac{\text{MSB}}{\text{MSW}}$$

2. **Permute data B times**:
   - Pool all observations
   - Randomly reassign to groups
   - Calculate F-statistic for each permutation

3. **Calculate p-value**:
   $$p\text{-value} = \frac{\#\{F_b \geq F_{\text{obs}}\}}{B}$$

#### Example: F-Statistic Based Test

```python
from scipy import stats

def perm_test_anova_f(df, group_col='Page', value_col='Time', n_perms=3000):
    """
    Permutation test for ANOVA using F-statistic.
    """
    groups = df[group_col].unique()
    group_sizes = {g: (df[group_col] == g).sum() for g in groups}

    # Observed F-statistic
    model = smf.ols(f'{value_col} ~ C({group_col})', data=df).fit()
    anova_table = sm.stats.anova_lm(model)
    f_obs = anova_table['F'].iloc[0]

    perm_f_stats = np.zeros(n_perms)
    for i in range(n_perms):
        # Shuffle values
        shuffled_values = np.random.permutation(df[value_col].values)
        perm_df = df.copy()
        perm_df[value_col] = shuffled_values

        # Calculate F-statistic
        perm_model = smf.ols(f'{value_col} ~ C({group_col})', data=perm_df).fit()
        perm_anova = sm.stats.anova_lm(perm_model)
        perm_f_stats[i] = perm_anova['F'].iloc[0]

    p_value = np.mean(perm_f_stats >= f_obs)
    return p_value, perm_f_stats, f_obs

# Run F-based permutation test
import statsmodels.formula.api as smf
import statsmodels.api as sm

p_val_f, perm_f, obs_f = perm_test_anova_f(four_sessions)
print(f"\nF-statistic based permutation test:")
print(f"Observed F: {obs_f:.4f}")
print(f"p-value: {p_val_f:.4f}")
```

### Comparison: Permutation vs. Parametric ANOVA

Both approaches give similar results when parametric assumptions hold:

```python
# Parametric ANOVA
f_stat, p_param = stats.f_oneway(
    four_sessions[four_sessions.Page == 'Page 1']['Time'],
    four_sessions[four_sessions.Page == 'Page 2']['Time'],
    four_sessions[four_sessions.Page == 'Page 3']['Time'],
    four_sessions[four_sessions.Page == 'Page 4']['Time']
)

print(f"\nParametric ANOVA:")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value: {p_param:.4f}")

print(f"\nPermutation ANOVA (variance-based):")
print(f"p-value: {p_val:.4f}")
```

### Advantages of Permutation ANOVA

1. **No distributional assumptions**: Works with any data distribution
2. **Naturally handles unequal variances**: No need for Levene's test
3. **Exact Type I error control**: p-value is exact (not approximate)
4. **Intuitive interpretation**: Results reflect actual randomization

### When to Use Permutation ANOVA

- **Small samples**: n < 30 per group
- **Non-normal data**: Verified with Q-Q plots or Shapiro-Wilk test
- **Unequal variances**: When Levene's test rejects homogeneity
- **Robustness check**: Compare results to parametric ANOVA
