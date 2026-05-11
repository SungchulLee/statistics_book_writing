# Group Comparison Examples

## Overview

Comparing distributions across groups is a core task in descriptive statistics. When data is partitioned by a categorical variable—airline, zip code, credit grade—the natural questions are: do the groups differ in center, spread, or shape?

Two standard tools for visual group comparison are **boxplots** and **violin plots**. Boxplots provide a compact summary of medians, quartiles, and outliers; violin plots show the full distributional shape, revealing features like bimodality that boxplots hide.

---

## Example 1: Airline Carrier Delays

### Setup

Flight delays matter for passenger logistics and airline operations. We compare the distribution of daily delay percentages across four carriers. Each carrier's delays are drawn from a Gamma distribution with different shape and scale parameters, reflecting different operational characteristics.

### Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
airlines = ['American', 'Delta', 'Southwest', 'United']
n_obs = 100

data_list = []
params = {
    'American':  (2, 3),      # shape, scale — more frequent delays
    'Delta':     (1.5, 2.5),  # moderate
    'Southwest': (1.2, 2),    # fewer delays
    'United':    (1.8, 3.2),  # high variability
}
for airline in airlines:
    shape, scale = params[airline]
    delays = np.random.gamma(shape=shape, scale=scale, size=n_obs)
    for delay in delays:
        data_list.append({'airline': airline, 'pct_carrier_delay': delay})

airline_stats = pd.DataFrame(data_list)
```

### Boxplot View

```python
fig, ax = plt.subplots(figsize=(8, 5))
airline_stats.boxplot(by='airline', column='pct_carrier_delay', ax=ax)
ax.set_xlabel('Airline')
ax.set_ylabel('Daily % of Delayed Flights')
ax.set_title('Airline Delay Comparison: Boxplots')
plt.suptitle('')
plt.tight_layout()
plt.show()
```

Reading the boxplot:

- The **box** spans $Q_1$ to $Q_3$ (middle 50% of delays).
- The **line** inside the box is the median.
- **Whiskers** extend to the most extreme observations within $1.5 \times \text{IQR}$.
- Points beyond the whiskers are **outliers**.

### Violin Plot View

```python
fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(data=airline_stats, x='airline', y='pct_carrier_delay',
               ax=ax, inner='quartile', color='lightblue')
ax.set_xlabel('Airline')
ax.set_ylabel('Daily % of Delayed Flights')
ax.set_title('Airline Delay Comparison: Violin Plots')
plt.tight_layout()
plt.show()
```

Reading the violin plot:

- **Wide sections** indicate many observations at that delay level.
- **Narrow sections** indicate few observations.
- A **bimodal** shape (two humps) suggests two typical delay scenarios.
- **Skewed** shapes indicate asymmetric delay distributions.

### Interpretation

Comparing across airlines reveals differences in both location and spread. An airline with a higher median and wider box has systematically worse delays. Violin plots add nuance: two airlines may have similar medians but very different shapes, which a boxplot alone would not show.

---

## Example 2: Housing Values by Zip Code

### Setup

Real estate investors want to understand how neighborhood affects home values. We compare simulated tax-assessed housing values across four King County, WA zip codes.

### Code

```python
np.random.seed(123)
zip_codes = [98188, 98105, 98108, 98126]
n_homes = 150

data_list = []
for zip_code in zip_codes:
    base_price = 300_000 if zip_code in [98105, 98108] else 450_000
    prices = np.random.normal(base_price, 100_000, n_homes)
    prices = np.clip(prices, 50_000, 2_000_000)
    for price in prices:
        data_list.append({'ZipCode': str(zip_code), 'TaxAssessedValue': price})

housing = pd.DataFrame(data_list)

fig, ax = plt.subplots(figsize=(8, 5))
housing.boxplot(by='ZipCode', column='TaxAssessedValue', ax=ax)
ax.set_xlabel('Zip Code')
ax.set_ylabel('Tax Assessed Value (\$)')
ax.set_title('Housing Values Across Neighborhoods')
plt.suptitle('')
plt.tight_layout()
plt.show()
```

### Interpretation

Zip codes with higher base prices show boxes shifted upward. Similar box widths indicate comparable variability. Outliers in expensive neighborhoods may represent luxury properties that inflate the mean while leaving the median relatively stable.

---

## Example 3: Income by Loan Credit Grade

### Setup

Lenders assess credit risk by examining income distributions across loan grades (A = best, G = worst). Lower grades tend to have lower and more dispersed incomes.

### Code

```python
np.random.seed(456)
grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
n_per_grade = 100

data_list = []
for grade in grades:
    grade_idx = ord(grade) - ord('A')
    base_income = 80_000 - grade_idx * 8_000
    income_std = 15_000 + grade_idx * 5_000
    incomes = np.random.normal(base_income, income_std, n_per_grade)
    incomes = np.clip(incomes, 10_000, 200_000)
    for income in incomes:
        data_list.append({'grade': grade, 'income': income})

loans = pd.DataFrame(data_list)

fig, ax = plt.subplots(figsize=(10, 5))
sns.violinplot(data=loans, x='grade', y='income', ax=ax, color='lightgreen')
ax.set_xlabel('Loan Grade (A=best, G=worst)')
ax.set_ylabel('Annual Income (\$)')
ax.set_title('Income Distribution by Credit Grade')
plt.tight_layout()
plt.show()
```

### Interpretation

- **Grade A** borrowers have higher, more concentrated incomes—lower default risk.
- **Grade G** borrowers have lower, more dispersed incomes—higher default risk.
- The violin shapes widen progressively from A to G, showing increasing income uncertainty.

---

## Boxplots vs Violin Plots

| Feature | Boxplot | Violin Plot |
|---|---|---|
| Shows quartiles | Yes | Yes (with `inner='quartile'`) |
| Shows outliers | Yes (individual points) | No (smoothed away) |
| Shows full shape | No | Yes |
| Reveals bimodality | No | Yes |
| Compactness | High | Moderate |

!!! tip "Best Practice"
    Use boxplots for a quick summary and violin plots when distributional shape matters. When in doubt, show both side by side.

---

## Exercises

**Exercise 1.**
Given a boxplot where the median line is closer to $Q_1$ than to $Q_3$, and the upper whisker is longer than the lower whisker, describe the likely shape of the distribution.

??? success "Solution to Exercise 1"
    The distribution is **right-skewed**. The median sitting closer to $Q_1$ means the upper half of the box is wider, indicating the data stretches further to the right. The longer upper whisker confirms that extreme values are more common on the high side.

---

**Exercise 2.**
Consider two groups with identical medians and identical IQRs, but Group A's violin plot shows a single peak while Group B's shows two peaks. How do the distributions differ, and which summary statistics would fail to capture this difference?

??? success "Solution to Exercise 2"
    Group A is **unimodal** and Group B is **bimodal**. The mean, median, variance, and IQR would all be similar or identical between the groups. Standard summary statistics based on center and spread do not capture modality. Only the full shape—visible in a violin plot, histogram, or density estimate—reveals the difference.

---

**Exercise 3.**
Using the airline delay data above, suppose American has median delay 6.0% with IQR 4.5%, and Southwest has median delay 2.4% with IQR 2.0%. A manager claims "American's delays are exactly 2.5 times worse." Critique this claim.

??? success "Solution to Exercise 3"
    The ratio of medians is $6.0 / 2.4 = 2.5$, so the claim holds for the median. However, "2.5 times worse" oversimplifies: the IQR ratio is $4.5 / 2.0 = 2.25$, so the spread differs by a different factor. Moreover, the shapes may differ (American's Gamma(2, 3) is more symmetric than Southwest's Gamma(1.2, 2)). A single multiplier cannot capture differences in center, spread, and shape simultaneously.

---

**Exercise 4.**
Explain why a boxplot can show outliers that a violin plot does not. Under what circumstances might a violin plot be misleading about the tails of a distribution?

??? success "Solution to Exercise 4"
    A boxplot marks individual observations beyond $1.5 \times \text{IQR}$ as discrete points. A violin plot uses kernel density estimation, which smooths the data. With few observations in the tails, the KDE estimate is very low and the violin tapers to a thin line, making extreme values invisible. This is misleading when:

    - The sample size is small and individual outliers matter.
    - The distribution has heavy tails (e.g., Cauchy) where extreme observations carry important information.
    - The bandwidth of the KDE is too large, over-smoothing the tails.

---

**Exercise 5.**
Prove that for any dataset, at least 50% of the observations lie inside the box of a boxplot (i.e., between $Q_1$ and $Q_3$).

??? success "Solution to Exercise 5"
    By definition, $Q_1$ is the 25th percentile and $Q_3$ is the 75th percentile. The proportion of data between them is

    $$
    P(Q_1 \le X \le Q_3) = 0.75 - 0.25 = 0.50
    $$

    so at least 50% of observations fall inside the box. This holds for any distribution, regardless of shape, because it follows directly from the definition of quartiles. $\square$

---

**Exercise 6.**
A **bee swarm plot** (or strip plot with jitter) shows every observation along a horizontal axis, with vertical jitter to avoid overplotting. When is a bee swarm preferable to a box plot or violin plot?

??? success "Solution to Exercise 6"
    Bee swarm plots show *every individual observation*, which is valuable when:

    - **Sample size is small** ($n < 50$): box plots and violins can mislead because their summary statistics are noisy with few observations. Showing each point lets the viewer assess the data directly.
    - **The exact number of observations matters**: in clinical trials or experimental work, the count per group is part of the story. Bee swarms make it visible at a glance.
    - **Outliers should not be visually marginalized**: a box plot's "outliers" are isolated dots that suggest "ignore these." A bee swarm treats every observation equally.
    - **Multimodality or clustering within a group**: visible as gaps or clumps in the bee swarm; sometimes obscured in a smoothed violin.

    Bee swarms are *not* preferable when:

    - Sample sizes are very large ($n > 1000$): jitter cannot prevent overplotting and the plot becomes a smear.
    - Comparing many groups: bee swarms take more horizontal space than box plots.
    - Outlier identification is the goal: box plots make outliers more visually prominent.

    A common composite plot uses bee swarm + overlaid box plot, giving both the individual data and a summary. Seaborn's `sns.stripplot(..., dodge=True) + sns.boxplot(...)` produces this.

---

**Exercise 7.**
The **violin plot's split mode** (half-violin per group) makes paired group comparison visually crisp. Explain when split-violins **mislead** the viewer and propose an alternative for the affected cases.

??? success "Solution to Exercise 7"
    Split-violins mislead when:

    - **Sample sizes are very unequal between groups**: each half is normalized to occupy the same visual width, hiding the imbalance. A 10-observation group appears as visually weighty as a 1000-observation group.
    - **Distributions have very different shapes**: the eye reads opposite halves as symmetric, which can suggest mirror-image distributions even when the densities are unrelated.
    - **The reference axis is arbitrary**: which group goes on which side affects how the viewer interprets "high" vs. "low." If conventional ordering does not exist (e.g., "Treatment A" vs. "Treatment B"), the split-violin can prime spurious interpretations.

    **Alternative:** use side-by-side full violins with sample sizes annotated below each ($n = 10$ vs. $n = 1000$ as labels). This sacrifices the visual elegance of split-violins for honesty about the data.

    For paired data (the same subject measured twice), neither split nor side-by-side violins capture the pairing. Use a **paired plot**: connect each subject's two measurements with a line. This shows individual changes and the magnitude of within-subject variation, which is the actual quantity of interest in paired designs.
