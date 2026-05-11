# Mean, Median, Mode

## Overview

Central tendency measures identify the center of a data distribution, summarizing a dataset with a single representative value. The three most commonly used measures are the **mean**, **median**, and **mode**. Each has distinct properties, strengths, and weaknesses that make it appropriate for different situations.

---

## 1. Mean

The mean is the arithmetic average of a set of numbers, calculated by summing all values and dividing by the count.

### Formulas

$$
\begin{array}{lllll}
\text{Population Mean} && \mu &=& \displaystyle\frac{\sum_{i=1}^N x_i}{N} \\[10pt]
\text{Sample Mean} && \bar{x} &=& \displaystyle\frac{\sum_{i=1}^n x_i}{n} \\[10pt]
\text{Expected Value} && \mathbb{E}[X] &=& \displaystyle\sum_i x_i \, \mathbb{P}(X = x_i) \quad \text{(discrete)} \\[6pt]
&&& =& \displaystyle\int_{-\infty}^{\infty} x \, f_X(x) \, dx \quad \text{(continuous)}
\end{array}
$$

### Example

For the dataset 70, 85, 90, 95, 100:

$$
\bar{x} = \frac{70 + 85 + 90 + 95 + 100}{5} = \frac{440}{5} = 88
$$

### Mean as Balancing Point

The mean is the value where the sum of deviations equals zero:

$$
\mu = \frac{\sum_{i=1}^N x_i}{N} \quad \Rightarrow \quad \sum_{i=1}^N (x_i - \mu) = 0
$$

This means the mean is the "center of gravity" of the data.

### Mean: Income Example

```python
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/loans_income.csv'
loans_data = pd.read_csv(url)

mean_income = loans_data['x'].mean()

fig, ax = plt.subplots(figsize=(12, 3))
ax.hist(loans_data['x'], bins=20, density=True, alpha=0.3,
        color='blue', edgecolor='black')
ax.plot([mean_income, mean_income], [0, 1.6e-5], '--c',
        alpha=0.7, label="Mean")
ax.legend()
ax.set_title("Histogram of Income Data with Mean Indicator")
ax.set_xlabel("Income")
ax.set_ylabel("Density")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
```

---

## 2. Median

The **median** is the middle value when data is arranged in order. It divides the dataset into two equal halves.

### How to Calculate

1. Sort the data in ascending order.
2. If $n$ is odd, the median is the middle value at position $(n+1)/2$.
3. If $n$ is even, the median is the average of the two middle values.

### Example

For the dataset 70, 85, 90, 95, 100 (odd count): Median = 90.

For the dataset 70, 85, 90, 95 (even count): Median = $(85 + 90)/2 = 87.5$.

### Median vs. Mean: Income Data

```python
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/loans_income.csv'
loans_data = pd.read_csv(url)

mean_income = loans_data['x'].mean()
median_income = loans_data['x'].median()

fig, ax = plt.subplots(figsize=(12, 3))
ax.hist(loans_data['x'], bins=20, density=True, alpha=0.3,
        color='blue', edgecolor='black')
ax.plot([mean_income, mean_income], [0, 1.6e-5], '--c',
        alpha=0.7, label="Mean")
ax.plot([median_income, median_income], [0, 1.6e-5], '--r',
        alpha=0.7, label="Median")
ax.legend()
ax.set_title("Histogram of Income Data with Mean and Median")
ax.set_xlabel("Income")
ax.set_ylabel("Density")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
```

### Median Is Robust Against Outliers

The median is far less affected by extreme values than the mean. This is demonstrated by adding outliers to income data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/loans_income.csv'
loans_data = pd.read_csv(url)
income_data = loans_data['x'].values

mean_income = income_data.mean()
median_income = np.median(income_data)

fig, (hist_ax, box_ax) = plt.subplots(1, 2, figsize=(12, 3))
plt.suptitle("Original Income Data", fontsize=20)

n, bin_edges, _ = hist_ax.hist(income_data, bins=20, density=True, alpha=0.5,
                                color='skyblue')
hist_ax.plot([mean_income, mean_income], [0, n.max()], '--b', label='Mean')
hist_ax.plot([median_income, median_income], [0, n.max()], '--r', label='Median')
hist_ax.legend()
hist_ax.set_title("Histogram")

box_ax.boxplot(income_data, vert=False, patch_artist=True)
box_ax.set_title("Boxplot")

for ax in (hist_ax, box_ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.show()

# Now add outliers
outliers = np.array([20_000_000] * 20)
data_with_outliers = np.concatenate((income_data, outliers))

mean_outliers = data_with_outliers.mean()
median_outliers = np.median(data_with_outliers)

fig, (hist_ax, box_ax) = plt.subplots(1, 2, figsize=(12, 3))
plt.suptitle("Income Data with Outliers", fontsize=20)

n, bin_edges, _ = hist_ax.hist(data_with_outliers, bins=bin_edges,
                                density=True, alpha=0.5, color='skyblue')
hist_ax.plot([mean_outliers, mean_outliers], [0, n.max()], '--b', label='Mean')
hist_ax.plot([median_outliers, median_outliers], [0, n.max()], '--r', label='Median')
hist_ax.legend()
hist_ax.set_title("Histogram")

box_ax.boxplot(data_with_outliers, vert=False, patch_artist=True)
box_ax.set_title("Boxplot")

for ax in (hist_ax, box_ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.show()
```

The mean shifts dramatically when outliers are introduced, while the median barely changes.

### Real-Life Examples Where Median Is Preferred

**Income Distribution:** A few extremely high earners skew the mean upward. Median household income provides a more accurate picture of what the "typical" person earns.

**Real Estate:** Median home prices better represent the housing market than mean prices, which can be inflated by a few luxury sales.

**Michael Jordan Case (NBA Salary):** Jordan's salary was so much higher than typical NBA players that it significantly inflated the mean. The median salary better reflects what most players earned.

**Public Policy:** Governments report median household income to assess financial well-being because it is not distorted by extreme wealth.

---

## 3. Trimmed Mean

The **trimmed mean** (or truncated mean) is the arithmetic mean calculated after removing a specified percentage of observations from both tails of the sorted distribution. This hybrid approach offers robustness against outliers while still utilizing most of the data.

### Definition

For a dataset with $n$ observations sorted as $x_{(1)} \le x_{(2)} \le \cdots \le x_{(n)}$, the $p$-trimmed mean removes $\lceil p \cdot n / 2 \rceil$ observations from each tail and averages the remaining values:

$$
\bar{x}_{p\%} = \frac{1}{n - 2\lceil p \cdot n / 2 \rceil} \sum_{i=\lceil p \cdot n / 2 \rceil + 1}^{n - \lceil p \cdot n / 2 \rceil} x_{(i)}
$$

### Example: Population Data

Using U.S. state population data, compare the mean, 10% trimmed mean, and median:

```python
import pandas as pd
from scipy.stats import trim_mean

# Load state data
state = pd.read_csv('state.csv')

# Regular mean (sensitive to outliers like California)
mean_pop = state['Population'].mean()
print(f"Mean Population: {mean_pop:,.0f}")

# 10% trimmed mean (removes 5% from each tail)
trimmed_mean_pop = trim_mean(state['Population'], 0.1)
print(f"10% Trimmed Mean: {trimmed_mean_pop:,.0f}")

# Median (completely robust)
median_pop = state['Population'].median()
print(f"Median Population: {median_pop:,.0f}")
```

**Output:**
```
Mean Population: 6,162,876
10% Trimmed Mean: 4,783,697
Median Population: 4,436,370
```

The trimmed mean occupies a middle ground: it's less influenced by extreme values (California's 37M population) than the mean, yet uses more data than the median alone. This makes it valuable when a moderate level of robustness is desired without completely ignoring the tails.

### When to Use Trimmed Mean

- **Moderate robustness:** You want outlier resistance but don't want to discard data entirely.
- **Academic traditions:** Some fields prefer trimmed means for hypothesis testing (e.g., psychology, education).
- **Olympic scoring:** Judges' scores are often averaged after trimming the highest and lowest.

---

## 4. Weighted Mean and Weighted Median

When observations have differing importance or frequency, the **weighted mean** and **weighted median** assign each value a weight reflecting its significance.

### Weighted Mean

The weighted mean is the sum of weighted values divided by the sum of weights:

$$
\bar{x}_w = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}
$$

where $w_i$ are the weights.

### Example: Murder Rate by State Population

When computing the national murder rate, states with larger populations should influence the average more heavily. Use the state's population as a weight:

```python
import pandas as pd
import numpy as np

state = pd.read_csv('state.csv')

# Unweighted mean murder rate
unweighted_mean = state['Murder.Rate'].mean()
print(f"Unweighted Mean Murder Rate: {unweighted_mean:.3f}")

# Weighted mean (weighted by population)
weighted_mean = np.average(state['Murder.Rate'], weights=state['Population'])
print(f"Weighted Mean Murder Rate: {weighted_mean:.3f}")
```

**Output:**
```
Unweighted Mean Murder Rate: 4.066
Weighted Mean Murder Rate: 4.446
```

The weighted mean is higher because highly populated states (CA, TX, FL, NY) tend to have higher murder rates than small states. The unweighted mean treats Montana (population 990K) and California (population 37M) as equal—a distortion that the weighted mean corrects.

### Weighted Median

The **weighted median** is the value where the cumulative weight reaches 50% of the total weight. Unlike the weighted mean, it requires a specialized function:

```python
import pandas as pd
import wquantiles

state = pd.read_csv('state.csv')

# Unweighted median
unweighted_median = state['Murder.Rate'].median()
print(f"Unweighted Median: {unweighted_median:.1f}")

# Weighted median (weighted by population)
weighted_median = wquantiles.median(state['Murder.Rate'],
                                    weights=state['Population'])
print(f"Weighted Median: {weighted_median:.1f}")
```

**Output:**
```
Unweighted Median: 4.0
Weighted Median: 4.4
```

### When to Use Weighted Statistics

**Financial Data:** Portfolio returns are weighted by asset values.

**Survey Data:** Responses are weighted to match population demographics.

**Aggregated Data:** When data represents groups (e.g., state-level statistics), weight by group size.

**Importance Weighting:** Some observations are more reliable or relevant than others.

---

## 5. Mode

The **mode** is the value that occurs most frequently in a dataset. A dataset may be unimodal (one mode), bimodal (two modes), multimodal (more than two modes), or have no mode if all values appear equally often.

### Computing Mode in Python

```python
import statistics

data = [4, 1, 2, 2, 3, 5]
mode = statistics.mode(data)
print(f"{mode = }")  # mode = 2
```

For datasets with multiple modes:

```python
import statistics

data = [4, 1, 2, 2, 3, 3, 5]
mode = statistics.mode(data)
print(f"{mode = }")  # Returns the first mode encountered

modes = statistics.multimode(data)
print(f"{modes = }")  # Returns all modes: [2, 3]
```

---

## 6. Comparing Mean, Median, Mode, and Other Measures

**Mean** is best for continuous, symmetrically distributed data without outliers. It uses all data points but is sensitive to extremes.

**Median** is preferred for skewed distributions or data with outliers. It represents the middle value and is robust to extreme observations.

**Mode** is most useful for categorical data or for identifying the most common value. It can be used with nominal data (e.g., most popular color).

### Relationship to Distribution Shape

- **Symmetric distribution:** Mean ≈ Median ≈ Mode
- **Right-skewed distribution:** Mode < Median < Mean
- **Left-skewed distribution:** Mean < Median < Mode

## Summary

Each measure of central tendency serves a distinct purpose. The mean provides a mathematical average but is vulnerable to outliers, the median offers a robust center that resists extreme values, and the mode identifies the most frequent observation. Choosing the appropriate measure depends on the data's distribution shape and the analytical question at hand.

## Exercises

**Exercise 1.**
A small company has five employees with annual salaries (in thousands of dollars): $35, 40, 42, 45, 250$.

**(a)** Compute the sample mean and the sample median.
**(b)** Replace the CEO's $\$250\text{k}$ salary with $\$500\text{k}$. Recompute the mean and median. Which changed more?
**(c)** Explain why the median is a *robust* measure of central tendency while the mean is not.

??? success "Solution to Exercise 1"
    (a) $\bar{x} = 412/5 = 82.4$, median $= 42$.

    (b) After the change: $\bar{x} = 662/5 = 132.4$, median still $= 42$. The mean rose by 50 (a 60.7% jump); the median is unchanged.

    (c) The median depends only on the *rank* of observations; changing the magnitude of any one observation (while preserving its rank) leaves it unchanged. The median has a breakdown point near 50% — about half the data must be corrupted before it can be moved arbitrarily. The mean's breakdown point is 0: a single extreme observation can move it arbitrarily far. This robustness/efficiency trade-off is fundamental.

---

**Exercise 2.**
Estimate the mean and variance from this grouped frequency table:

| Score range | Midpoint $m_i$ | Frequency $f_i$ |
|:---:|:---:|:---:|
| 50–59 | 54.5 | 3 |
| 60–69 | 64.5 | 5 |
| 70–79 | 74.5 | 10 |
| 80–89 | 84.5 | 8 |
| 90–99 | 94.5 | 4 |

Use $\bar{x} = \sum f_i m_i / \sum f_i$ and $s^2 = \sum f_i (m_i - \bar{x})^2 / (n - 1)$. Why are these *estimates* rather than exact values?

??? success "Solution to Exercise 2"
    $\sum f_i m_i = 163.5 + 322.5 + 745 + 676 + 378 = 2285$, so $\bar{x} = 2285/30 \approx 76.17$.

    Weighted squared deviations (showing the totals): $\sum f_i (m_i - \bar{x})^2 \approx 4016.7$, so $s^2 \approx 4016.7/29 \approx 138.5$, $s \approx 11.77$.

    **Estimates because** we substitute the midpoint $m_i$ for every observation in that bin — the true scores within "70–79" could lie anywhere in $[70, 79)$, not all at 74.5. The approximation is exact when the data are uniformly distributed within each bin and improves as bins narrow.

---

**Exercise 3.**
Prove that the sample mean $\bar{x} = (1/n)\sum x_i$ is the unique minimizer of $\sum (x_i - c)^2$ over $c \in \mathbb{R}$. What does the median minimize?

??? success "Solution to Exercise 3"
    **Mean:** differentiate $f(c) = \sum (x_i - c)^2$ with respect to $c$:

    $$
    f'(c) = -2 \sum (x_i - c) = 0 \implies c = \frac{1}{n}\sum x_i = \bar{x}
    $$

    Since $f''(c) = 2n > 0$, this is the unique minimum. The mean minimizes squared error.

    **Median:** the *median* minimizes the absolute-error loss $g(c) = \sum |x_i - c|$. The proof goes through subgradients: the derivative of $|x - c|$ with respect to $c$ is $-\mathrm{sign}(x - c)$, so $g'(c) = -(\#\{x_i > c\} - \#\{x_i < c\})$. Setting this to zero requires the number of $x_i$ above $c$ to equal the number below — which is the definition of the median.

    The two measures are the $L^2$ and $L^1$ projections of the data onto the constants, respectively. This characterization extends to quantile regression: the $\tau$-th quantile minimizes the asymmetric absolute loss $\sum \rho_\tau(x_i - c)$ where $\rho_\tau(u) = u(\tau - \mathbf{1}\{u < 0\})$.

---

**Exercise 4.**
A dataset has $n = 100$ observations with sample mean 50 and sample standard deviation 10. A single observation is corrupted from 60 to 1060. How does the sample mean change? How does the sample standard deviation change? Compare with what would happen to a 10%-trimmed mean.

??? success "Solution to Exercise 4"
    **Mean:** the new mean is $\bar{x}_{\text{new}} = 50 + (1060 - 60)/100 = 50 + 10 = 60$. The mean jumped by 10 — the entire population SD.

    **SD:** the contribution of the new observation to the sum of squared deviations becomes large: $\sum (x_i - \bar{x})^2$ increases roughly by $(1060 - 60)^2 \approx 10^6$. Recomputing precisely, $s^2_{\text{new}} \approx 10^4 + O(10^4 / n)$, so $s_{\text{new}} \approx 100$ — a tenfold inflation.

    **10%-trimmed mean:** the corrupted observation is in the top 10% of the data (likely the maximum), so it is trimmed. The trimmed mean is computed only from the middle 80% of the sorted data and is hardly affected: it might change by less than 0.1 units.

    Lesson: a single corrupted observation can severely distort both the mean and especially the standard deviation, while trimmed estimators are immune. This is why robust statistics matter in real data.

---

**Exercise 5.**
The mean, median, and mode are equal in a symmetric unimodal distribution; in a right-skewed distribution they are ordered Mode < Median < Mean. Prove the **mean is greater than the median** for any continuous distribution with finite mean whose density is positive on a half-line and decreasing on the upper tail (a typical right-skewed distribution like Exponential or Lognormal).

??? success "Solution to Exercise 5"
    Let $m$ denote the median (so $P(X \le m) = P(X \ge m) = 1/2$) and $\mu = \mathbb{E}[X]$.

    $$
    \mu - m = \mathbb{E}[X - m] = \int_{-\infty}^m (x - m) f(x)\,dx + \int_m^{\infty} (x - m) f(x)\,dx
    $$

    Substituting $u = m - x$ in the first integral and $v = x - m$ in the second:

    $$
    \mu - m = -\int_0^{\infty} u\, f(m - u)\,du + \int_0^{\infty} v\, f(m + v)\,dv = \int_0^{\infty} v\,[f(m + v) - f(m - v)]\,dv
    $$

    For a right-skewed distribution, the right tail $f(m + v)$ stays positive longer than the left tail $f(m - v)$ decays. Specifically, if $f$ decreases more slowly to the right of $m$ than to the left, $f(m + v) > f(m - v)$ for $v$ in the relevant tail range, making the integrand positive on average, so $\mu - m > 0$.

    A clean special case: Exponential$(\lambda)$ has $m = \ln(2)/\lambda \approx 0.693/\lambda$ while $\mu = 1/\lambda > 0.693/\lambda$. $\square$

---

**Exercise 6.**
The **mode** is the value of $x$ maximizing the density (or PMF). For a continuous unimodal symmetric distribution, the mean, median, and mode all coincide. But for a *mixture* of two Gaussians, even a "symmetric" mixture, the mode can disagree with the mean. Construct a bimodal symmetric mixture, identify all modes, and explain when the data analyst should report each.

??? success "Solution to Exercise 6"
    Consider $f(x) = 0.5 \cdot \phi(x; -3, 1) + 0.5 \cdot \phi(x; 3, 1)$ where $\phi(\cdot; \mu, \sigma)$ is the normal density. The mixture is symmetric about $x = 0$:

    - **Mean** $= \mu = 0$ (symmetry).
    - **Median** $= 0$ (symmetry).
    - **Modes** $= -3$ and $+3$ (local maxima of $f$).

    The mean and median land in a *valley* of the density — a point where the mixture is *least* probable. They are technically correct measures of center, but they convey misleading intuition for a bimodal distribution.

    **What to report:** when a histogram or density estimate reveals bimodality, the analyst should describe the modes and their relative weights, not just a single measure of center. Saying "the average household income is \$60,000" is technically correct but actively misleading if the underlying distribution is bimodal (working class vs. professional class). Visualization first; central-tendency summary second. Modern reporting practice often includes a **kernel density plot** alongside summary statistics for exactly this reason.
