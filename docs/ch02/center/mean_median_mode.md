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

## 3. Mode

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

## 4. Comparing Mean, Median, and Mode

**Mean** is best for continuous, symmetrically distributed data without outliers. It uses all data points but is sensitive to extremes.

**Median** is preferred for skewed distributions or data with outliers. It represents the middle value and is robust to extreme observations.

**Mode** is most useful for categorical data or for identifying the most common value. It can be used with nominal data (e.g., most popular color).

### Relationship to Distribution Shape

- **Symmetric distribution:** Mean ≈ Median ≈ Mode
- **Right-skewed distribution:** Mode < Median < Mean
- **Left-skewed distribution:** Mean < Median < Mode

## Summary

Each measure of central tendency serves a distinct purpose. The mean provides a mathematical average but is vulnerable to outliers, the median offers a robust center that resists extreme values, and the mode identifies the most frequent observation. Choosing the appropriate measure depends on the data's distribution shape and the analytical question at hand.
