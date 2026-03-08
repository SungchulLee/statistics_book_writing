# Boxplots

## Overview

A **box plot** (or box-and-whisker plot) is a standardized way of displaying the distribution of data based on the five-number summary: minimum, first quartile ($Q_1$), median ($Q_2$), third quartile ($Q_3$), and maximum. It provides a compact visual summary of center, spread, skewness, and outliers simultaneously.

## Anatomy of a Box Plot

The components of a box plot are:

- **Box:** Spans from $Q_1$ to $Q_3$, covering the interquartile range (IQR = $Q_3 - Q_1$). The length of the box represents the middle 50% of the data.
- **Median line:** A line inside the box at $Q_2$.
- **Whiskers:** Extend from the box to the most extreme data points within $1.5 \times \text{IQR}$ of $Q_1$ and $Q_3$.
- **Outliers:** Individual points plotted beyond the whiskers.

## Basic Box Plot: Titanic Passenger Ages

```python
import matplotlib.pyplot as plt
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url, index_col='PassengerId')

fig, ax = plt.subplots(figsize=(5, 3))
df['Age'].plot(kind='box', ax=ax, vert=False)
ax.set_title("Horizontal Boxplot of Passenger Ages on Titanic")
ax.set_xlabel("Age")
ax.spines[["top", "left", "right"]].set_visible(False)
plt.show()
```

## Detecting Skewness from Box Plots

Box plots provide a quick diagnostic for distribution shape:

$$
\begin{array}{lll}
\text{Left\_Box} > \text{Right\_Box} &\Rightarrow& \text{Left-skewed} \\
\text{Left\_Box} < \text{Right\_Box} &\Rightarrow& \text{Right-skewed} \\
\text{Boxes equal, Left\_Whisker} > \text{Right\_Whisker} &\Rightarrow& \text{Left-skewed} \\
\text{Boxes equal, Left\_Whisker} < \text{Right\_Whisker} &\Rightarrow& \text{Right-skewed} \\
\text{Both equal} &\Rightarrow& \text{Symmetric} \\
\end{array}
$$

## Paired Histogram and Box Plot

Displaying a histogram alongside a box plot makes the connection between shape and summary statistics explicit:

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

np.random.seed(0)
main_data = stats.norm().rvs(1_000)
right_1 = stats.norm(loc=2).rvs(200)
right_2 = stats.norm(loc=4).rvs(100)
combined = np.concatenate((main_data, right_1, right_2))

fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(12, 6))

ax_hist.hist(combined, density=True, bins=30)
ax_hist.set_title('Histogram of Right-Skewed Data')

ax_box.boxplot(combined, vert=False)
ax_box.set_title('Boxplot of Right-Skewed Data')

plt.tight_layout()
plt.show()
```

## Comparative Box Plots

Box plots are most powerful when used to compare distributions across groups:

```python
import numpy as np
import matplotlib.pyplot as plt

data_a = np.array([1, 2, 0, 0, 0, 1, 3, 1, 2, 1, 2, 4, 5, -1, -2, 0, 8])
data_b = np.array([1, 2, 0, 0, 0, 1, 3, 1, 2, 1, 2, 4, 5, -1, -2, 0, -8]) * 0.5
data_c = np.array([1, 2, 0, 0, 0, 1, 3, 1, 2, 1, 2, 4, 5, -1, -2, 0, 10, -7]) * 0.25

fig, ax = plt.subplots()
ax.boxplot([data_a, data_b, data_c],
           labels=["$10^4$", "$5 \\cdot 10^4$", "$10^5$"])
ax.plot([0, 1, 2, 3, 4], [1, 1, 1, 1, 1],
        label="FIM Delta", linestyle="--", color="r", alpha=0.7)
ax.legend()
ax.set_ylim(-10.0, 10.0)
ax.set_xlabel('Number of Samples')
ax.set_ylabel('MC Delta')
plt.show()
```

## Summary

Box plots are a compact, information-rich visualization that reveal center (median), spread (IQR and whisker length), skewness (box and whisker asymmetry), and outliers (individual points) all in a single graphic. They are especially effective for comparing distributions across groups or conditions.
