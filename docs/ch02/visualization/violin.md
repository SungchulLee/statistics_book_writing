# Violin Plots

## Overview

A **violin plot** combines a box plot with a kernel density estimate (KDE) on each side, showing the full distribution shape alongside summary statistics. Where a box plot reduces the distribution to five numbers plus outliers, a violin plot reveals multimodality, skewness, and density variations that box plots hide.

## Violin Plot vs. Box Plot

The key advantage of violin plots over box plots is the ability to show the **probability density** of the data at different values. This makes them particularly useful for:

- Detecting bimodal or multimodal distributions that a box plot would miss.
- Comparing distribution shapes across groups when the differences are subtle.
- Communicating the full distributional story to an audience.

## Basic Violin Plot

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

# Create a bimodal distribution that a box plot would obscure
data_1 = np.concatenate([np.random.normal(0, 1, 500),
                          np.random.normal(5, 1, 500)])
data_2 = np.random.normal(2.5, 2, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Violin plot reveals bimodality
ax1.violinplot([data_1, data_2], showmeans=True, showmedians=True)
ax1.set_title("Violin Plot")
ax1.set_xticks([1, 2])
ax1.set_xticklabels(["Bimodal", "Unimodal"])

# Box plot hides the bimodality
ax2.boxplot([data_1, data_2], labels=["Bimodal", "Unimodal"])
ax2.set_title("Box Plot")

plt.tight_layout()
plt.show()
```

## Violin Plot with Seaborn

Seaborn provides a more polished violin plot with built-in grouping:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

fig, ax = plt.subplots(figsize=(10, 4))
sns.violinplot(data=df, x="Pclass", y="Age", hue="Sex",
               split=True, ax=ax)
ax.set_title("Age Distribution by Class and Sex (Titanic)")
plt.show()
```

The `split=True` option places the two hue categories on opposite sides of each violin, enabling direct visual comparison within each class.

## When to Use Violin Plots

Violin plots are most valuable when comparing the shapes of distributions across groups, especially when the distributions may be non-normal or multimodal. For simple comparisons where only the median and IQR matter, box plots remain more concise and easier to read.

## Summary

Violin plots extend box plots by adding density information, making them ideal for revealing distributional details such as multimodality and asymmetry. They are particularly effective in group comparisons where distribution shape—not just summary statistics—drives the analysis.
