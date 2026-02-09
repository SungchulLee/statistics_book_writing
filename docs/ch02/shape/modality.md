# Modality

## Overview

The **modality** of a distribution describes the number of distinct peaks (modes) in its shape. Identifying modality is a critical first step in exploratory data analysis because it reveals whether the data comes from a single population or is a mixture of distinct subgroups.

## Unimodal Distributions

A **unimodal** distribution has a single peak. The most familiar example is the normal (bell curve) distribution, where data clusters around one central value.

**Example:** Heights of adult women in a single country typically form a unimodal distribution centered near the population mean.

## Bimodal Distributions

A **bimodal** distribution has two distinct peaks, indicating that the data likely contains two separate groups or processes.

**Example:** Test scores in a class might be bimodal if one group of students studied extensively and another did not, producing peaks at high and low scores with a valley in between.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)

# Two normal distributions centered at different locations
data_normal_1 = stats.norm().rvs(1_000)
data_normal_2 = stats.norm(loc=6).rvs(1_000)
combined_data = np.concatenate((data_normal_1, data_normal_2))

fig, ax = plt.subplots(figsize=(12, 3))
ax.hist(combined_data, bins=30, color='skyblue', edgecolor='black')
ax.set_title("Histogram of Bimodal Distribution")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
```

The two peaks are clearly visible, each corresponding to one of the component normal distributions.

## Multimodal Distributions

A **multimodal** distribution has more than two peaks. This often arises from mixing three or more subpopulations.

**Example:** The distribution of commute times in a large metropolitan area might show peaks at walking distance, short drive, and long commute durations.

## Why Modality Matters

Detecting modality has practical consequences for analysis:

- A bimodal or multimodal distribution signals that **summary statistics like the mean may be misleading**, as the mean could fall in a valley between peaks where few observations actually lie.
- It suggests that the data should be **disaggregated** into subgroups before further analysis.
- Standard parametric methods assuming unimodality (e.g., t-tests, normal-based confidence intervals) may be inappropriate for multimodal data.

## Detecting Modality

Common approaches include visual inspection of histograms and density plots, kernel density estimation (KDE) with varying bandwidths, and formal tests such as the dip test of unimodality. In practice, the histogram with a reasonable number of bins is the simplest and most effective first check.

## Summary

Modality provides essential information about the structure of a dataset. Unimodal distributions suggest a single homogeneous population, while bimodal or multimodal shapes point to underlying subgroups that deserve separate investigation.
