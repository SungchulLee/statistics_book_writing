# Outliers and Leverage

## Overview

**Outliers** are data points that significantly differ from other observations in a dataset. They may be unusually high or low and can arise due to variability in the data, errors in data collection, or they may indicate special cases that deserve further investigation. Detecting and understanding outliers is crucial because they can distort statistical analyses such as the mean, variance, and regression models.

---

## 1. Types of Outliers

**Univariate Outliers:** Unusual with respect to a single variable. For example, in a dataset of student heights, an individual who is extremely short or tall compared to the others.

**Multivariate Outliers:** Appear normal when each variable is considered separately, but unusual patterns emerge when the relationship between multiple variables is examined.

## 2. Causes of Outliers

- **Measurement Error:** Mistakes in data entry, instrument errors, or inaccuracies during measurement.
- **Experimental Error:** Anomalous conditions during data collection.
- **Natural Variation:** Inherent variability in the system being studied.
- **Sampling Error:** Rare cases included in the dataset or insufficient sample size.

## 3. Effects of Outliers

**Impact on Central Tendency:** Outliers pull the mean toward extreme values, making it an inaccurate representation. For example, a CEO's salary in a small sample of salaries can skew the mean upward significantly.

**Impact on Variability:** Outliers inflate variance and standard deviation, as these measures are sensitive to extreme values.

**Impact on Statistical Models:** Outliers can have a disproportionate influence on regression models, potentially leading to misleading or biased coefficients that reduce generalizability.

---

## 4. Identifying Outliers

### Box Plot Method

Data points beyond $1.5 \times \text{IQR}$ from $Q_1$ or $Q_3$ are flagged as outliers:

- Lower fence: $Q_1 - 1.5 \times \text{IQR}$
- Upper fence: $Q_3 + 1.5 \times \text{IQR}$

### Z-Score Method

The Z-score measures how many standard deviations a data point is from the mean. Points with $|Z| > 3$ are typically considered outliers:

$$
Z = \frac{x - \mu}{\sigma}
$$

### IQR Method

Values below $Q_1 - 1.5 \times \text{IQR}$ or above $Q_3 + 1.5 \times \text{IQR}$ are classified as outliers.

### Scatterplot (Multivariate)

In multivariate data, scatterplots can reveal points that deviate significantly from the overall pattern or trend.

### Cook's Distance (Regression)

Cook's Distance identifies influential data points that have a large impact on regression model predictions. High values indicate potential outliers with leverage.

---

## 5. Five-Number Summary

The five-number summary provides a concise description that naturally highlights potential outliers through the box plot:

$$
\text{Min} \quad Q_1 \quad \text{Median} \quad Q_3 \quad \text{Max}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.array([1, 2, 0, 0, 0, 1, 3, 1, 2, 1, 2, 4, 5, -1, -2, 0, 8])

quantiles = {"Min": 0, "Q1": 0.25, "Median": 0.5, "Q3": 0.75, "Max": 1}

for label, q in quantiles.items():
    print(f"{label:6} : {np.quantile(data, q)}")

fig, ax = plt.subplots(figsize=(2, 3))
ax.boxplot(data)
ax.set_title("Boxplot of Data")
plt.show()
```

### Comparative Box Plots

Box plots are particularly effective when comparing distributions across groups or conditions:

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

---

## 6. Handling Outliers

**Investigate the Source:** Confirm whether outliers are erroneous before taking action. If confirmed as errors, correct or remove them.

**Transform the Data:** Log or square-root transformations can reduce the influence of outliers by compressing the scale.

**Use Robust Statistical Methods:** The median, IQR, and robust regression techniques (e.g., Lasso, Ridge) are less sensitive to outliers.

**Trimming or Winsorizing:** Trimming removes extreme data points. Winsorizing replaces outliers with the nearest non-outlier value.

**Keep the Outliers:** Sometimes outliers represent rare but important cases (e.g., extreme market events in finance) and should be retained for further investigation.

---

## 7. Real-Life Examples

**Income Distribution:** Extreme outliers such as tech billionaire incomes drastically increase the mean, making the median a more representative measure.

**Stock Market Analysis:** Large market movements during crises (e.g., 2008 financial crisis) appear as outliers in historical price data.

**Medical Studies:** Patients with unique drug responses may be outliers that reveal important information about subgroup effects.

## Summary

Outliers deserve careful attention rather than automatic removal. Understanding their source—whether error, natural variation, or a genuinely rare event—determines the appropriate response. The combination of visual tools (box plots, scatter plots) and numerical methods (Z-scores, IQR fences, Cook's Distance) provides a robust framework for outlier detection and management.
