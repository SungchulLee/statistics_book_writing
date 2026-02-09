# IQR and Robust Measures

## Overview

The **range**, **interquartile range (IQR)**, and **percentiles** are measures of spread that complement variance and standard deviation. The IQR is particularly valued as a **robust** measure—one that is resistant to the influence of outliers.

---

## 1. Range

### Definition

The range is the simplest measure of dispersion: the difference between the maximum and minimum values.

$$
\text{Range} = \text{Max} - \text{Min}
$$

### Example

For the dataset 70, 85, 90, 95, 100: Range = $100 - 70 = 30$.

### Computing Range

```python
import pandas as pd

url = 'https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/loans_income.csv'
loans_data = pd.read_csv(url)

data_range = loans_data['x'].max() - loans_data['x'].min()
print(f"{data_range = }")
```

### Limitations

The range is highly sensitive to outliers because it depends entirely on the two most extreme values. It provides no information about how data is distributed between these extremes.

---

## 2. Interquartile Range (IQR)

### Definition

The IQR measures the spread of the middle 50% of the data, effectively reducing the impact of outliers. It is the difference between the third quartile ($Q_3$, the 75th percentile) and the first quartile ($Q_1$, the 25th percentile).

$$
\text{IQR} = Q_3 - Q_1
$$

### Example

For the dataset 1, 3, 4, 6, 7, 9, 11: $Q_1 = 3$, $Q_3 = 9$, so $\text{IQR} = 9 - 3 = 6$.

### IQR and Standard Deviation: Income Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

url = 'https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/loans_income.csv'
df = pd.read_csv(url)

mean_income = df['x'].mean()
median_income = df['x'].median()
std_dev = df['x'].std()
q1 = df['x'].quantile(0.25)
q3 = df['x'].quantile(0.75)
iqr = stats.iqr(df['x'])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

# Mean ± Std Dev
ax1.hist(df['x'], bins=30, density=True, alpha=0.3, color='skyblue')
ax1.axvline(mean_income, color='blue', linestyle='--', label='mean')
ax1.axvline(mean_income - std_dev, color='red', linestyle='--', label='mean - std')
ax1.axvline(mean_income + std_dev, color='red', linestyle='--', label='mean + std')
ax1.legend()
ax1.set_title("Mean and Std Dev")

# Median and Quartiles
ax2.hist(df['x'], bins=30, density=True, alpha=0.3, color='skyblue')
ax2.axvline(median_income, color='blue', linestyle='--', label='median')
ax2.axvline(q1, color='red', linestyle='--', label='Q1')
ax2.axvline(q3, color='red', linestyle='--', label='Q3')
ax2.legend()
ax2.set_title("Median and Quartiles")

# Boxplot
ax3.boxplot(df['x'], vert=True, patch_artist=True)
ax3.set_title("Boxplot")

plt.tight_layout()
plt.show()
```

### Computing Quartiles

```python
import pandas as pd

url = 'https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/loans_income.csv'
df = pd.read_csv(url)

q1 = df['x'].quantile(0.25)
q2 = df['x'].median()
q3 = df['x'].quantile(0.75)

print(f"{q1 = }")
print(f"{q2 = }")  # Median
print(f"{q3 = }")
```

---

## 3. Percentiles

The $p$-th percentile is the value below which $p\%$ of the data falls.

### Percentiles and Deciles

$$
\begin{array}{llll}
D_1 = P_{10}, & D_2 = P_{20}, & \ldots, & D_9 = P_{90}
\end{array}
$$

### Percentiles and Quartiles

$$
Q_1 = P_{25}, \quad Q_2 = P_{50}, \quad Q_3 = P_{75}
$$

### Percentiles and Median

$$
\text{Median} = Q_2 = D_5 = P_{50}
$$

---

## 4. Comparing Measures of Spread

| Measure | Robustness | Information | Best For |
|---|---|---|---|
| Range | Not robust (extreme sensitivity) | Only two values | Quick overview |
| IQR | Robust (ignores outer 50%) | Middle 50% spread | Skewed data, outlier-prone data |
| Std Dev | Not robust (sensitive to outliers) | All data points | Symmetric, normal-like data |

### Real-Life Examples

**Income Variability:** The range shows the gap between richest and poorest. The IQR reveals how middle-income earners differ. The standard deviation quantifies overall income inequality.

**Student Test Scores:** Low standard deviation means most students scored similarly. A large IQR might indicate a wide spread in the middle tier of performers.

**Stock Market Volatility:** Variance and standard deviation are standard risk measures in finance. High standard deviation indicates greater price fluctuation and higher investment risk.

---

## 5. Practical Considerations

**Sample vs. Population:** When computing variance and standard deviation, use $n-1$ (Bessel's correction) for samples to obtain unbiased estimates.

**Data Distribution:** For normal distributions, standard deviation has a clean interpretation (empirical rule). For skewed distributions, the IQR paired with the median provides a more meaningful summary.

**Complementary Use:** In practice, reporting both mean ± standard deviation and median with IQR gives readers a complete picture, especially when the distribution shape is unknown or potentially skewed.

## Summary

The IQR and related percentile-based measures provide robust alternatives to variance and standard deviation for describing data spread. By focusing on the middle 50% of the data, the IQR is insensitive to outliers, making it the preferred measure of spread for skewed distributions and datasets with extreme values.
