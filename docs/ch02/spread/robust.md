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

## Exercises

**Exercise 1.**
Consider the dataset $\{2, 4, 5, 7, 8, 9, 11, 13, 15, 80\}$. Compute the range, the IQR, and the sample standard deviation. Which measure is most affected by the outlier at 80?

??? success "Solution to Exercise 1"
    **Range:** $80 - 2 = 78$.

    **IQR:** With $n = 10$ sorted values, the lower half is $\{2, 4, 5, 7, 8\}$ and the upper half is $\{9, 11, 13, 15, 80\}$. Thus $Q_1 = 5$ and $Q_3 = 13$, giving $\text{IQR} = 13 - 5 = 8$.

    **Standard deviation:** The mean is $\bar{x} = (2+4+5+7+8+9+11+13+15+80)/10 = 154/10 = 15.4$. The sum of squared deviations is $(2-15.4)^2 + \cdots + (80-15.4)^2 = 179.56 + 129.96 + 108.16 + 70.56 + 54.76 + 40.96 + 19.36 + 5.76 + 0.16 + 4177.16 = 4786.4$. Then $s = \sqrt{4786.4/9} \approx \sqrt{531.8} \approx 23.06$.

    The **range** is most dramatically affected (78 vs. what would be 13 without the outlier). The **standard deviation** is also heavily inflated (23.06 vs. roughly 4.2 without the outlier). The **IQR** is unaffected by the outlier since it depends only on the middle 50% of the data.

---

**Exercise 2.**
Explain why the IQR has a breakdown point of 25%, while the range has a breakdown point of 0%.

??? success "Solution to Exercise 2"
    The **breakdown point** of a statistic is the proportion of data that can be made arbitrarily extreme before the statistic becomes unbounded or meaningless.

    The **range** depends on exactly two values: the minimum and the maximum. Changing just one observation (the minimum or the maximum) to an extreme value will change the range arbitrarily. Therefore, a single corrupted observation (proportion $1/n \to 0\%$ as $n \to \infty$) can make the range arbitrarily large. The breakdown point is 0%.

    The **IQR** depends on $Q_1$ and $Q_3$, which are determined by the middle portion of the data. To shift $Q_1$ or $Q_3$ arbitrarily, you would need to corrupt more than 25% of the observations (either the bottom 25% or the top 25%). Therefore, the IQR can tolerate up to 25% contamination before it breaks down.

---

**Exercise 3.**
A dataset has $Q_1 = 20$, median $= 30$, and $Q_3 = 55$. Without seeing the raw data, what can you infer about the shape of the distribution from these three numbers alone?

??? success "Solution to Exercise 3"
    The distance from $Q_1$ to the median is $30 - 20 = 10$, while the distance from the median to $Q_3$ is $55 - 30 = 25$. Since the upper half of the IQR is much wider than the lower half, the distribution is **right-skewed** (positively skewed). The data is more spread out above the median than below it, indicating a longer right tail.

---

**Exercise 4.**
For a standard normal distribution $N(0,1)$, the theoretical quartiles are $Q_1 \approx -0.6745$ and $Q_3 \approx 0.6745$. Compute the theoretical IQR and compare it to the standard deviation $\sigma = 1$. What is the ratio $\text{IQR}/\sigma$?

??? success "Solution to Exercise 4"
    The theoretical IQR is:

    $$
    \text{IQR} = Q_3 - Q_1 = 0.6745 - (-0.6745) = 1.349
    $$

    The ratio is:

    $$
    \frac{\text{IQR}}{\sigma} = \frac{1.349}{1} = 1.349
    $$

    This means that for any normal distribution, $\text{IQR} \approx 1.349\sigma$. This relationship can be used to estimate the standard deviation from the IQR when the data is approximately normal: $\hat{\sigma} \approx \text{IQR}/1.349$.

---

**Exercise 5.**
Compare the **trimmed standard deviation** (computed after removing the top and bottom $p\%$) with the IQR as robust scale estimators. What are the trade-offs?

??? success "Solution to Exercise 5"
    **Trimmed SD:** sort the data, remove the top $p\%$ and bottom $p\%$, compute SD on the remaining $1 - 2p$ fraction. Trade-offs:

    - Breakdown point $= \min(p, 0.5)$. With $p = 0.25$, matches IQR.
    - Uses information from many observations (the middle $1 - 2p$), so generally more efficient than IQR which uses only two quantile estimates.
    - Smooth — small perturbations in data produce small changes (unlike the IQR, which is a function of only two order statistics).
    - Requires choosing $p$ and re-scaling for consistency under a reference distribution.

    **IQR:** $Q_3 - Q_1$.

    - Breakdown point $= 0.25$ (corrupt only one quartile required).
    - Simpler computation; familiar to most analysts via box plots.
    - Lower statistical efficiency under normality (about 37%) than trimmed SD.

    **Practical recommendation:** for descriptive summaries, IQR is sufficient and standard. For estimators feeding into downstream statistical procedures (where lower variance matters), the trimmed SD or even better M-estimators are preferred.

---

**Exercise 6.**
For a **lognormal distribution** with parameters $(\mu, \sigma^2)$ on the log scale, the variance and IQR can disagree dramatically. Discuss why and what this means for reporting spread in skewed data.

??? success "Solution to Exercise 6"
    If $X = \exp(Y)$ with $Y \sim N(\mu, \sigma^2)$, then $X$ is lognormal with mean $e^{\mu + \sigma^2/2}$ and variance $(e^{\sigma^2} - 1) e^{2\mu + \sigma^2}$.

    The variance grows roughly as $e^{\sigma^2}$, which can be very large for moderate $\sigma$ (e.g., $\sigma = 2$ gives variance multiplier $\sim 54$). The IQR depends only on the 25th and 75th percentiles of the lognormal, which are $\exp(\mu \pm 0.6745\sigma)$. So IQR grows linearly in $\exp$ of $\sigma$, much slower than variance.

    **Concrete example:** for $\mu = 0$, $\sigma = 2$:

    - Mean of $X \approx 7.39$, variance $\approx 401$, SD $\approx 20$.
    - $Q_1 \approx 0.259$, $Q_3 \approx 3.86$, IQR $\approx 3.6$.

    SD is about $5.5\times$ the IQR. Reporting "mean $\pm$ SD" of $7.4 \pm 20$ is misleading — the interval includes negative values, impossible for a positive random variable, and the SD is dominated by the long right tail rather than the typical scale.

    **Reporting recommendation for skewed data:** always report the **median + IQR**, or even better the **median + the 10th and 90th percentiles**, instead of mean ± SD. Many fields (income reporting, drug pharmacokinetics, earthquake magnitudes) work on log scales for exactly this reason.
