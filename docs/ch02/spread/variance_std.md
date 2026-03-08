# Variance and Standard Deviation

## Overview

Variance and standard deviation are the most widely used measures of statistical dispersion. They quantify how much individual data points deviate from the mean, providing essential information about the spread and consistency of a dataset.

---

## 1. Variance

### Definition

Variance measures the average of the squared deviations from the mean. By squaring, it ensures all deviations contribute positively and penalizes larger deviations more heavily.

### Formulas

**Population variance:**

$$
\sigma^2 = \frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}
$$

**Sample variance (with Bessel's correction):**

$$
s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n - 1}
$$

The denominator $n - 1$ corrects for the downward bias that arises from using the sample mean $\bar{x}$ instead of the true population mean $\mu$.

### Example

For the dataset 70, 85, 90, 95, 100 with mean $\bar{x} = 88$:

1. Squared deviations: $(70-88)^2 = 324$, $(85-88)^2 = 9$, $(90-88)^2 = 4$, $(95-88)^2 = 49$, $(100-88)^2 = 144$
2. Sum: $324 + 9 + 4 + 49 + 144 = 530$
3. Sample variance: $s^2 = 530 / 4 = 132.5$

### Computing Variance in Python

```python
import numpy as np

sample_data = np.array([1.5, 2.5, 4, 2, 1, 1])

# Population variance (ddof=0, the default)
population_variance = sample_data.var()
print(f"Population Variance (ddof=0): {population_variance}")

# Sample variance (ddof=1)
sample_variance = sample_data.var(ddof=1)
print(f"Sample Variance (ddof=1): {sample_variance}")
```

### Interpretation

A variance of 132.5 means the exam scores vary, on average, by a squared distance of 132.5 units from the mean. Because variance is expressed in squared units, it can be difficult to interpret directly—which is why the standard deviation is often preferred.

---

## 2. Standard Deviation

### Definition

The standard deviation is the square root of the variance. It returns the measure of spread to the original units of the data, making it directly interpretable.

### Formulas

**Population standard deviation:**

$$
\sigma = \sqrt{\frac{\sum_{i=1}^{N} (x_i - \mu)^2}{N}}
$$

**Sample standard deviation:**

$$
s = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n - 1}}
$$

### Example

Using the variance from above: $s = \sqrt{132.5} \approx 11.51$.

This means the exam scores deviate from the mean by about 11.51 points on average.

### Computing Standard Deviation in Python

```python
import numpy as np

sample_data = np.array([1.5, 2.5, 4, 2, 1, 1])

# Population standard deviation (ddof=0)
population_std = sample_data.std()
print(f"Population Standard Deviation (ddof=0): {population_std}")

# Sample standard deviation (ddof=1)
sample_std = sample_data.std(ddof=1)
print(f"Sample Standard Deviation (ddof=1): {sample_std}")
```

### Applications

Standard deviation is used across many domains: assessing the volatility of financial returns, measuring the spread of scientific measurements, evaluating manufacturing consistency, and more. In a normal distribution, approximately 68% of data falls within one standard deviation of the mean, 95% within two, and 99.7% within three (the empirical rule).

---

## 3. Population vs. Sample: The `ddof` Parameter

When computing variance and standard deviation in NumPy and pandas, the `ddof` (delta degrees of freedom) parameter controls the denominator:

| Context | Denominator | `ddof` | Use When |
|---|---|---|---|
| Population | $N$ | 0 | You have the entire population |
| Sample | $n - 1$ | 1 | You have a sample from a larger population |

NumPy defaults to `ddof=0` (population), while pandas defaults to `ddof=1` (sample). Always be explicit about which you are computing.

---

## 4. Practical Considerations

**Data Distribution:** In a normal distribution, standard deviation has a clean interpretation via the empirical rule. In skewed distributions, it may not accurately reflect the typical spread.

**Sensitivity to Outliers:** Both variance and standard deviation are sensitive to extreme values because squaring amplifies large deviations. For skewed data or data with outliers, the IQR is a more robust alternative.

**Real-Life Examples:**

- **Stock Market Volatility:** Standard deviation of returns measures risk. Higher standard deviation means greater price fluctuation and higher investment risk.
- **Student Test Scores:** Low standard deviation indicates most students scored similarly; high standard deviation reveals wide performance variation.
- **Manufacturing Quality:** Standard deviation of product measurements indicates process consistency. Lower values mean tighter quality control.

## Summary

Variance and standard deviation provide a complete picture of data variability by considering every observation's distance from the mean. The standard deviation, being in the original units, is more interpretable and widely used. Understanding the distinction between population and sample formulas—and setting `ddof` correctly—is essential for accurate statistical analysis.
