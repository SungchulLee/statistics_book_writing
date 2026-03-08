# Median Absolute Deviation (MAD)

## Overview

The **Median Absolute Deviation (MAD)** is a robust measure of statistical dispersion that measures the spread of data around the median. Unlike variance and standard deviation, MAD is resistant to outliers, making it an ideal complement to the median for describing skewed or contaminated datasets.

---

## Definition

The MAD is computed in three steps:

1. Find the median $M = \text{median}(x_1, x_2, \ldots, x_n)$
2. Compute the absolute deviations: $d_i = |x_i - M|$ for each observation
3. Find the median of these deviations: $\text{MAD} = \text{median}(d_1, d_2, \ldots, d_n)$

$$
\text{MAD} = \text{median}(|x_i - \text{median}(x)|)
$$

### Standardization Constant

To make MAD directly comparable to standard deviation (particularly for normally distributed data), multiply by a standardization constant:

$$
\text{Standardized MAD} = 0.6745 \times \text{MAD}
$$

The constant 0.6745 is the 75th percentile of the standard normal distribution, chosen so that for normally distributed data, standardized MAD ≈ standard deviation.

---

## Example: U.S. State Population

Using state population data, compute MAD and compare to standard deviation:

```python
import pandas as pd
from statsmodels import robust

# Load state data
state = pd.read_csv('state.csv')

# Standard deviation (sensitive to outliers)
std_dev = state['Population'].std()
print(f"Standard Deviation: {std_dev:,.0f}")

# MAD using statsmodels
mad = robust.scale.mad(state['Population'])
print(f"MAD (standardized): {mad:,.0f}")

# Manual calculation
median_pop = state['Population'].median()
abs_deviations = abs(state['Population'] - median_pop)
mad_manual = abs_deviations.median()
mad_standardized = mad_manual / 0.6744897501960817
print(f"MAD (manual calc): {mad_standardized:,.0f}")
```

**Output:**
```
Standard Deviation: 6,848,235
MAD (standardized): 3,849,876
MAD (manual calc): 3,849,876
```

California's extreme population (37M vs. a median of 4.4M) heavily influences the standard deviation, pulling it upward. The MAD, based on deviations from the median, is less affected by this outlier.

---

## Why MAD is Robust

Consider the effect of outliers on these two measures:

```python
import pandas as pd
import numpy as np
from statsmodels import robust

# Original state population data
state = pd.read_csv('state.csv')
original_std = state['Population'].std()
original_mad = robust.scale.mad(state['Population'])

# Introduce extreme outliers
population_with_outliers = pd.concat([
    state['Population'],
    pd.Series([100_000_000, 150_000_000])  # Two fictional giant states
])

outlier_std = population_with_outliers.std()
outlier_mad = robust.scale.mad(population_with_outliers)

print("Impact of Outliers:")
print(f"  Std Dev: {original_std:,.0f} → {outlier_std:,.0f} ({100 * (outlier_std - original_std) / original_std:.1f}% increase)")
print(f"  MAD:     {original_mad:,.0f} → {outlier_mad:,.0f} ({100 * (outlier_mad - original_mad) / original_mad:.1f}% increase)")
```

Adding two extreme outliers dramatically increases standard deviation but barely affects MAD. This demonstrates MAD's robustness.

---

## Robustness Properties

MAD is a **robust** statistic with:

- **Breakdown point:** Up to 50% of data can be arbitrarily contaminated before MAD becomes unreliable, compared to 0% for standard deviation.
- **Influence function:** Bounded—one extreme outlier has limited effect.
- **Efficiency:** For normally distributed data, MAD is about 64% as efficient as standard deviation. This efficiency loss is small, given MAD's massive robustness gain.

---

## Comparison: Standard Deviation vs. MAD

| Characteristic | Standard Deviation | MAD |
|---|---|---|
| Sensitivity to outliers | High | Low |
| Uses all data points | Yes | Yes |
| Breakdown point | 0% | 50% |
| Computational complexity | $O(n)$ | $O(n \log n)$ (due to sorting) |
| Interpretability | Familiar to most analysts | Less familiar |
| Efficiency (normal data) | 100% | 64% |

---

## When to Use MAD

**Skewed distributions:** Income, wealth, or other right-skewed financial data
**Outlier-prone datasets:** Sensor measurements, astronomical observations
**Robust estimation:** When you cannot trust all data points equally
**Non-normal data:** Heavy-tailed or multimodal distributions

---

## Practical Example: Financial Returns

For stock market analysis, MAD can be more representative than standard deviation:

```python
import pandas as pd
from statsmodels import robust

# Hypothetical daily stock returns
returns = pd.Series([0.01, 0.02, -0.01, 0.015, -0.005, 0.03, -0.02,
                      0.01, -0.01, 0.005, -0.015, 0.02, -0.50])  # One crash day

print(f"Standard Deviation: {returns.std():.4f}")
print(f"MAD (standardized): {robust.scale.mad(returns):.4f}")

# The crash day (-0.50) inflates std dev much more than MAD
```

The single crash day (-0.50) vastly increases standard deviation, which might overstate typical daily volatility. MAD provides a clearer picture of routine variation.

---

## Computing MAD in Python

### Using statsmodels (recommended)

```python
from statsmodels import robust
import pandas as pd

data = pd.Series([1, 2, 3, 4, 5, 100])  # Last value is an outlier
mad = robust.scale.mad(data)
print(f"MAD: {mad:.2f}")
```

### Manual Calculation

```python
import pandas as pd
import numpy as np

data = pd.Series([1, 2, 3, 4, 5, 100])
median = data.median()
abs_dev = abs(data - median)
mad = abs_dev.median()
mad_standardized = mad / 0.6744897501960817  # Standardize for normal data
print(f"MAD (standardized): {mad_standardized:.2f}")
```

---

## Summary

The Median Absolute Deviation is a powerful tool for measuring data spread in the presence of outliers. By basing dispersion on deviations from the median (itself robust), MAD achieves a level of stability that variance and standard deviation cannot match. For any analysis involving skewed data, outliers, or non-normal distributions, pairing the median with MAD provides a more trustworthy summary than the mean with standard deviation.
