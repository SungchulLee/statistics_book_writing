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

## Exercises

**Exercise 1.**
A quality-control process records 10 diameter measurements (mm): $10.1, 10.0, 9.9, 10.2, 10.0, 9.8, 10.1, 10.0, 15.3, 10.0$. (a) Compute $s$. (b) Compute MAD. (c) Compute scaled MAD ($1.4826 \cdot \text{MAD}$). Compare and explain.

??? success "Solution to Exercise 1"
    (a) $\bar x = 10.54$. Sum of squared deviations $= 25.284$ (with the $(15.3 - 10.54)^2 = 22.66$ term contributing about 90%). $s^2 = 25.284/9 = 2.809$, $s \approx 1.676$.

    (b) Sorted: $9.8, 9.9, 10.0, 10.0, 10.0, 10.0, 10.1, 10.1, 10.2, 15.3$. Median $= 10.0$. Sorted absolute deviations: $0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 5.3$. MAD $= (0.1 + 0.1)/2 = 0.1$.

    (c) Scaled MAD $= 1.4826 \times 0.1 \approx 0.148$.

    The SD ($1.68$) is more than 11× the scaled MAD ($0.15$). The single outlier 15.3 inflates the SD enormously while leaving the MAD essentially untouched. The MAD is a far more honest measure of typical spread for this data.

---

**Exercise 2.**
Derive the consistency constant $1/\Phi^{-1}(0.75) \approx 1.4826$ that scales MAD to equal $\sigma$ under a normal distribution.

??? success "Solution to Exercise 2"
    For $X \sim N(\mu, \sigma^2)$, the median is $\mu$, so $|X - \mu|/\sigma$ has the **half-normal** distribution. We want to find $c$ such that the median of $|X - \mu|$ equals $\sigma$ when MAD is multiplied by $c$ — i.e., $c \cdot \text{MAD} = \sigma$.

    By symmetry, $P(|X - \mu| \le m) = P(-m \le X - \mu \le m) = 2\Phi(m/\sigma) - 1$. Setting this to 0.5 (the definition of the median):

    $$
    2\Phi(m/\sigma) - 1 = 0.5 \implies \Phi(m/\sigma) = 0.75 \implies m/\sigma = \Phi^{-1}(0.75) \approx 0.6745
    $$

    So the population MAD is $0.6745 \sigma$. The scaling constant $c = 1/0.6745 \approx 1.4826$ converts MAD back to $\sigma$. This is why most software libraries (R's `mad()`, statsmodels' `robust.scale.mad`) automatically apply this constant.

---

**Exercise 3.**
The **breakdown point** of an estimator is the fraction of data that must be replaced by arbitrary values before the estimator can be made arbitrarily far from the true value. Show that the breakdown point of MAD is 50% while that of the SD is 0%.

??? success "Solution to Exercise 3"
    **SD breakdown 0%:** consider a sample of size $n$ with finite values. Replace any single observation $x_i$ with a value $M$. The new mean grows like $M/n$ but the new SD grows like $M/\sqrt{n}$. As $M \to \infty$, both grow without bound. So replacing $1/n$ of the data (the smallest non-zero fraction) is enough to push the SD arbitrarily high. Since $1/n \to 0$, the breakdown point is 0.

    **MAD breakdown 50%:** the median of any sample changes by at most one rank position per replacement; after replacing fewer than half the values, the median is still pinned to the original "middle" data. Similarly, the median of the absolute deviations $|x_i - \text{median}|$ depends on the bulk of the data. Only when we replace at least $\lceil n/2 \rceil$ values can we shift the median (and thus the MAD) to arbitrary positions. So the breakdown is $\lfloor n/2 \rfloor / n \approx 0.5$ for large $n$.

    This is the highest possible breakdown point for any reasonable location/scale estimator — the 50% theoretical ceiling, matched only by the median and MAD (and a few other M-estimators).

---

**Exercise 4.**
The MAD has lower **efficiency** than the SD under normality: about 37% Gaussian efficiency. Define statistical efficiency and explain the bias-variance trade-off that justifies preferring MAD anyway in many applied contexts.

??? success "Solution to Exercise 4"
    **Efficiency** of an estimator $\hat\theta$ relative to a benchmark estimator $\hat\theta^*$ is the ratio of their asymptotic variances. Under normality, $\sigma_{\text{eff(MAD)}} \approx 0.37 \cdot \sigma_{\text{eff(SD)}}$, meaning MAD has roughly $1/0.37 \approx 2.7$ times higher variance than the SD when the data is truly normal.

    **Trade-off:**

    - If the data is *exactly* normal, the SD wastes no information and has 100% efficiency; MAD wastes data and is less precise.
    - If the data is *contaminated* — even a tiny fraction of outliers — the SD's variance balloons because outliers contribute squared terms. MAD's variance stays roughly the same.

    For real data which is almost never exactly normal, the cost of MAD's lower Gaussian efficiency is more than compensated by its insensitivity to contamination. The general design principle is: **never optimize for the worst case (heavy contamination) at the cost of catastrophic failure under modest violations of assumed normality.** This is the heart of "robust statistics."

---

**Exercise 5.**
The **modified Z-score** for outlier detection is $M_i = 0.6745 \cdot (x_i - \tilde x) / \text{MAD}$. Why is this preferred over the classical Z-score $Z_i = (x_i - \bar x) / s$ for outlier detection?

??? success "Solution to Exercise 5"
    The classical Z-score uses $\bar x$ (sensitive to outliers) and $s$ (highly sensitive). Outliers inflate both, **masking** themselves: the very point that should be flagged has reduced $|Z|$ because it pulled the mean and SD toward itself.

    The modified Z-score uses the median (breakdown 50%) and MAD (breakdown 50%). Outliers have negligible effect on either, so the Z-like statistic stays large for genuine outliers. The factor $0.6745$ makes the modified Z-score comparable in scale to a classical Z under normality — i.e., $|M_i| > 3.5$ corresponds to roughly the same tail rarity as $|Z_i| > 3$ in clean data.

    Iglewicz and Hoaglin (1993) recommended the threshold $|M_i| > 3.5$ for outlier flagging, providing a robust alternative to the classical $|Z| > 3$ rule.

---

**Exercise 6.**
The MAD is one of several robust scale estimators. Compare it briefly with the **interquartile range** (IQR) and **Qn estimator** (Rousseeuw–Croux). When would you choose each?

??? success "Solution to Exercise 6"
    **MAD:** median of $|x_i - \tilde x|$. Breakdown 50%, Gaussian efficiency 37%. Simple, widely implemented, the default robust scale.

    **IQR:** $Q_3 - Q_1$. Breakdown 25% (need only corrupt one quartile). Simpler conceptually but lower breakdown. Scale-consistent with $\sigma$ via $1.349 \sigma$ for normal data. The de-facto standard for box plots.

    **Qn estimator** (Rousseeuw and Croux 1993): a robust scale based on differences $|x_i - x_j|$, computed as the first quartile of all such pairwise differences with a normalizing constant. Breakdown 50%, Gaussian efficiency 82% — about 2× better than MAD. The cost is $O(n \log n)$ computation versus $O(n)$ for MAD.

    **When to use:**

    - **MAD**: default robust scale; simple, fast, well-known.
    - **IQR**: box plots and quick descriptive summaries; not when high breakdown is essential.
    - **Qn**: large samples where the higher Gaussian efficiency matters and computation cost is acceptable. State-of-the-art for serious robust estimation.
