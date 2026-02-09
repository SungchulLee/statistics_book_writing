# Effect Size and Cramér's V

## Overview

The chi-square test statistic tells us whether there is a statistically significant association, but it does not tell us how **strong** that association is. For large samples, even trivially small deviations from independence can produce highly significant p-values. Effect size measures address this limitation by quantifying the **magnitude** of the association, independent of sample size.

## Cramér's V

**Cramér's V** is the most commonly used effect size measure for chi-square tests. It is defined as:

$$
V = \sqrt{\frac{\chi^2}{n \cdot (q - 1)}}
$$

where:

- $\chi^2$ is the chi-square test statistic,
- $n$ is the total sample size,
- $q = \min(r, c)$ is the smaller of the number of rows $r$ and columns $c$.

### Properties

- $V$ ranges from 0 to 1.
- $V = 0$ indicates no association (complete independence).
- $V = 1$ indicates perfect association.
- $V$ is symmetric: it does not depend on which variable is in the rows vs. columns.

### Interpretation Guidelines

| Cramér's V   | Interpretation  |
|:------------:|:---------------:|
| 0.00 – 0.10  | Negligible      |
| 0.10 – 0.30  | Small           |
| 0.30 – 0.50  | Medium          |
| 0.50+        | Large           |

These thresholds are approximate and context-dependent. In some fields, even a "small" effect size may be practically meaningful.

### Special Case: 2×2 Tables

For a $2 \times 2$ table, $q - 1 = 1$, so Cramér's V simplifies to:

$$
V = \sqrt{\frac{\chi^2}{n}} = |\phi|
$$

where $\phi$ is the **phi coefficient**, another common measure of association for $2 \times 2$ tables.

## Python Implementation

```python
import numpy as np
from scipy import stats

def cramers_v(observed):
    """
    Compute Cramér's V for a contingency table.

    Parameters:
    observed (numpy array): 2D array of observed counts.

    Returns:
    float: Cramér's V statistic.
    """
    chi2, p_value, df, expected = stats.chi2_contingency(observed)
    n = observed.sum()
    q = min(observed.shape) - 1
    v = np.sqrt(chi2 / (n * q))
    return v, chi2, p_value

# Example: Gender vs Handedness
observed = np.array([[934, 1070], [113, 92], [20, 8]])
v, chi2, p_value = cramers_v(observed)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Cramér's V: {v:.4f}")
```

## When to Use Effect Size

- Always report effect size alongside the chi-square test result, especially for large samples.
- Use effect size to compare the strength of associations across different studies or datasets.
- A statistically significant result with a very small Cramér's V may not be practically important.
- A non-significant result with a moderate Cramér's V in a small sample may warrant further investigation with a larger sample.
