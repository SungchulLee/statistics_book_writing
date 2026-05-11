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

## Exercises

**Exercise 1.**
A chi-square test of independence on a $3 \times 2$ table with $n = 500$ observations yields $\chi^2 = 12.5$. Compute Cramér's V and classify the effect size.

??? success "Solution to Exercise 1"
    For a $3 \times 2$ table, $\min(r-1, c-1) = \min(2, 1) = 1$.

    $$
    V = \sqrt{\frac{\chi^2}{n \cdot \min(r-1, c-1)}} = \sqrt{\frac{12.5}{500 \times 1}} = \sqrt{0.025} \approx 0.158
    $$

    Using Cohen's benchmarks for $df^* = 1$: small $\approx 0.10$, medium $\approx 0.30$, large $\approx 0.50$. With $V = 0.158$, this is a **small to medium** effect size.

---

**Exercise 2.**
Two studies test the same hypothesis about the association between gender and voting preference. Study A ($n = 100$) finds $\chi^2 = 4.0$ ($p = 0.046$). Study B ($n = 10{,}000$) finds $\chi^2 = 4.0$ ($p = 0.046$). Compute Cramér's V for both. What does this reveal?

??? success "Solution to Exercise 2"
    Both are $2 \times 2$ tables, so $\min(r-1, c-1) = 1$.

    **Study A:** $V = \sqrt{4.0 / (100 \times 1)} = \sqrt{0.04} = 0.20$ (small-to-medium effect).

    **Study B:** $V = \sqrt{4.0 / (10{,}000 \times 1)} = \sqrt{0.0004} = 0.02$ (negligible effect).

    Despite identical $\chi^2$ values and p-values, the effect sizes are dramatically different. Study A found a meaningful association; Study B found a statistically significant but practically negligible association. This illustrates why effect size should always be reported alongside significance tests.

---

**Exercise 3.**
Explain why Cramér's V is preferred over the raw chi-square statistic for comparing the strength of association across different studies.

??? success "Solution to Exercise 3"
    The chi-square statistic is proportional to sample size: $\chi^2 \approx n \cdot V^2$. Doubling $n$ roughly doubles $\chi^2$ even if the strength of association remains the same. This makes $\chi^2$ unsuitable for comparing across studies with different sample sizes.

    Cramér's V normalizes by $n$ and by the table dimensions, producing a value in $[0, 1]$ that is independent of sample size. A $V = 0.30$ in a study of 100 people represents the same strength of association as $V = 0.30$ in a study of 10,000 people, making cross-study comparisons meaningful.

---

**Exercise 4.**
For a $2 \times 2$ table, show that Cramér's V equals the absolute value of the phi coefficient $|\phi|$.

??? success "Solution to Exercise 4"
    For a $2 \times 2$ table, $\min(r-1, c-1) = \min(1, 1) = 1$. Therefore:

    $$
    V = \sqrt{\frac{\chi^2}{n \cdot 1}} = \sqrt{\frac{\chi^2}{n}}
    $$

    The phi coefficient is defined as:

    $$
    \phi = \sqrt{\frac{\chi^2}{n}}
    $$

    (with a sign convention for $2 \times 2$ tables). Since $V$ takes the square root and is always non-negative, $V = |\phi|$. For $2 \times 2$ tables, Cramér's V and the absolute phi coefficient are identical measures.
