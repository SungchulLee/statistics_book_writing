# Paired-Sample Non-Parametric Tests

## Paired-Sample Wilcoxon Signed-Rank Test

The **Paired-Sample Wilcoxon Signed-Rank Test** is a non-parametric test used to determine whether the median difference between paired observations is significantly different from zero. It serves as a non-parametric alternative to the paired t-test when the data do not meet the normality assumption.

### Key Features

- **Purpose**: Test whether the median difference between paired observations is zero.
- **Null Hypothesis ($H_0$)**: The median of the differences between the paired samples is zero.
- **Alternative Hypothesis ($H_a$)**:
    - Two-tailed: The median difference is not zero.
    - One-tailed: The median difference is either greater than or less than zero.
- **Data Requirements**:
    - Data must be paired and continuous or ordinal.
    - The differences between pairs should be symmetrically distributed.

### Test Procedure

**Step 1: Calculate Differences**

Compute the difference between paired observations:

$$d_i = X_i - Y_i$$

Ignore pairs where $d_i = 0$ (these are excluded from the test).

**Step 2: Rank the Absolute Differences**

Take the absolute values $|d_i|$ and rank them in ascending order, assigning tied ranks if necessary.

**Step 3: Assign Signs to Ranks**

Assign the sign of each difference to its corresponding rank.

**Step 4: Compute the Test Statistic**

- Sum the ranks of the positive differences: $W^+$
- Sum the ranks of the negative differences: $W^-$
- The test statistic is: $W = \min(W^+, W^-)$

**Step 5: Determine Significance**

- Compare $W$ to a critical value from the Wilcoxon Signed-Rank Test table for the given sample size and significance level ($\alpha$).
- For larger samples ($n > 20$), use the normal approximation:

$$Z = \frac{W - \frac{n(n+1)}{4}}{\sqrt{\frac{n(n+1)(2n+1)}{24}}}$$

---

### Worked Example: Teaching Method Improvement

**Scenario**: A researcher wants to test whether a new teaching method improves test scores. Ten students take a test before and after using the new method. Test if there is a significant improvement in scores.

**Data:**

| Student | Before | After | Difference ($d_i$) |
|---|---|---|---|
| 1 | 70 | 72 | 2 |
| 2 | 68 | 69 | 1 |
| 3 | 75 | 78 | 3 |
| 4 | 80 | 85 | 5 |
| 5 | 72 | 75 | 3 |
| 6 | 74 | 76 | 2 |
| 7 | 69 | 70 | 1 |
| 8 | 77 | 79 | 2 |
| 9 | 73 | 74 | 1 |
| 10 | 76 | 80 | 4 |

**Step 1 — Differences:** All differences are positive: $d = [2, 1, 3, 5, 3, 2, 1, 2, 1, 4]$

**Step 2 — Rank Absolute Differences:**

$$|d| = [2, 1, 3, 5, 3, 2, 1, 2, 1, 4]$$

Sorted: $[1, 1, 1, 2, 2, 2, 3, 3, 4, 5]$

Tied ranks: $[2, 2, 2, 5, 5, 5, 7.5, 7.5, 9, 10]$

Assigned to original order: $[5, 2, 7.5, 10, 7.5, 5, 2, 5, 2, 9]$

**Step 3 — Assign Signs:** Since all differences are positive, all ranks are positive.

**Step 4 — Compute $W^+$ and $W^-$:**

- $W^+ = 5 + 2 + 7.5 + 10 + 7.5 + 5 + 2 + 5 + 2 + 9 = 55$
- $W^- = 0$
- $W = \min(55, 0) = 0$

**Step 5 — Determine Significance:**

Using a Wilcoxon table for $n = 10$, $\alpha = 0.05$ (two-tailed), the critical value is 8. Since $W = 0 < 8$, we **reject** $H_0$.

**Conclusion**: There is significant evidence that the new teaching method improves test scores.

---

### Python Implementation

```python
import numpy as np
from scipy.stats import wilcoxon

# Data: before and after scores
before = np.array([70, 68, 75, 80, 72, 74, 69, 77, 73, 76])
after = np.array([72, 69, 78, 85, 75, 76, 70, 79, 74, 80])

# Perform Wilcoxon Signed-Rank Test
stat, p_value = wilcoxon(after, before)

print(f"Test Statistic: {stat}")
print(f"P-value: {p_value}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: Significant improvement in scores.")
else:
    print("Fail to reject the null hypothesis: No significant improvement in scores.")
```

---

## Paired Sign Test

When the symmetry assumption of the Wilcoxon Signed-Rank test is violated, the **Paired Sign Test** can be used. It only considers the signs of the paired differences, ignoring both magnitude and ranks.

### Procedure

1. Compute differences $d_i = X_i - Y_i$.
2. Count the number of positive ($n_+$) and negative ($n_-$) differences. Ignore ties ($d_i = 0$).
3. Under $H_0$, each difference is equally likely to be positive or negative, so $n_+ \sim \text{Binomial}(n, 0.5)$.
4. Compute p-value using the binomial distribution.

```python
from scipy.stats import binom
import numpy as np

before = np.array([70, 68, 75, 80, 72, 74, 69, 77, 73, 76])
after = np.array([72, 69, 78, 85, 75, 76, 70, 79, 74, 80])

differences = after - before
n_plus = np.sum(differences > 0)
n_minus = np.sum(differences < 0)
n = n_plus + n_minus
W = min(n_plus, n_minus)

p_value = 2 * binom.cdf(W, n, 0.5)  # Two-tailed

print(f"n+ = {n_plus}, n- = {n_minus}")
print(f"P-value: {p_value:.4f}")
```

---

## Comparison: Paired t-Test vs Non-Parametric Alternatives

| Feature | Paired t-Test | Paired Wilcoxon | Paired Sign Test |
|---|---|---|---|
| **Assumption** | Normal differences | Symmetric differences | None |
| **Tests** | Mean difference | Median difference | Median difference |
| **Uses** | Raw differences | Ranks of differences | Signs only |
| **Power** | Highest (when normal) | Moderate | Lowest |
| **Robustness** | Sensitive to outliers | Moderate | Very robust |

### Guideline for Choosing

- **Normal differences**: Use the **paired t-test** for maximum power.
- **Symmetric but non-normal differences**: Use the **paired Wilcoxon signed-rank test**.
- **No assumptions met** (skewed, ordinal): Use the **paired sign test**.

### Advantages and Limitations

**Advantages:**

- Does not assume normality.
- Robust to outliers.
- Works with ordinal data.

**Limitations:**

- Assumes symmetry of the distribution of differences (Wilcoxon only).
- Less powerful than parametric tests when normality holds.
- Sign test discards magnitude information, reducing power further.
