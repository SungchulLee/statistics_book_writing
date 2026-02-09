# 19.1 One-Sample Non-Parametric Tests

This section introduces three foundational non-parametric tests that operate on a single sample (or treat paired data as a single sample of differences): the **Runs Test** for randomness, the **Sign Test** for the median, and the **Wilcoxon Signed-Rank Test** which uses both the sign and magnitude of deviations.

---

## 19.1.1 Runs Test (Wald–Wolfowitz)

### Concept

The **Wald–Wolfowitz runs test** checks whether a two-valued data sequence is random. A **run** is a maximal consecutive subsequence of identical elements. For example, in the sequence

$$
\underbrace{+ + +}_{\text{run 1}} \; \underbrace{- -}_{\text{run 2}} \; \underbrace{+}_{\text{run 3}} \; \underbrace{- - -}_{\text{run 4}}
$$

there are 4 runs. Too few runs suggest the data are clustered (positive autocorrelation); too many suggest systematic alternation.

### Hypotheses

$$
H_0: \text{The elements of the sequence are mutually independent (random).}
$$

$$
H_a: \text{The elements are not independent.}
$$

### Test Statistic

Given a sequence of length $N$ containing $N_+$ values of one type and $N_- = N - N_+$ of the other:

**Number of runs:**

$$
R = \frac{N_+ + N_- + 1 - \sum_{i=1}^{N-1} x_i \, x_{i+1}}{2}
$$

where the data are coded as $+1$ and $-1$.

**Mean and standard deviation under $H_0$:**

$$
\mu_R = \frac{2 \, N_+ \, N_-}{N} + 1
$$

$$
\sigma_R = \sqrt{\frac{(\mu_R - 1)(\mu_R - 2)}{N - 1}}
$$

**Standardized test statistic (normal approximation):**

$$
Z = \frac{R - \mu_R}{\sigma_R}
$$

Under $H_0$, $Z \xrightarrow{d} \mathcal{N}(0,1)$ for large $N$.

**p-value (two-sided):**

$$
p = 2 \, \Phi(-|Z|)
$$

### Implementation

```python
import numpy as np
import scipy.stats as stats

def runs_test(data):
    """
    Wald-Wolfowitz runs test for randomness.

    Parameters
    ----------
    data : array-like
        A sequence of +1 and -1 values.

    Returns
    -------
    statistic : float
        The Z test statistic.
    p_value : float
        Two-sided p-value.
    """
    data = np.asarray(data)
    N = data.shape[0]
    N_plus = (data == 1).sum()
    N_minus = N - N_plus

    mu = 2 * N_plus * N_minus / N + 1
    sigma = np.sqrt((mu - 1) * (mu - 2) / (N - 1))

    # Count runs using adjacent-element products
    R = (N_plus + N_minus + 1 - np.sum(data[1:] * data[:-1])) / 2

    statistic = (R - mu) / sigma
    p_value = 2 * stats.norm.cdf(-abs(statistic))

    return statistic, p_value
```

### Worked Examples

**Example 1 — Clustered data (non-random):**

```python
data = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
statistic, p_value = runs_test(data * 2 - 1)  # convert {0,1} → {-1,+1}
print(f"{statistic = :.4f}")   # large |Z|
print(f"{p_value   = :.4f}")   # p < 0.05 → reject randomness
```

This sequence has only 2 runs (one block of 1s, one block of 0s), far fewer than expected under randomness.

**Example 2 — Well-mixed data (random):**

```python
data = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
statistic, p_value = runs_test(data * 2 - 1)
print(f"{statistic = :.4f}")
print(f"{p_value   = :.4f}")   # p > 0.05 → cannot reject randomness
```

**Example 3 — Simulated binomial data:**

```python
np.random.seed(1)
data = np.random.binomial(n=1, p=0.4, size=200)
statistic, p_value = runs_test(data * 2 - 1)
print(f"{statistic = :.4f}")
print(f"{p_value   = :.4f}")   # truly random → expect p > 0.05
```

### Interpretation

| Outcome | Meaning |
|:--------|:--------|
| $p < \alpha$ | Reject $H_0$: samples are **not** independently drawn |
| $p \geq \alpha$ | Fail to reject $H_0$: samples are consistent with independence |

### Applications in Finance

The runs test is commonly applied to stock return series to test the **random walk hypothesis**: if successive returns are independent, the sequence of positive and negative returns should appear random.

---

## 19.1.2 Sign Test

### Concept

The **sign test** is one of the simplest non-parametric tests. It tests whether the median of a distribution equals a hypothesized value $m_0$ by counting how many observations fall above and below $m_0$.

For paired data $(x_i, y_i)$, we compute differences $d_i = x_i - y_i$ and test whether the median difference is zero. The test only uses the **signs** of the differences, ignoring their magnitudes.

### Hypotheses

Let $p = P(X > m_0)$ (or $p = P(d_i > 0)$ for paired data). Under $H_0$: median $= m_0$, we have $p = 0.5$.

| Test type | $H_0$ | $H_a$ |
|:----------|:-------|:-------|
| Two-sided | $p = 0.5$ | $p \neq 0.5$ |
| Left-tailed | $p = 0.5$ | $p < 0.5$ |
| Right-tailed | $p = 0.5$ | $p > 0.5$ |

### Handling Ties

Observations exactly equal to $m_0$ (or tied pairs where $d_i = 0$) are **excluded** from the analysis. Only the remaining $n = n_+ + n_-$ observations are used.

### Test Statistic

$$
T = \sum_{i=1}^{N} \operatorname{sign}(d_i) \quad \text{(excluding ties)}
$$

Equivalently, let $\hat{p} = n_+ / n$ where $n = n_+ + n_-$. The normal approximation gives:

$$
Z = \frac{\hat{p} - 0.5}{\sqrt{0.25 / n}}
$$

### Example Data

The following paired data compare post-treatment and pre-treatment scores for 15 students:

| Student | Post | Pre | Sign | Abs Diff | Rank of Abs Diff |
|:-------:|:----:|:---:|:----:|:--------:|:----------------:|
| 6 | 54 | 54 | 0 | 0 | 1 |
| 13 | 78 | 78 | 0 | 0 | 2 |
| 15 | 76 | 76 | 0 | 0 | 3 |
| 2 | 70 | 72 | − | 2 | 4 |
| 4 | 65 | 68 | − | 3 | 5 |
| 3 | 81 | 75 | + | 6 | 6 |
| 7 | 94 | 88 | + | 6 | 7 |
| 10 | 65 | 57 | + | 8 | 8 |
| 11 | 95 | 86 | + | 9 | 9 |
| 8 | 91 | 81 | + | 10 | 10 |
| 9 | 77 | 65 | + | 12 | 11 |
| 12 | 89 | 87 | + | 12 | 12 |
| 5 | 79 | 65 | + | 14 | 13 |
| 1 | 93 | 76 | + | 17 | 14 |
| 14 | 80 | 77 | + | 17 | 15 |

After excluding the 3 ties: $n_+ = 10$, $n_- = 2$, $n = 12$.

### Implementation

```python
import numpy as np
import scipy.stats as stats

def sign_test(paired_data, test_type="two-sided"):
    """
    Sign test for paired observations.

    Parameters
    ----------
    paired_data : ndarray of shape (n, 2)
        Column 0 is post-treatment, column 1 is pre-treatment.
    test_type : str
        One of "less", "two-sided", "greater".

    Returns
    -------
    z : float
        The Z test statistic.
    p_value : float
    """
    p_0, q_0 = 0.5, 0.5

    # Ties are not counted
    n_plus = np.sum(paired_data[:, 0] > paired_data[:, 1])
    n_minus = np.sum(paired_data[:, 0] < paired_data[:, 1])
    n = n_plus + n_minus
    p_hat = n_plus / n

    z = (p_hat - p_0) / np.sqrt(p_0 * q_0 / n)

    if test_type == "less":
        p_value = stats.norm.cdf(z)
    elif test_type == "two-sided":
        p_value = 2 * stats.norm.cdf(-abs(z))
    elif test_type == "greater":
        p_value = stats.norm.sf(z)

    return z, p_value
```

```python
# Usage
data = np.array([
    [93, 76], [70, 72], [81, 75], [65, 68], [79, 65],
    [54, 54], [94, 88], [91, 81], [77, 65], [65, 57],
    [95, 86], [89, 87], [78, 78], [80, 77], [76, 76]
])
z, p_value = sign_test(data)
print(f"{z       = :.4f}")
print(f"{p_value = :.4f}")
```

### When to Use

The sign test is appropriate when only the direction of change matters (or is measurable) and when differences can only be classified as positive, negative, or tied. If the magnitudes of differences are also meaningful, the **Wilcoxon signed-rank test** (Section 19.1.3) will generally have greater power.

---

## 19.1.3 Wilcoxon Signed-Rank Test

### Concept

The **Wilcoxon signed-rank test** extends the sign test by considering not only the *direction* but also the *magnitude* of each deviation from the hypothesized median. It does so by ranking the absolute deviations and summing ranks for positive and negative differences separately.

!!! note "Not to be confused with"
    The **Wilcoxon rank-sum test** (Section 19.3.1) is for two independent samples. The **signed-rank** test is for one sample or paired data.

### Hypotheses

For paired data with differences $d_i = x_i - y_i$:

$$
H_0: \text{The median of } d_i \text{ is zero}
$$

$$
H_a: \text{The median of } d_i \text{ is not zero (two-sided)}
$$

### Test Statistic

1. Compute differences $d_i = x_i - y_i$.
2. Rank the $|d_i|$ values from smallest to largest.
3. The test statistic is:

$$
T = \sum_{i=1}^{N} \operatorname{sign}(d_i) \times \text{Rank}(|d_i|)
$$

This gives more weight to observations that deviate further from zero, unlike the sign test which treats all non-zero differences equally.

### Handling Ties (Zero Differences)

The `zero_method` parameter controls how zero differences are handled:

| Method | Description |
|:-------|:------------|
| `"wilcox"` | Discard zeros, rank remaining values |
| `"pratt"` | Include zeros in ranking, then exclude from sum |
| `"zsplit"` | Split zeros evenly between positive and negative ranks |

### Example Calculation

Using the same student data from the Sign Test section, with the Pratt method:

$T = (-1)(4) + (-1)(5) + (+1)(6) + (+1)(7) + \cdots + (+1)(14) + (+1)(15)$

The positive ranks dominate, yielding a large positive $T$.

### Implementation with SciPy

```python
import numpy as np
import scipy.stats as stats

data = np.array([
    [93, 76], [70, 72], [81, 75], [65, 68], [79, 65],
    [54, 54], [94, 88], [91, 81], [77, 65], [65, 57],
    [95, 86], [89, 87], [78, 78], [80, 77], [76, 76]
])

statistic, p_value = stats.wilcoxon(
    data[:, 0], data[:, 1],
    alternative="two-sided",
    mode="approx",        # normal approximation
    zero_method="pratt"   # include zeros in ranking
)
print(f"{statistic = }")
print(f"{p_value   = :.4f}")
```

### Sign Test vs Wilcoxon Signed-Rank Test

| Feature | Sign Test | Wilcoxon Signed-Rank |
|:--------|:----------|:---------------------|
| Uses signs | ✓ | ✓ |
| Uses magnitudes | ✗ | ✓ (via ranks) |
| Assumption | None beyond i.i.d. | Symmetric distribution of differences |
| Power | Lower | Higher (when symmetry holds) |
| Robustness to outliers | Very high | High |

The sign test requires only that $P(d_i > 0) = P(d_i < 0)$ under $H_0$, while the Wilcoxon signed-rank test additionally assumes the distribution of differences is symmetric about zero.
