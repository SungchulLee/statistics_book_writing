# Spearman Rank Correlation

The Pearson correlation captures linear relationships, but many real-world associations are monotonic without being strictly linear. For instance, the relationship between years of experience and salary may be consistently increasing but not along a straight line. **Spearman's rank correlation coefficient** measures the strength and direction of *monotonic* relationships by applying the Pearson formula to the ranks of the data rather than the raw values.

---

## From Values to Ranks

The key idea behind Spearman's correlation is simple: replace each observation with its rank, then compute the Pearson correlation on those ranks. This makes the measure robust to outliers and invariant to any monotone transformation of the data.

Given paired observations $(x_1, y_1), \ldots, (x_n, y_n)$:

1. Rank the $x$-values from smallest to largest, assigning rank $R(x_i)$ to each $x_i$.
2. Rank the $y$-values similarly, assigning rank $R(y_i)$ to each $y_i$.
3. Compute the Pearson correlation between the ranks $R(x_i)$ and $R(y_i)$.

When there are tied values, each tied observation receives the average of the ranks it would have occupied.

---

## Definition

**Spearman's rank correlation coefficient** $r_s$ is

$$
r_s = \frac{\sum_{i=1}^n (R(x_i) - \bar{R}_x)(R(y_i) - \bar{R}_y)}{\sqrt{\sum_{i=1}^n (R(x_i) - \bar{R}_x)^2} \; \sqrt{\sum_{i=1}^n (R(y_i) - \bar{R}_y)^2}}
$$

where $\bar{R}_x$ and $\bar{R}_y$ are the mean ranks. Since the ranks are simply $1, 2, \ldots, n$ (when there are no ties), the mean rank is $\bar{R} = (n+1)/2$.

### Shortcut Formula (No Ties)

When there are no tied ranks, the formula simplifies to

$$
r_s = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n(n^2 - 1)}
$$

where $d_i = R(x_i) - R(y_i)$ is the difference between the ranks of the $i$-th pair. This shortcut formula is widely used for hand calculation.

---

## Properties

1. **Range.** $-1 \le r_s \le 1$, just like the Pearson coefficient.

2. **Perfect monotonic relationship.** $r_s = 1$ if and only if $R(x_i) = R(y_i)$ for all $i$ (the ranks are identical). This means $Y$ is a perfectly increasing function of $X$, though not necessarily linear. Similarly, $r_s = -1$ when the ranks are perfectly reversed.

3. **Invariance under monotone transformations.** If $f$ is any strictly increasing function, then $r_s(f(X), Y) = r_s(X, Y)$. This means Spearman's $r_s$ is unchanged by log transforms, square roots, or any other order-preserving transformation.

4. **Robustness to outliers.** Because ranks compress extreme values, a single outlier has limited influence on $r_s$.

---

## When to Use Spearman vs Pearson

| Criterion | Pearson $r$ | Spearman $r_s$ |
|:---|:---|:---|
| Relationship type | Linear | Monotonic |
| Data scale | Interval or ratio | Ordinal, interval, or ratio |
| Sensitivity to outliers | High | Low |
| Distribution assumption | Best with bivariate normal | Distribution-free |
| Interpretation | Strength of linear association | Strength of monotonic association |

Use Spearman when:

- The relationship is monotonic but not linear (e.g., exponential, logarithmic).
- The data contain outliers or are heavily skewed.
- The variables are measured on an ordinal scale (e.g., Likert ratings).

Use Pearson when:

- The relationship is approximately linear.
- Both variables are continuous and roughly normally distributed.
- You want a measure specifically tied to linear prediction.

---

## Example: Monotonic but Nonlinear Relationship

Consider the relationship $Y = e^X$ for $X = 1, 2, \ldots, 8$. The relationship is perfectly monotonic (increasing) but nonlinear. Spearman's $r_s$ captures this perfectly, while Pearson's $r$ will be less than $1$.

| $x_i$ | $y_i = e^{x_i}$ | $R(x_i)$ | $R(y_i)$ | $d_i$ |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.72 | 1 | 1 | 0 |
| 2 | 7.39 | 2 | 2 | 0 |
| 3 | 20.09 | 3 | 3 | 0 |
| 4 | 54.60 | 4 | 4 | 0 |
| 5 | 148.41 | 5 | 5 | 0 |
| 6 | 403.43 | 6 | 6 | 0 |
| 7 | 1096.63 | 7 | 7 | 0 |
| 8 | 2980.96 | 8 | 8 | 0 |

Since $d_i = 0$ for all $i$, the shortcut formula gives

$$
r_s = 1 - \frac{6 \cdot 0}{8(64 - 1)} = 1
$$

The Pearson $r$ for this data is approximately $0.76$ because the exponential curve departs substantially from a straight line.

---

## Handling Ties

When tied values occur, the shortcut formula is no longer exact. The standard approach is:

1. Assign each tied observation the **average rank** (midrank).
2. Use the full formula (Pearson correlation on ranks) rather than the shortcut.

??? example "Tied ranks illustration"
    Suppose the $x$-values are $\{3, 5, 5, 7, 9\}$. The values 5 and 5 would occupy ranks 2 and 3, so each receives the average rank $(2 + 3)/2 = 2.5$. The final ranks are $\{1, 2.5, 2.5, 4, 5\}$.

---

## Computation in Python

```python
import numpy as np
from scipy import stats

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.exp(x)

# Spearman correlation
r_s, p_value = stats.spearmanr(x, y)
print(f"Spearman r_s = {r_s:.4f}, p-value = {p_value:.6f}")

# Compare with Pearson
r_p, p_p = stats.pearsonr(x, y)
print(f"Pearson  r   = {r_p:.4f}, p-value = {p_p:.6f}")
```

The `scipy.stats.spearmanr` function handles ties automatically using midranks. For hypothesis testing details, see [Testing Spearman's rho](../correlation_test/test_spearman.md).

---

## Summary

Spearman's rank correlation $r_s$ measures the strength and direction of any monotonic relationship between two variables. By operating on ranks rather than raw values, it is robust to outliers and applicable to ordinal data. While Pearson's $r$ is optimal for linear relationships with normally distributed data, Spearman's $r_s$ is the preferred choice when the relationship is monotonic but nonlinear, when the data contain outliers, or when the measurement scale is ordinal.

## Exercises

**Exercise 1.**
Compute Spearman's rank correlation for $X = (10, 20, 30, 40, 50)$ and $Y = (15, 25, 5, 35, 45)$.

??? success "Solution to Exercise 1"
    Assign ranks: $R_X = (1, 2, 3, 4, 5)$, $R_Y = (2, 3, 1, 4, 5)$.

    Compute rank differences $d_i = R_{X,i} - R_{Y,i}$: $d = (-1, -1, 2, 0, 0)$.

    $$
    \sum d_i^2 = 1 + 1 + 4 + 0 + 0 = 6
    $$

    Using the shortcut formula:

    $$
    r_s = 1 - \frac{6\sum d_i^2}{n(n^2 - 1)} = 1 - \frac{6 \times 6}{5 \times 24} = 1 - \frac{36}{120} = 1 - 0.3 = 0.7
    $$

---

**Exercise 2.**
Explain why Spearman's $r_s$ is exactly the Pearson correlation of the ranks. Why does this make $r_s$ robust to outliers?

??? success "Solution to Exercise 2"
    By definition, $r_s = r(\text{rank}(X), \text{rank}(Y))$: the Pearson correlation applied to the rank-transformed data. The shortcut formula $1 - 6\sum d_i^2/[n(n^2-1)]$ is algebraically equivalent (when there are no ties).

    This makes $r_s$ robust to outliers because ranking discards information about the magnitude of observations. An extreme outlier (e.g., changing $X_5 = 50$ to $X_5 = 5000$) does not change its rank (still rank 5), so $r_s$ is unchanged. In contrast, Pearson's $r$ would be heavily influenced because it uses the actual values in computing covariances and variances.

---

**Exercise 3.**
Data: $X = (1, 2, 3, 4, 5)$, $Y = (1, 4, 9, 16, 25)$ (i.e., $Y = X^2$). Compute Pearson's $r$ and Spearman's $r_s$. Why do they differ?

??? success "Solution to Exercise 3"
    **Spearman's $r_s$:** Since $Y = X^2$ is a strictly increasing function on positive $X$, the ranks of $Y$ are identical to the ranks of $X$. Therefore $r_s = 1.0$.

    **Pearson's $r$:** $\bar{X} = 3$, $\bar{Y} = 11$. Computing:

    $$
    \sum(X_i - 3)(Y_i - 11) = (-2)(-10) + (-1)(-7) + (0)(-2) + (1)(5) + (2)(14) = 20 + 7 + 0 + 5 + 28 = 60
    $$

    $$
    \sum(X_i-3)^2 = 10, \quad \sum(Y_i - 11)^2 = 100 + 49 + 4 + 25 + 196 = 374
    $$

    $$
    r = \frac{60}{\sqrt{10 \times 374}} = \frac{60}{61.16} \approx 0.981
    $$

    Pearson's $r \approx 0.98 < 1$ because it measures linear association, and $Y = X^2$ is nonlinear (though nearly linear in this range). Spearman's $r_s = 1$ because the relationship is perfectly monotonic.

---

**Exercise 4.**
Under what conditions does Spearman's $r_s$ equal Pearson's $r$ exactly?

??? success "Solution to Exercise 4"
    Spearman's $r_s$ equals Pearson's $r$ when the ranks are a linear function of the original values, which happens when:

    1. **The data have no ties and the values are equally spaced** (or more generally, when both $X$ and $Y$ have the same distribution of values such that ranking is a linear transformation).

    2. **Both variables are uniformly distributed** on their range (ranks are proportional to the values).

    In practice, $r_s \approx r$ when the relationship between $X$ and $Y$ is approximately linear and neither variable has extreme outliers. The two measures diverge when (a) the relationship is nonlinear but monotonic ($r_s > r$), (b) outliers are present ($r$ is distorted, $r_s$ is not), or (c) the relationship is linear but with outliers ($r$ may be pulled toward zero, $r_s$ remains stable).
