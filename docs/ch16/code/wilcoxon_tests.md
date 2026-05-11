# Wilcoxon Tests

## Overview

The **Wilcoxon tests** are a family of rank-based nonparametric procedures that use the
ordinal information in the data without assuming a specific parametric distribution. The two
most important members are the **Wilcoxon signed-rank test** for paired or one-sample
problems and the **Wilcoxon rank-sum test** (equivalent to the Mann--Whitney $U$ test) for
comparing two independent samples. Both tests are more powerful than the sign test when the
underlying distributions are symmetric, yet remain valid under much weaker conditions than
their parametric counterparts.

## Wilcoxon Signed-Rank Test

### Setup

Given $n$ paired observations $(X_i, Y_i)$, form the differences $D_i = X_i - Y_i$.
Exclude any ties ($D_i = 0$), leaving $n'$ nonzero differences.

1. Rank the absolute values $|D_1|, |D_2|, \dots, |D_{n'}|$ from $1$ to $n'$.
2. Attach the sign of $D_i$ to each rank, producing **signed ranks**
   $R_i^{+} = \operatorname{rank}(|D_i|) \cdot \operatorname{sgn}(D_i)$.
3. Compute the test statistic

$$
W^{+} = \sum_{i:\, D_i > 0} \operatorname{rank}(|D_i|).
$$

Under $H_0{:}\;\text{median}(D) = 0$ with symmetric differences, each signed rank is
equally likely to be positive or negative, so

$$
\operatorname{E}[W^{+}] = \frac{n'(n'+1)}{4}, \qquad
\operatorname{Var}(W^{+}) = \frac{n'(n'+1)(2n'+1)}{24}.
$$

The standardized statistic

$$
Z = \frac{W^{+} - \operatorname{E}[W^{+}]}{\sqrt{\operatorname{Var}(W^{+})}}
$$

is approximately standard normal for moderate $n'$.

### Implementation

```python
import numpy as np
from scipy import stats

paired_data = np.array([
    [93, 76], [70, 72], [81, 75], [65, 68], [79, 65],
    [54, 54], [94, 88], [91, 81], [77, 65], [65, 57],
    [95, 86], [89, 87], [78, 78], [80, 77], [76, 76]
])

statistic, p_value = stats.wilcoxon(
    paired_data[:, 0], paired_data[:, 1],
    alternative="two-sided",
    mode="approx",
    zero_method="pratt"
)
print(f"W+ = {statistic}, p = {p_value:.4f}")
```

## Wilcoxon Rank-Sum Test

### Setup

Given two independent samples $X_1, \dots, X_m$ and $Y_1, \dots, Y_n$, pool and rank all
$N = m + n$ observations together. Let $W$ be the sum of ranks assigned to the first sample.

Under $H_0$: the two populations have identical distributions,

$$
\operatorname{E}[W] = \frac{m(N + 1)}{2}, \qquad
\operatorname{Var}(W) = \frac{m\,n\,(N + 1)}{12}.
$$

The standardized rank-sum statistic

$$
Z = \frac{W - \operatorname{E}[W]}{\sqrt{\operatorname{Var}(W)}}
$$

is asymptotically standard normal.

### Implementation

```python
from scipy import stats

x = [93, 70, 81, 65, 79, 54, 94, 91, 77, 65, 95, 89, 78, 80, 76]
y = [76, 72, 75, 68, 65, 54, 88, 81, 65, 57, 86, 87, 78, 77, 76]

statistic, p_value = stats.ranksums(x, y, alternative="two-sided")
print(f"Z = {statistic:.4f}, p = {p_value:.4f}")
```

## Relationship to the Mann--Whitney U

The Mann--Whitney $U$ statistic counts the number of pairs $(X_i, Y_j)$ where $X_i > Y_j$.
It is related to the rank-sum by

$$
U = W - \frac{m(m+1)}{2},
$$

so the two tests are algebraically equivalent and always produce the same $p$-value.

## Interpretation

| Test | Null Hypothesis | Key Assumption |
|---|---|---|
| Signed-rank | Median difference is zero | Differences are **symmetric** about zero |
| Rank-sum | Two populations are identical | Observations are **independent** across groups |

- **Signed-rank vs. sign test**: The signed-rank test uses both the sign and the rank of each
  difference, giving it greater power when the symmetry assumption holds.
- **Rank-sum vs. two-sample $t$-test**: The rank-sum test is robust to outliers and
  heavy-tailed distributions but slightly less powerful than the $t$-test under exact
  normality.

## Exercises

**Exercise 1.** Six subjects produce the following paired differences:
$D = (4, -1, 7, 3, -2, 5)$. Compute the signed-rank statistic $W^+$ by hand.

??? success "Solution to Exercise 1"

    Rank the absolute values:

    | $D_i$ | $\lvert D_i \rvert$ | Rank | Sign |
    |---|---|---|---|
    | $-1$ | $1$ | $1$ | $-$ |
    | $-2$ | $2$ | $2$ | $-$ |
    | $3$  | $3$ | $3$ | $+$ |
    | $4$  | $4$ | $4$ | $+$ |
    | $5$  | $5$ | $5$ | $+$ |
    | $7$  | $7$ | $6$ | $+$ |

    $$
    W^+ = 3 + 4 + 5 + 6 = 18.
    $$

    The maximum possible value is $6 \cdot 7 / 2 = 21$, so $W^+ = 18$ out of $21$
    suggests a strong positive shift. $\square$

---

**Exercise 2.** Show that $\operatorname{E}[W^+] = n'(n'+1)/4$ under $H_0$.

??? success "Solution to Exercise 2"

    Under the null hypothesis, each difference $D_i$ is equally likely to be positive or
    negative (by the symmetry assumption). Therefore the signed rank for observation $i$
    with rank $r_i$ contributes $r_i$ to $W^+$ with probability $1/2$ and $0$ otherwise.

    $$
    \operatorname{E}[W^+]
      = \sum_{i=1}^{n'} r_i \cdot \frac{1}{2}
      = \frac{1}{2} \sum_{i=1}^{n'} i
      = \frac{1}{2} \cdot \frac{n'(n'+1)}{2}
      = \frac{n'(n'+1)}{4}.
    $$

    $\square$

---

**Exercise 3.** Two independent groups have the following values:

- Group A: $12, 15, 18, 22, 25$
- Group B: $8, 10, 14, 19, 21, 24$

Perform the Wilcoxon rank-sum test at $\alpha = 0.05$ using Python.

??? success "Solution to Exercise 3"

    ```python
    from scipy import stats

    a = [12, 15, 18, 22, 25]
    b = [8, 10, 14, 19, 21, 24]

    stat, p = stats.ranksums(a, b)
    print(f"Z = {stat:.4f}, p = {p:.4f}")
    ```

    Pooling and ranking the 11 values: $8(1), 10(2), 12(3), 14(4), 15(5), 18(6),
    19(7), 21(8), 22(9), 24(10), 25(11)$.

    Group A ranks: $3 + 5 + 6 + 9 + 11 = 34$.
    Expected: $5 \cdot 12 / 2 = 30$.

    The $p$-value is well above $0.05$, so we fail to reject the null hypothesis that
    the two groups come from the same distribution. $\square$

---

**Exercise 4.** Explain why the Wilcoxon signed-rank test requires the assumption that
the distribution of differences is symmetric, while the sign test does not. Give an example
of a distribution where this distinction matters.

??? success "Solution to Exercise 4"

    The signed-rank test assigns rank magnitudes to each observation and then uses the
    symmetry assumption to conclude that each signed rank is equally likely to be positive
    or negative *independently*. If the distribution of $|D_i|$ differs between positive
    and negative differences (i.e., the distribution is skewed), the null distribution of
    $W^+$ is no longer the one derived under symmetry.

    The sign test only uses $\operatorname{sgn}(D_i)$, so it requires only that
    $P(D_i > 0) = P(D_i < 0) = 0.5$ under $H_0$---a property that holds for any
    continuous distribution with median zero, symmetric or not.

    **Example**: Suppose $D_i$ follows an Exponential(1) distribution shifted to have
    median zero: $D_i \sim \text{Exp}(1) - \ln 2$. This distribution is right-skewed.
    The sign test is valid (the median is zero), but the signed-rank test's null
    distribution is incorrect because the symmetry assumption fails. $\square$

---

**Exercise 5.** Derive the relationship $U = W - m(m+1)/2$ between the Mann--Whitney $U$
statistic and the Wilcoxon rank-sum $W$.

??? success "Solution to Exercise 5"

    Let $R_1, R_2, \dots, R_m$ be the ranks of the $m$ observations from sample $X$ in
    the combined ranking of all $N = m + n$ observations, so $W = \sum_{i=1}^{m} R_i$.

    The Mann--Whitney $U$ counts pairs:

    $$
    U = \sum_{i=1}^{m} \sum_{j=1}^{n} \mathbf{1}[X_i > Y_j].
    $$

    For a fixed $X_i$ with rank $R_i$ among all $N$ observations, the number of
    $Y_j$ values smaller than $X_i$ equals the number of observations ranked below
    $R_i$ that come from sample $Y$, which is $R_i - (\text{number of } X\text{'s ranked}
    \leq R_i)$. Summing over all $X_i$:

    $$
    U = \sum_{i=1}^{m} \bigl(R_i - i\bigr) = W - \sum_{i=1}^{m} i = W - \frac{m(m+1)}{2}.
    $$

    $\square$
