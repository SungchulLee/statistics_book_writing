# Two-Sample and Multi-Group Tests

## Overview

When comparing two or more groups without assuming normality, several rank-based and
median-based nonparametric procedures are available. This page covers the **Wilcoxon
rank-sum test**, the **Wilcoxon signed-rank test**, the **Mann--Whitney $U$ test**, the
**Kruskal--Wallis $H$ test**, and **Mood's median test**. Each test converts raw observations
into ranks or binary indicators relative to the grand median, yielding distribution-free
inference that is robust to outliers and non-normal data.

## Wilcoxon Rank-Sum Test

The rank-sum test compares two **independent** samples by pooling all $N = m + n$
observations, ranking them from $1$ to $N$, and summing the ranks of one sample.

Under $H_0$ (identical distributions), the rank-sum $W$ for sample 1 satisfies

$$
\operatorname{E}[W] = \frac{m(N+1)}{2}, \qquad
\operatorname{Var}(W) = \frac{m\,n\,(N+1)}{12}.
$$

The standardized statistic

$$
Z = \frac{W - \operatorname{E}[W]}{\sqrt{\operatorname{Var}(W)}}
$$

is compared to the standard normal distribution.

```python
from scipy import stats

post = [93, 70, 81, 65, 79, 54, 94, 91, 77, 65, 95, 89, 78, 80, 76]
pre  = [76, 72, 75, 68, 65, 54, 88, 81, 65, 57, 86, 87, 78, 77, 76]

stat, p = stats.ranksums(post, pre, alternative="two-sided")
print(f"Z = {stat:.4f}, p = {p:.2%}")
```

## Wilcoxon Signed-Rank Test

For **paired** data the signed-rank test ranks the absolute differences $|D_i|$,
attaches the original signs, and sums the positive signed ranks to obtain $W^+$.

Under $H_0{:}\;\text{median}(D) = 0$ with symmetric differences,

$$
\operatorname{E}[W^+] = \frac{n'(n'+1)}{4}, \qquad
\operatorname{Var}(W^+) = \frac{n'(n'+1)(2n'+1)}{24},
$$

where $n'$ is the number of nonzero differences.

```python
from scipy import stats
import numpy as np

paired_data = np.array([
    [93, 76], [70, 72], [81, 75], [65, 68], [79, 65],
    [54, 54], [94, 88], [91, 81], [77, 65], [65, 57],
    [95, 86], [89, 87], [78, 78], [80, 77], [76, 76]
])

stat, p = stats.wilcoxon(
    paired_data[:, 0], paired_data[:, 1],
    alternative="two-sided", mode="approx", zero_method="pratt"
)
print(f"W+ = {stat}, p = {p:.4f}")
```

## Mann--Whitney U Test

The Mann--Whitney $U$ statistic counts the number of pairs $(X_i, Y_j)$ where one
observation exceeds the other:

$$
U = \sum_{i=1}^{m} \sum_{j=1}^{n} \mathbf{1}[X_i > Y_j].
$$

It is related to the rank-sum by $U = W - m(m+1)/2$, so the two tests are equivalent.
SciPy's implementation handles **ties** via a continuity-corrected normal approximation.

```python
from scipy import stats

data0 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
data1 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]

stat, p = stats.mannwhitneyu(data0, data1)
print(f"U = {stat}, p = {p:.2%}")
```

## Kruskal--Wallis H Test

The Kruskal--Wallis test extends the rank-sum idea to $k \geq 2$ independent groups. Pool
all $N$ observations, rank them, and compute

$$
H = \frac{12}{N(N+1)} \sum_{j=1}^{k} \frac{R_j^2}{n_j} - 3(N+1),
$$

where $R_j$ is the sum of ranks in group $j$ and $n_j$ is the group size. Under $H_0$
(all groups come from the same population), $H$ is approximately
$\chi^2_{k-1}$-distributed.

```python
from scipy import stats

data0 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
data1 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]
data2 = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84]

stat, p = stats.kruskal(data0, data1, data2)
print(f"H = {stat:.4f}, p = {p:.2%}")
```

## Mood's Median Test

Mood's median test is a simpler alternative to Kruskal--Wallis. It computes the **grand
median** of all observations, classifies each observation as above or below that median, and
forms a $2 \times k$ contingency table. A chi-squared test on this table assesses whether
the groups have the same median.

```python
from scipy import stats

data0 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
data1 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]
data2 = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84]

result = stats.median_test(data0, data1, data2)
print(f"Grand median = {result.median}")
print(f"Contingency table:\n{result.table}")
print(f"p-value = {result.pvalue:.4f}")
```

## Interpretation

| Test | Use Case | Null Hypothesis | Distribution of Statistic |
|---|---|---|---|
| Rank-sum | Two independent samples | Identical distributions | Normal (large sample) |
| Signed-rank | Paired samples | Median difference is zero | Normal (large sample) |
| Mann--Whitney $U$ | Two independent samples (with ties) | $P(X > Y) = 0.5$ | Normal (large sample) |
| Kruskal--Wallis | $k$ independent groups | All groups from same population | $\chi^2_{k-1}$ |
| Mood's median | $k$ independent groups | All groups share the same median | $\chi^2_{k-1}$ |

**Choosing among the tests**:

- For two independent samples without ties, the rank-sum and Mann--Whitney tests are
  equivalent; use Mann--Whitney when ties are present.
- For paired data, use the signed-rank test (requires symmetric differences) or the sign
  test (no symmetry needed).
- For three or more groups, Kruskal--Wallis is generally more powerful than Mood's median
  test, but Mood's test is simpler and more robust to outliers.

## Exercises

**Exercise 1.** Two groups of students take different versions of an exam. Group A scores
are $(72, 78, 81, 85, 90)$ and Group B scores are $(68, 74, 77, 83, 88, 92)$. Perform the
Wilcoxon rank-sum test by hand: pool, rank, compute $W$ for Group A, and find the $Z$
statistic.

??? success "Solution to Exercise 1"

    Pool and rank all $N = 11$ values:

    | Value | 68 | 72 | 74 | 77 | 78 | 81 | 83 | 85 | 88 | 90 | 92 |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | Rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
    | Group | B | A | B | B | A | A | B | A | B | A | B |

    For Group A ($m = 5$): $W = 2 + 5 + 6 + 8 + 10 = 31$.

    $$
    \operatorname{E}[W] = \frac{5 \cdot 12}{2} = 30, \qquad
    \operatorname{Var}(W) = \frac{5 \cdot 6 \cdot 12}{12} = 30.
    $$

    $$
    Z = \frac{31 - 30}{\sqrt{30}} \approx 0.183.
    $$

    The two-sided $p$-value is $2\mathcal{N}(-0.183) \approx 0.855$. We fail to reject
    $H_0$---no significant difference between the groups. $\square$

---

**Exercise 2.** Three treatments are applied to independent groups with the following
outcomes:

- Treatment 1: $14, 18, 22, 25$
- Treatment 2: $19, 23, 27, 30, 35$
- Treatment 3: $10, 15, 20$

Compute the Kruskal--Wallis $H$ statistic by hand.

??? success "Solution to Exercise 2"

    Pool and rank all $N = 12$ values:

    | Value | 10 | 14 | 15 | 18 | 19 | 20 | 22 | 23 | 25 | 27 | 30 | 35 |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|
    | Rank | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
    | Group | 3 | 1 | 3 | 1 | 2 | 3 | 1 | 2 | 1 | 2 | 2 | 2 |

    Rank sums: $R_1 = 2+4+7+9 = 22$, $R_2 = 5+8+10+11+12 = 46$, $R_3 = 1+3+6 = 10$.

    $$
    H = \frac{12}{12 \cdot 13}\left(\frac{22^2}{4} + \frac{46^2}{5} + \frac{10^2}{3}\right) - 3 \cdot 13.
    $$

    $$
    H = \frac{12}{156}\left(121 + 423.2 + 33.33\right) - 39
      = \frac{12 \cdot 577.53}{156} - 39
      \approx 44.42 - 39
      = 5.42.
    $$

    With $k - 1 = 2$ degrees of freedom, $P(\chi^2_2 > 5.42) \approx 0.067$.
    At $\alpha = 0.05$ we would fail to reject, though the result is suggestive. $\square$

---

**Exercise 3.** Mood's median test converts each observation into a binary indicator
(above or below the grand median). Explain why this approach is less powerful than
Kruskal--Wallis but more robust to extreme outliers.

??? success "Solution to Exercise 3"

    Mood's median test reduces each observation to a single bit of information---whether
    it exceeds the grand median or not. This means it ignores *how far* each observation
    is from the median, discarding ordinal information that ranks preserve. Since
    Kruskal--Wallis uses full rank information, it captures more of the distributional
    shift and therefore has greater power.

    However, this same information reduction makes Mood's test more robust. An extreme
    outlier (e.g., a value of $10{,}000$ in a dataset with median $30$) receives the
    same binary code ("above") as a moderate value of $35$. In Kruskal--Wallis, the
    outlier's extreme rank can inflate the rank sum for its group, potentially distorting
    results. In Mood's test, the outlier has no more influence than any other above-median
    observation. $\square$

---

**Exercise 4.** Using the Mann--Whitney $U$ data from the code example above, verify by
hand that $U = W - m(m+1)/2$, where $W$ is the rank-sum for `data0` and $m = 16$.

??? success "Solution to Exercise 4"

    The combined sample has $N = 16 + 15 = 31$ observations. After pooling and ranking,
    the sum of ranks for `data0` is $W$.

    Using SciPy:

    ```python
    from scipy import stats

    data0 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
    data1 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]

    U, p = stats.mannwhitneyu(data0, data1)
    print(f"U = {U}")

    # Compute rank-sum W for data0
    import numpy as np
    combined = np.array(data0 + data1)
    ranks = stats.rankdata(combined)
    W = ranks[:16].sum()
    print(f"W = {W}")
    print(f"W - m(m+1)/2 = {W - 16*17/2}")
    ```

    We can verify that `U` matches `W - 16*17/2 = W - 136`. The rank-sum and the
    Mann--Whitney statistic carry the same information, confirming their equivalence.
    $\square$

---

**Exercise 5.** Prove that the Kruskal--Wallis $H$ statistic reduces to the square of the
rank-sum $Z$ statistic when $k = 2$.

??? success "Solution to Exercise 5"

    With $k = 2$ groups of sizes $m$ and $n = N - m$, the rank sums satisfy
    $R_1 + R_2 = N(N+1)/2$. The Kruskal--Wallis statistic is

    $$
    H = \frac{12}{N(N+1)}\left(\frac{R_1^2}{m} + \frac{R_2^2}{n}\right) - 3(N+1).
    $$

    Substituting $R_2 = N(N+1)/2 - R_1$ and simplifying (set $W = R_1$):

    $$
    H = \frac{12}{N(N+1)} \cdot \frac{n\,W^2 + m\bigl(\tfrac{N(N+1)}{2} - W\bigr)^2}{m\,n} - 3(N+1).
    $$

    After expanding and collecting terms, the cross terms cancel and we obtain

    $$
    H = \frac{\bigl(W - m(N+1)/2\bigr)^2}{m\,n\,(N+1)/12} = Z^2,
    $$

    where $Z = (W - \operatorname{E}[W])/\sqrt{\operatorname{Var}(W)}$ is the rank-sum
    test statistic. This confirms that the $\chi^2_1$ distribution of $H$ is the square
    of the standard normal distribution of $Z$. $\square$
