# Kendall's Tau

Like Spearman's $r_s$, Kendall's tau is a rank-based measure of association between two variables. However, instead of correlating the ranks themselves, Kendall's tau counts the number of **concordant** and **discordant** pairs among the observations. This pair-counting approach gives Kendall's tau a more direct probabilistic interpretation and often better statistical properties for small samples.

---

## Concordant and Discordant Pairs

Given $n$ paired observations $(x_1, y_1), \ldots, (x_n, y_n)$, consider any pair of observations $(x_i, y_i)$ and $(x_j, y_j)$ with $i < j$.

- The pair is **concordant** if $x_i < x_j$ and $y_i < y_j$, or $x_i > x_j$ and $y_i > y_j$. In other words, the two variables rank the pair in the same order.

- The pair is **discordant** if $x_i < x_j$ and $y_i > y_j$, or $x_i > x_j$ and $y_i < y_j$. The two variables rank the pair in opposite order.

- The pair is **tied** if $x_i = x_j$ or $y_i = y_j$ (or both).

The total number of pairs is $\binom{n}{2} = \frac{n(n-1)}{2}$.

---

## Kendall's Tau-a

The simplest version, **Kendall's tau-a**, is defined as

$$
\tau_a = \frac{C - D}{\binom{n}{2}} = \frac{C - D}{\frac{n(n-1)}{2}}
$$

where $C$ is the number of concordant pairs and $D$ is the number of discordant pairs.

Tau-a does not make any adjustment for ties. When ties are present, $\tau_a$ cannot reach $\pm 1$ even for a perfectly monotonic relationship, because tied pairs are neither concordant nor discordant.

---

## Kendall's Tau-b

To handle ties, **Kendall's tau-b** adjusts the denominator:

$$
\tau_b = \frac{C - D}{\sqrt{(C + D + T_X)(C + D + T_Y)}}
$$

where:

- $T_X$ = number of pairs tied on $X$ but not on $Y$
- $T_Y$ = number of pairs tied on $Y$ but not on $X$

Tau-b can attain $\pm 1$ when the data are perfectly monotonic, even in the presence of ties (as long as ties exist on at most one variable). It is the version most commonly reported in practice and the default in most statistical software.

---

## Probabilistic Interpretation

Kendall's tau has an elegant probabilistic interpretation. For two randomly chosen observation pairs $(x_i, y_i)$ and $(x_j, y_j)$:

$$
\tau = P(\text{concordant}) - P(\text{discordant})
$$

This means:

- $\tau = 1$: every pair is concordant (perfect agreement in ordering).
- $\tau = -1$: every pair is discordant (perfect reversal).
- $\tau = 0$: concordant and discordant pairs are equally likely (no monotonic association).

---

## Properties

1. **Range.** $-1 \le \tau \le 1$ for both tau-a and tau-b (tau-a may not achieve the bounds when ties exist).

2. **Symmetry.** $\tau_{XY} = \tau_{YX}$.

3. **Invariance under monotone transformations.** Like Spearman's $r_s$, Kendall's tau depends only on the ordering of observations, not their numerical values.

4. **Robustness.** Kendall's tau is robust to outliers since it depends on relative orderings.

5. **Magnitude.** For the same data, $|\tau|$ is typically smaller than $|r_s|$. A rough conversion is $\tau \approx \frac{2}{\pi} \arcsin(r_s)$, though this is only exact for bivariate normal data.

---

## Example Calculation

Consider the five paired observations:

| $i$ | $x_i$ | $y_i$ |
|:---:|:---:|:---:|
| 1 | 1 | 3 |
| 2 | 2 | 5 |
| 3 | 3 | 4 |
| 4 | 4 | 2 |
| 5 | 5 | 1 |

There are $\binom{5}{2} = 10$ pairs. Sorting by $x$ (already sorted), we compare each pair:

| Pair $(i,j)$ | $x$ order | $y$ order | Result |
|:---:|:---:|:---:|:---:|
| (1,2) | $1 < 2$ | $3 < 5$ | Concordant |
| (1,3) | $1 < 3$ | $3 < 4$ | Concordant |
| (1,4) | $1 < 4$ | $3 > 2$ | Discordant |
| (1,5) | $1 < 5$ | $3 > 1$ | Discordant |
| (2,3) | $2 < 3$ | $5 > 4$ | Discordant |
| (2,4) | $2 < 4$ | $5 > 2$ | Discordant |
| (2,5) | $2 < 5$ | $5 > 1$ | Discordant |
| (3,4) | $3 < 4$ | $4 > 2$ | Discordant |
| (3,5) | $3 < 5$ | $4 > 1$ | Discordant |
| (4,5) | $4 < 5$ | $2 > 1$ | Concordant |

We have $C = 3$ concordant and $D = 7$ discordant pairs. Since there are no ties:

$$
\tau_a = \tau_b = \frac{3 - 7}{10} = -0.4
$$

The negative value indicates a (moderate) tendency for $Y$ to decrease as $X$ increases.

---

## Kendall vs Spearman

| Feature | Kendall $\tau$ | Spearman $r_s$ |
|:---|:---|:---|
| Basis | Concordant/discordant pairs | Rank correlation |
| Typical magnitude | Smaller in absolute value | Larger in absolute value |
| Small-sample behavior | Better variance properties | More variable |
| Ties handling | Tau-b adjusts explicitly | Midranks |
| Probabilistic interpretation | Direct: $P(C) - P(D)$ | Indirect |
| Computational cost | $O(n \log n)$ with merge sort | $O(n \log n)$ for ranking |

For large samples without ties, both tests have similar statistical power. Kendall's tau is often preferred in small samples or when a clear probabilistic interpretation is desired.

---

## Computation in Python

```python
import numpy as np
from scipy import stats

x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 4, 2, 1])

# Kendall's tau-b (default in SciPy)
tau, p_value = stats.kendalltau(x, y)
print(f"Kendall tau-b = {tau:.4f}, p-value = {p_value:.4f}")

# Compare with Spearman
r_s, p_s = stats.spearmanr(x, y)
print(f"Spearman r_s  = {r_s:.4f}, p-value = {p_s:.4f}")
```

The `scipy.stats.kendalltau` function computes tau-b by default. For hypothesis testing details, see [Testing Kendall's tau](../correlation_test/test_kendall.md).

---

## Summary

Kendall's tau measures monotonic association by comparing concordant and discordant pairs. The tau-b variant adjusts for ties and is the standard choice in practice. Compared to Spearman's $r_s$, Kendall's tau tends to be smaller in absolute value for the same data but offers a direct probabilistic interpretation and better small-sample properties. Both are robust, rank-based alternatives to Pearson's $r$ for nonlinear monotonic relationships.

## Exercises

**Exercise 1.**
Compute Kendall's $\tau$ for the data: $X = (1, 2, 3, 4, 5)$, $Y = (2, 4, 1, 3, 5)$.

??? success "Solution to Exercise 1"
    Count concordant and discordant pairs among all $\binom{5}{2} = 10$ pairs:

    | Pair $(i,j)$ | $X_j - X_i$ | $Y_j - Y_i$ | Concordant? |
    |---|---|---|---|
    | (1,2) | + | + | C |
    | (1,3) | + | $-$ | D |
    | (1,4) | + | + | C |
    | (1,5) | + | + | C |
    | (2,3) | + | $-$ | D |
    | (2,4) | + | $-$ | D |
    | (2,5) | + | + | C |
    | (3,4) | + | + | C |
    | (3,5) | + | + | C |
    | (4,5) | + | + | C |

    Concordant: $C = 7$, Discordant: $D = 3$.

    $$
    \tau = \frac{C - D}{\binom{n}{2}} = \frac{7 - 3}{10} = 0.4
    $$

---

**Exercise 2.**
Explain the conceptual difference between Kendall's $\tau$ and Spearman's $\rho$, even though both measure monotonic association.

??? success "Solution to Exercise 2"
    **Kendall's $\tau$** counts the proportion of concordant minus discordant pairs among all pairs of observations. It has a direct probabilistic interpretation: $\tau = P(\text{concordant}) - P(\text{discordant})$ for a randomly chosen pair.

    **Spearman's $\rho$** is the Pearson correlation applied to the ranks. It measures the linear association between ranks and is sensitive to the magnitude of rank differences.

    Key differences: (1) $\tau$ is based on pairwise comparisons (ordinal), while $\rho$ uses actual rank values; (2) $\tau$ tends to be smaller in absolute value than $\rho$ for the same data; (3) $\tau$ has simpler asymptotic properties and its standard error is easier to compute; (4) $\rho$ is more powerful for detecting linear rank relationships, while $\tau$ is more robust.

---

**Exercise 3.**
Why is Kendall's $\tau$ preferred over Pearson's $r$ when data contain outliers or have a nonlinear but monotonic relationship?

??? success "Solution to Exercise 3"
    Kendall's $\tau$ is based only on the ordinal pattern (which observation is larger), not on the actual values. This makes it:

    1. **Robust to outliers:** A single extreme value changes at most $n - 1$ pairwise comparisons (out of $\binom{n}{2}$), and only if it changes the ordering. Pearson's $r$, which uses squared deviations, is heavily influenced by a single extreme point.

    2. **Appropriate for nonlinear monotonic relationships:** If $Y$ is a monotonically increasing but nonlinear function of $X$ (e.g., $Y = e^X$), Kendall's $\tau = 1$ (perfect concordance) while Pearson's $r < 1$ because $r$ measures only linear association.

    3. **Distribution-free:** No normality assumption is required for the validity of inference based on $\tau$.

---

**Exercise 4.**
How are ties handled in Kendall's $\tau$? State the formula for $\tau_b$ (the version adjusted for ties).

??? success "Solution to Exercise 4"
    When ties exist, a pair with $X_i = X_j$ or $Y_i = Y_j$ is neither concordant nor discordant. The basic $\tau_a$ formula $\tau_a = (C - D)/\binom{n}{2}$ does not account for ties, so $|\tau_a| < 1$ even for perfectly monotonic data with ties.

    Kendall's $\tau_b$ adjusts the denominator:

    $$
    \tau_b = \frac{C - D}{\sqrt{(C + D + T_X)(C + D + T_Y)}}
    $$

    where $T_X$ is the number of pairs tied on $X$ only, and $T_Y$ is the number of pairs tied on $Y$ only. This ensures $\tau_b$ can reach $\pm 1$ when the data are as concordant (or discordant) as possible given the tie structure. Most software reports $\tau_b$ by default.
