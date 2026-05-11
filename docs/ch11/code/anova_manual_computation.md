# Manual Analysis of Variance with Fisher Method

## Overview

Understanding ANOVA deeply requires computing every quantity by hand at least once. This page derives the one-way ANOVA decomposition from first principles, walks through a manual calculation of SST, SSE, MST, MSE, and the F-statistic on synthetic height data, verifies the result against `scipy.stats.f_oneway`, and then applies the Fisher Least Significant Difference (LSD) post-hoc procedure to identify which specific pairs of groups differ.

## One-Way ANOVA Decomposition

Suppose we observe $k$ groups with sample sizes $n_1, \dots, n_k$ and total sample size $N = \sum_{i=1}^{k} n_i$. Let $\bar{y}$ denote the grand mean and $\bar{y}_i$ the mean of group $i$. The total variation decomposes as

$$
\underbrace{\sum_{i=1}^{k}\sum_{j=1}^{n_i}(y_{ij} - \bar{y})^2}_{\text{SS}_{\text{total}}} = \underbrace{\sum_{i=1}^{k} n_i (\bar{y}_i - \bar{y})^2}_{\text{SST (between)}} + \underbrace{\sum_{i=1}^{k}\sum_{j=1}^{n_i}(y_{ij} - \bar{y}_i)^2}_{\text{SSE (within)}}
$$

The mean squares and F-statistic are

$$
\text{MST} = \frac{\text{SST}}{k - 1}, \qquad \text{MSE} = \frac{\text{SSE}}{N - k}, \qquad F = \frac{\text{MST}}{\text{MSE}}
$$

Under $H_0: \mu_1 = \mu_2 = \cdots = \mu_k$, we have $F \sim F(k-1,\, N-k)$.

## Manual Computation in Python

The following function computes every ANOVA quantity from scratch:

```python
import numpy as np
from scipy import stats

def manual_anova(groups):
    all_data = np.concatenate(list(groups.values()))
    grand_mean = all_data.mean()
    N = len(all_data)
    k = len(groups)

    SST = sum(len(g) * (g.mean() - grand_mean) ** 2
              for g in groups.values())
    SSE = sum(np.sum((g - g.mean()) ** 2)
              for g in groups.values())

    MST = SST / (k - 1)
    MSE = SSE / (N - k)
    F = MST / MSE
    p_value = 1 - stats.f.cdf(F, k - 1, N - k)
    return SST, SSE, MST, MSE, F, p_value
```

The scipy verification is a single line:

```python
F_scipy, p_scipy = stats.f_oneway(*groups.values())
```

Both approaches produce identical $F$ and $p$-values, confirming the manual calculation.

## Fisher LSD Post-Hoc Comparison

After rejecting the global null, Fisher's Least Significant Difference identifies which pairs of means differ. For groups $i$ and $j$, the LSD threshold is

$$
\text{LSD} = t_{\alpha/2,\, N-k} \sqrt{\text{MSE}\!\left(\frac{1}{n_i} + \frac{1}{n_j}\right)}
$$

If $|\bar{y}_i - \bar{y}_j| > \text{LSD}$, the pair is declared significantly different at level $\alpha$.

```python
from itertools import combinations

def fisher_lsd(groups, MSE, alpha=0.05):
    names = list(groups.keys())
    N_total = sum(len(g) for g in groups.values())
    k = len(groups)
    df_within = N_total - k
    results = []
    for (n1, g1), (n2, g2) in combinations(groups.items(), 2):
        t_crit = stats.t.ppf(1 - alpha / 2, df_within)
        lsd_val = t_crit * np.sqrt(MSE * (1/len(groups[n1]) + 1/len(groups[n2])))
        diff = abs(groups[n1].mean() - groups[n2].mean())
        results.append({"pair": f"{n1} vs {n2}",
                        "diff": diff, "LSD": lsd_val,
                        "significant": diff > lsd_val})
    return results
```

## Interpretation

In the accompanying script, three groups of simulated heights (Dutch $\mu = 183$, Japanese $\mu = 172$, Danish $\mu = 181$, each $n = 30$) produce a large F-statistic with $p \approx 0$, decisively rejecting $H_0$.

The Fisher LSD follow-up typically finds:

- **Dutch vs. Japanese:** significant (large mean difference).
- **Dutch vs. Danish:** not significant (similar means).
- **Japanese vs. Danish:** significant (large mean difference).

This illustrates a common pattern: the global ANOVA rejects, but not every pairwise comparison is significant. Post-hoc methods are essential for determining which specific groups drive the overall effect.

## Exercises

**Exercise 1.**
Given three groups with means $\bar{y}_1 = 10$, $\bar{y}_2 = 14$, $\bar{y}_3 = 12$, each of size $n = 20$, and grand mean $\bar{y} = 12$, compute SST.

??? success "Solution to Exercise 1"
    Using $\text{SST} = \sum_{i=1}^{k} n_i (\bar{y}_i - \bar{y})^2$:

    $$
    \text{SST} = 20(10 - 12)^2 + 20(14 - 12)^2 + 20(12 - 12)^2 = 20(4) + 20(4) + 20(0) = 160
    $$

---

**Exercise 2.**
Explain why the Fisher LSD procedure does not control the family-wise error rate when the number of groups $k$ is large. What alternative would you recommend?

??? success "Solution to Exercise 2"
    Fisher LSD performs each pairwise comparison at level $\alpha$ without adjustment. With $\binom{k}{2}$ comparisons, the probability of at least one false rejection grows rapidly. For $k = 5$ groups there are 10 pairwise tests, giving a family-wise error rate that can approach $1 - (1 - \alpha)^{10} \approx 0.40$ under the global null.

    The Tukey Honest Significant Difference (HSD) method is the standard alternative. It controls the family-wise error rate at $\alpha$ for all pairwise comparisons simultaneously by using the Studentized range distribution rather than the $t$-distribution.

---

**Exercise 3.**
Show that $\text{SS}_{\text{total}} = \text{SST} + \text{SSE}$ by expanding the identity $y_{ij} - \bar{y} = (\bar{y}_i - \bar{y}) + (y_{ij} - \bar{y}_i)$.

??? success "Solution to Exercise 3"
    Squaring both sides and summing:

    $$
    \sum_{i}\sum_{j}(y_{ij} - \bar{y})^2 = \sum_{i}\sum_{j}(\bar{y}_i - \bar{y})^2 + 2\sum_{i}\sum_{j}(\bar{y}_i - \bar{y})(y_{ij} - \bar{y}_i) + \sum_{i}\sum_{j}(y_{ij} - \bar{y}_i)^2
    $$

    The cross term vanishes because for each group $i$:

    $$
    \sum_{j=1}^{n_i}(y_{ij} - \bar{y}_i) = 0
    $$

    Therefore $(\bar{y}_i - \bar{y})\sum_j (y_{ij} - \bar{y}_i) = 0$ for every $i$. The remaining two terms are exactly $\text{SST}$ (noting $\sum_j (\bar{y}_i - \bar{y})^2 = n_i(\bar{y}_i - \bar{y})^2$) and $\text{SSE}$. $\square$

---

**Exercise 4.**
In the height example, suppose the Danish group has $n = 5$ instead of $n = 30$. How would this affect the LSD threshold for Dutch vs. Danish compared to the balanced case?

??? success "Solution to Exercise 4"
    The LSD threshold is

    $$
    \text{LSD} = t_{\alpha/2,\, N-k}\sqrt{\text{MSE}\!\left(\frac{1}{n_i} + \frac{1}{n_j}\right)}
    $$

    With $n_{\text{Danish}} = 5$ instead of 30, the term $1/n_j$ increases from $1/30 \approx 0.033$ to $1/5 = 0.2$. The sum $1/n_i + 1/n_j$ increases from roughly $0.067$ to $0.233$, nearly quadrupling the expression under the square root. Consequently the LSD threshold increases substantially, making it harder to declare the Dutch-Danish difference significant. Additionally, the total $N$ decreases and $\text{MSE}$ may change, further widening the threshold.

---

**Exercise 5.**
Under what conditions does $\text{MST}$ provide an unbiased estimate of $\sigma^2$? What does $\text{MST}$ estimate when $H_0$ is false?

??? success "Solution to Exercise 5"
    Under $H_0: \mu_1 = \cdots = \mu_k$, each group mean $\bar{Y}_i$ estimates the common mean $\mu$, and

    $$
    E[\text{MST}] = \sigma^2
    $$

    so MST is an unbiased estimator of the common variance. When $H_0$ is false,

    $$
    E[\text{MST}] = \sigma^2 + \frac{\sum_{i=1}^{k} n_i (\mu_i - \bar{\mu})^2}{k - 1}
    $$

    where $\bar{\mu} = \sum n_i \mu_i / N$. The second term is positive whenever the group means are not all equal, so $E[\text{MST}] > \sigma^2$. Since $E[\text{MSE}] = \sigma^2$ regardless of $H_0$, the ratio $F = \text{MST}/\text{MSE}$ tends to be larger than 1 under the alternative, which is why the F-test has power to detect differences.
