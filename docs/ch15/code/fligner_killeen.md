# Fligner-Killeen Test (scipy)

## Overview

The Fligner--Killeen test is a nonparametric test for the equality of variances across multiple groups. Unlike Bartlett's test (which assumes normality) and the Brown--Forsythe test (which is robust but still parametric in its $F$-distribution reference), the Fligner--Killeen test uses ranks of absolute deviations from group medians. It is among the most robust variance-homogeneity tests available and performs well under a wide range of distributional shapes.

---

## Test Setup

Given $k$ independent groups with sizes $n_1, \ldots, n_k$ and total sample size $N = \sum_{i=1}^k n_i$, the hypotheses are:

$$
H_0 : \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2 \quad \text{vs} \quad H_1 : \text{not all } \sigma_i^2 \text{ are equal}
$$

## How It Works

The Fligner--Killeen test proceeds in three steps:

**Step 1.** Compute absolute deviations from the group median:

$$
z_{ij} = |x_{ij} - \tilde{x}_i|
$$

**Step 2.** Rank all $N$ deviations, then transform the ranks using normal quantile scores:

$$
a_{ij} = \mathcal{N}^{-1}\!\left(\frac{1 + R_{ij}/(N+1)}{2}\right)
$$

where $R_{ij}$ is the rank of $z_{ij}$ among all $z$-values and $\mathcal{N}^{-1}$ is the standard normal quantile function.

**Step 3.** Compute a chi-squared statistic from the group means of the normal scores:

$$
X^2 = \frac{\sum_{i=1}^{k} n_i (\bar{a}_{i\cdot} - \bar{a}_{\cdot\cdot})^2}{\hat{V}}
$$

where $\hat{V}$ is the variance of the scores. Under $H_0$, $X^2 \sim \chi^2(k-1)$ approximately.

---

## Code

SciPy provides `scipy.stats.fligner` for a direct implementation:

```python
import numpy as np
from scipy.stats import fligner

g1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
g2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)
g3 = np.array([32, 35, 34, 30, 33, 34, 32, 31], dtype=float)

X2, p = fligner(g1, g2, g3, center='median')
print(f"Fligner-Killeen X2 = {X2:.6f}, p-value = {p:.6f}")
```

---

## Interpretation

- A large $X^2$ statistic (small $p$-value) leads to rejecting $H_0$, concluding that the variances differ across groups.
- For the three groups above, which have similar spread but different locations, the test statistic should be small and the $p$-value large.
- The normal-score transformation in Step 2 makes the test insensitive to outliers and distributional shape. This gives the Fligner--Killeen test excellent Type I error control under non-normality.

---

## Comparison with Other Tests

| Test | Centering | Reference distribution | Robustness |
|---|---|---|---|
| Bartlett | N/A (uses log variances) | $\chi^2$ | Sensitive to non-normality |
| Levene (mean) | Group mean | $F$ | Moderate |
| Brown--Forsythe | Group median | $F$ | Good |
| Fligner--Killeen | Group median + normal scores | $\chi^2$ | Excellent |

The Fligner--Killeen test is the most robust but may have slightly less power than Bartlett under strict normality. For general-purpose use with unknown distributions, it is an excellent choice.

---

## Exercises

**Exercise 1.** Compute the Fligner--Killeen test statistic by hand for two groups: $\mathbf{x}_1 = (3, 7, 5)$ and $\mathbf{x}_2 = (1, 10, 6, 4)$. Show the deviations from group medians, the ranks, the normal quantile scores, and the final $X^2$.

??? success "Solution to Exercise 1"

    Group medians: $\tilde{x}_1 = 5$, $\tilde{x}_2 = 5$.

    Absolute deviations: Group 1: $|3-5|=2$, $|7-5|=2$, $|5-5|=0$. Group 2: $|1-5|=4$, $|10-5|=5$, $|6-5|=1$, $|4-5|=1$.

    Combined deviations sorted: $0, 1, 1, 2, 2, 4, 5$ with ranks (midrank for ties): $R = 1, 2.5, 2.5, 4.5, 4.5, 6, 7$.

    Normal quantile scores $a = \mathcal{N}^{-1}((1 + R/8)/2)$:

    - $R=1$: $\mathcal{N}^{-1}(0.5625) = 0.157$
    - $R=2.5$: $\mathcal{N}^{-1}(0.656) = 0.402$
    - $R=4.5$: $\mathcal{N}^{-1}(0.781) = 0.774$
    - $R=6$: $\mathcal{N}^{-1}(0.875) = 1.150$
    - $R=7$: $\mathcal{N}^{-1}(0.938) = 1.534$

    Group 1 scores: $0.774, 0.774, 0.157$; mean $\bar{a}_1 = 0.568$.
    Group 2 scores: $1.150, 1.534, 0.402, 0.402$; mean $\bar{a}_2 = 0.872$.
    Overall mean: $\bar{a} = (3 \times 0.568 + 4 \times 0.872)/7 = 0.742$.

    The test statistic is computed from the between-group variance of scores divided by the overall variance of scores, yielding $X^2$. With $k-1 = 1$ degree of freedom and a small $X^2$, we fail to reject equal variances. $\square$

---

**Exercise 2.** Generate three groups of size 40 from a standard Cauchy distribution (extremely heavy tails, equal variances). Apply Bartlett's test, the Brown--Forsythe test, and the Fligner--Killeen test at $\alpha = 0.05$ over 2000 replications. Report the false-positive rate of each. Which test best controls the Type I error?

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy.stats import bartlett, levene, fligner

    rng = np.random.default_rng(42)
    rej = {"bartlett": 0, "brown_forsythe": 0, "fligner": 0}
    n_sims = 2000

    for _ in range(n_sims):
        g1 = rng.standard_cauchy(40)
        g2 = rng.standard_cauchy(40)
        g3 = rng.standard_cauchy(40)

        _, p_b = bartlett(g1, g2, g3)
        _, p_bf = levene(g1, g2, g3, center='median')
        _, p_fk = fligner(g1, g2, g3, center='median')

        if p_b < 0.05: rej["bartlett"] += 1
        if p_bf < 0.05: rej["brown_forsythe"] += 1
        if p_fk < 0.05: rej["fligner"] += 1

    for name, count in rej.items():
        print(f"{name:20s} FPR = {count/n_sims:.3f}")
    ```

    Bartlett's test will have a grossly inflated false-positive rate (often 0.30+). The Brown--Forsythe test will be better but may still exceed 0.05 under Cauchy tails. The Fligner--Killeen test, thanks to its rank-based normal scores, will be closest to the nominal 0.05 level. $\square$

---

**Exercise 3.** Explain why the normal quantile score transformation $a = \mathcal{N}^{-1}((1 + R/(N+1))/2)$ is used rather than the raw ranks. What would happen if raw ranks were used instead?

??? success "Solution to Exercise 3"

    The normal quantile scores serve two purposes:

    1. **Normalization**: Raw ranks are uniformly distributed, not normally distributed. The $\mathcal{N}^{-1}$ transformation converts them to approximately standard normal values, so that standard chi-squared asymptotics apply more accurately to the test statistic.

    2. **Downweighting extremes**: Raw ranks give equal spacing to all observations, meaning an extreme outlier (rank $N$) has the same "distance" from rank $N-1$ as any other adjacent pair. The normal quantile scores compress the tails: the difference between the largest and second-largest score is larger than between consecutive interior scores, but not as extreme as the actual data values might be. This provides some resistance to outliers while still preserving the ordering information.

    If raw ranks were used instead, the resulting test would still be nonparametric and valid, but the chi-squared approximation would be less accurate, and the test would have lower power against normal alternatives. The normal-score version is asymptotically optimal among rank-based tests when the underlying distribution is normal. $\square$

---

**Exercise 4.** Apply the Fligner--Killeen test to compare the variability of monthly returns of three simulated stock portfolios. Generate 60 months of returns for each: Portfolio A from $\mathcal{N}(0.01, 0.04^2)$, Portfolio B from $\mathcal{N}(0.01, 0.06^2)$, and Portfolio C from $\mathcal{N}(0.01, 0.08^2)$. Report the test result and discuss its practical significance.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy.stats import fligner

    rng = np.random.default_rng(42)
    port_a = rng.normal(0.01, 0.04, 60)
    port_b = rng.normal(0.01, 0.06, 60)
    port_c = rng.normal(0.01, 0.08, 60)

    X2, p = fligner(port_a, port_b, port_c, center='median')
    print(f"Fligner-Killeen: X2 = {X2:.4f}, p = {p:.6f}")
    print(f"Sample SDs: A={port_a.std(ddof=1):.4f}, "
          f"B={port_b.std(ddof=1):.4f}, C={port_c.std(ddof=1):.4f}")
    ```

    The true standard deviations (4%, 6%, 8%) represent meaningfully different risk levels. With 60 monthly observations per portfolio, the Fligner--Killeen test should reject $H_0$ of equal variances. In practice, investors care about volatility differences because they affect risk-adjusted returns. A statistically significant Fligner--Killeen result confirms that these portfolios carry genuinely different levels of risk, supporting differentiated portfolio allocation strategies. $\square$

---

**Exercise 5.** Derive the asymptotic null distribution of the Fligner--Killeen test statistic. Specifically, show that under $H_0$ with equal group sizes $n$ and $k$ groups, the test statistic converges to $\chi^2(k-1)$ as $n \to \infty$.

??? success "Solution to Exercise 5"

    Under $H_0$, all observations come from the same distribution, so the deviations $z_{ij} = |x_{ij} - \tilde{x}_i|$ are identically distributed across groups. The normal quantile scores $a_{ij}$ are therefore also identically distributed across groups.

    Let $\bar{a}_{i\cdot}$ be the mean score in group $i$ and $\bar{a}_{\cdot\cdot}$ the overall mean. Under $H_0$, $\bar{a}_{i\cdot}$ are approximately independent (for large $n$) with:

    $$
    E[\bar{a}_{i\cdot}] = \mu_a, \qquad \operatorname{Var}(\bar{a}_{i\cdot}) = \frac{\sigma_a^2}{n}
    $$

    where $\mu_a$ and $\sigma_a^2$ are the mean and variance of the score distribution. The test statistic is:

    $$
    X^2 = \frac{n}{\hat{\sigma}_a^2} \sum_{i=1}^{k} (\bar{a}_{i\cdot} - \bar{a}_{\cdot\cdot})^2
    $$

    By the multivariate CLT, the vector $\sqrt{n}(\bar{a}_{1\cdot} - \mu_a, \ldots, \bar{a}_{k\cdot} - \mu_a)$ converges to $\mathcal{N}(\mathbf{0}, \sigma_a^2 I_k)$. After centering at $\bar{a}_{\cdot\cdot}$, the quadratic form involves a projection onto the $(k-1)$-dimensional subspace orthogonal to the constant vector, yielding:

    $$
    X^2 \xrightarrow{d} \chi^2(k-1)
    $$

    This follows from the standard result that a quadratic form in a multivariate normal vector, with an idempotent matrix of rank $k-1$, has a $\chi^2(k-1)$ distribution. $\square$
