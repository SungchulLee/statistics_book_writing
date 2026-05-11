# Levene's Test

## Overview

Levene's test assesses whether two or more groups have equal variances (homoscedasticity). Unlike Bartlett's test, Levene's test is robust to departures from normality because it reduces the variance-comparison problem to a one-way ANOVA on absolute deviations. This robustness makes it the most commonly recommended preliminary check before procedures that assume equal variances, such as the pooled two-sample $t$-test or one-way ANOVA.

## Test Setup

Given $k$ independent groups of sizes $n_1, \ldots, n_k$, the hypotheses are

$$
H_0 : \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2 \quad \text{versus} \quad H_1 : \text{not all } \sigma_i^2 \text{ are equal}.
$$

## Test Statistic

For each observation $X_{ij}$, define the transformed value

$$
Z_{ij} = |X_{ij} - \bar{X}_i|,
$$

where $\bar{X}_i$ is the group mean (Levene's original formulation). The test statistic is the one-way ANOVA F-statistic applied to the $Z_{ij}$:

$$
W = \frac{(N - k) \sum_{i=1}^{k} n_i (\bar{Z}_{i\cdot} - \bar{Z}_{\cdot\cdot})^2}{(k - 1) \sum_{i=1}^{k} \sum_{j=1}^{n_i} (Z_{ij} - \bar{Z}_{i\cdot})^2},
$$

where $N = \sum n_i$, $\bar{Z}_{i\cdot}$ is the mean of $Z_{ij}$ within group $i$, and $\bar{Z}_{\cdot\cdot}$ is the overall mean. Under $H_0$,

$$
W \;\dot{\sim}\; F(k-1,\; N-k).
$$

!!! note "Centering variants"
    Replacing the group mean $\bar{X}_i$ with the group **median** yields the Brown--Forsythe test, which is even more robust for skewed distributions. Using a **trimmed mean** (e.g., 10% trimmed) offers a compromise.

## Code

SciPy provides `scipy.stats.levene` directly:

```python
import numpy as np
import scipy.stats as stats

size, seed = 100, 1
x = stats.norm(loc=0, scale=1).rvs(size, random_state=seed)

for scale in [1.00, 1.05, 1.10, 1.15, 1.20]:
    y = stats.norm(loc=1, scale=scale).rvs(size, random_state=seed)
    stat, pval = stats.levene(x, y)
    print(f"sigma_y={scale:.2f}  F={stat:.2f}  p={pval:.3f}")
```

The `center` parameter controls the centering strategy:

```python
# Brown-Forsythe variant (median-centered)
stat, pval = stats.levene(x, y, center='median')

# Original Levene (mean-centered)
stat, pval = stats.levene(x, y, center='mean')

# Trimmed-mean variant
stat, pval = stats.levene(x, y, center='trimmed')
```

## Interpretation

- A large $W$ (small $p$-value) indicates that the groups differ in spread.
- Levene's test is appropriate as a preliminary check before ANOVA or the pooled $t$-test.
- For skewed distributions, the median-centered variant (Brown--Forsythe) controls the Type I error rate more tightly than the mean-centered version.

## Exercises

**Exercise 1.** Two groups have the following data: Group A = $\{3, 7, 8, 5, 6\}$, Group B = $\{12, 14, 11, 19, 15\}$. Compute the Levene test statistic $W$ by hand using mean-centering.

??? success "Solution to Exercise 1"

    Group means: $\bar{X}_A = 5.8$, $\bar{X}_B = 14.2$.

    Absolute deviations:

    - Group A: $|3-5.8|=2.8$, $|7-5.8|=1.2$, $|8-5.8|=2.2$, $|5-5.8|=0.8$, $|6-5.8|=0.2$.
    - Group B: $|12-14.2|=2.2$, $|14-14.2|=0.2$, $|11-14.2|=3.2$, $|19-14.2|=4.8$, $|15-14.2|=0.8$.

    Group means of deviations: $\bar{Z}_A = 1.44$, $\bar{Z}_B = 2.24$, overall $\bar{Z} = 1.84$.

    Between-group SS: $5(1.44-1.84)^2 + 5(2.24-1.84)^2 = 5(0.16) + 5(0.16) = 1.6$.

    Within-group SS for A: $(2.8-1.44)^2 + (1.2-1.44)^2 + (2.2-1.44)^2 + (0.8-1.44)^2 + (0.2-1.44)^2 = 1.8496 + 0.0576 + 0.5776 + 0.4096 + 1.5376 = 4.432$.

    Within-group SS for B: $(2.2-2.24)^2 + (0.2-2.24)^2 + (3.2-2.24)^2 + (4.8-2.24)^2 + (0.8-2.24)^2 = 0.0016 + 4.1616 + 0.9216 + 6.5536 + 2.0736 = 13.712$.

    $$
    W = \frac{(10 - 2) \cdot 1.6}{(2 - 1) \cdot (4.432 + 13.712)} = \frac{12.8}{18.144} \approx 0.706.
    $$

    With $F(1, 8)$, the $p$-value is large, so we fail to reject equal variances.

---

**Exercise 2.** Explain intuitively why applying ANOVA to the absolute deviations $|X_{ij} - \bar{X}_i|$ tests for equality of variances.

??? success "Solution to Exercise 2"

    The absolute deviation $|X_{ij} - \bar{X}_i|$ measures how far each observation is from its group center. If a group has a larger variance, the average absolute deviation will be larger. Levene's test thus converts a variance-comparison problem into a mean-comparison problem: it asks whether the average spreads differ across groups. Since ANOVA is designed to compare group means, applying it to the $Z_{ij}$ is a natural and effective approach. The key insight is that $E[|X - \mu|] = \sigma\sqrt{2/\pi}$ for normal data, so differences in $\sigma$ translate into differences in $E[Z]$. $\square$

---

**Exercise 3.** Write a simulation comparing the Type I error rate of Levene's test (mean-centered) and Levene's test (median-centered, i.e., Brown--Forsythe) under data drawn from $\text{Exp}(1)$. Use $k = 3$ groups, $n = 20$, and $\alpha = 0.05$.

??? success "Solution to Exercise 3"

    ```python
    import numpy as np
    import scipy.stats as stats

    rng = np.random.default_rng(0)
    n_sims, n, alpha = 5000, 20, 0.05
    rej_mean, rej_median = 0, 0

    for _ in range(n_sims):
        g1 = rng.exponential(1, n)
        g2 = rng.exponential(1, n)
        g3 = rng.exponential(1, n)
        _, p_mean = stats.levene(g1, g2, g3, center='mean')
        _, p_med = stats.levene(g1, g2, g3, center='median')
        if p_mean < alpha:
            rej_mean += 1
        if p_med < alpha:
            rej_median += 1

    print(f"Mean-centered:   {rej_mean/n_sims:.3f}")
    print(f"Median-centered: {rej_median/n_sims:.3f}")
    ```

    The median-centered version will have a rejection rate closer to 0.05 because the exponential distribution is right-skewed, and the median is more resistant to outliers than the mean.

---

**Exercise 4.** Prove that for two groups of equal size $n$ under normality, Levene's test (mean-centered) is equivalent to a two-sample $t$-test on the absolute deviations.

??? success "Solution to Exercise 4"

    With $k = 2$ groups, the one-way ANOVA F-statistic with 1 numerator degree of freedom satisfies $F = t^2$, where $t$ is the two-sample $t$-statistic. Since Levene's test statistic $W$ is defined as the ANOVA $F$-statistic on the $Z_{ij} = |X_{ij} - \bar{X}_i|$, and ANOVA with two groups reduces to a $t$-test, we have

    $$
    W = t_Z^2,
    $$

    where $t_Z$ is the pooled two-sample $t$-statistic comparing $\bar{Z}_A$ to $\bar{Z}_B$. The $p$-value from $F(1, 2n-2)$ equals the two-sided $p$-value from $t(2n-2)$. Therefore Levene's test in the two-group case is exactly a two-sample $t$-test on absolute deviations from group means. $\square$

---

**Exercise 5.** A researcher has four treatment groups with sample sizes 25, 30, 28, and 22. She runs Levene's test and obtains $W = 3.12$ with $p = 0.028$. She plans to run a one-way ANOVA. Describe two courses of action she might take given this result, and explain the trade-offs.

??? success "Solution to Exercise 5"

    **Option 1: Use Welch's ANOVA.** Since $p = 0.028 < 0.05$ indicates unequal variances, she can switch from the standard one-way ANOVA (which assumes homoscedasticity) to Welch's ANOVA (`scipy.stats.alexandergovern` or the Welch correction), which does not require equal variances. Trade-off: Welch's ANOVA can be slightly less powerful than standard ANOVA when variances are actually equal, but it maintains correct Type I error control under heteroscedasticity.

    **Option 2: Transform the data.** A variance-stabilizing transformation (e.g., log, square root, or Box--Cox) may equalize the variances, after which standard ANOVA can proceed. Trade-off: the hypotheses are now about the transformed means, which may be harder to interpret. Also, the transformation may not fully stabilize variances, and it changes the scale of inference.

    A third possibility is to proceed with standard ANOVA anyway if the sample sizes are nearly equal, since ANOVA is somewhat robust to moderate heteroscedasticity when group sizes are balanced. However, with the given $p$-value, Welch's ANOVA is the safest choice.
