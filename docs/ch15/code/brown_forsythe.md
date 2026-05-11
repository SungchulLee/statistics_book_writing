# Brown-Forsythe Test (scipy)

## Overview

The Brown--Forsythe test is a robust alternative to Bartlett's test for assessing the equality of variances across multiple groups. It is a variant of Levene's test that uses group medians instead of group means for centering, making it resistant to departures from normality. SciPy implements it through `scipy.stats.levene` with the parameter `center='median'`.

---

## Test Setup

Given $k$ independent groups with sizes $n_1, \ldots, n_k$, the hypotheses are:

$$
H_0 : \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2 \quad \text{vs} \quad H_1 : \text{not all } \sigma_i^2 \text{ are equal}
$$

## How It Works

The Brown--Forsythe test transforms the original observations into absolute deviations from the group median:

$$
z_{ij} = |x_{ij} - \tilde{x}_i|
$$

where $\tilde{x}_i$ is the median of group $i$. Then a one-way ANOVA $F$-test is performed on the transformed values $z_{ij}$:

$$
W = \frac{\sum_{i=1}^{k} n_i (\bar{z}_{i\cdot} - \bar{z}_{\cdot\cdot})^2 \,/\, (k - 1)}{\sum_{i=1}^{k}\sum_{j=1}^{n_i} (z_{ij} - \bar{z}_{i\cdot})^2 \,/\, (N - k)}
$$

where $\bar{z}_{i\cdot}$ is the mean of the transformed values in group $i$, $\bar{z}_{\cdot\cdot}$ is the overall mean, and $N = \sum n_i$. Under $H_0$, $W$ is approximately $F(k-1, N-k)$.

---

## Why the Median?

The original Levene's test uses the group **mean** for centering ($z_{ij} = |x_{ij} - \bar{x}_i|$). Brown and Forsythe (1974) showed that replacing the mean with the median produces a test that:

- Maintains the correct Type I error rate under skewed and heavy-tailed distributions.
- Has comparable power to Levene's test under normality.
- Is less affected by outliers because the median is a robust measure of location.

---

## Code

SciPy makes this straightforward with a single function call:

```python
import numpy as np
from scipy.stats import levene

g1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
g2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)
g3 = np.array([32, 35, 34, 30, 33, 34, 32, 31], dtype=float)

W, p = levene(g1, g2, g3, center='median')
print(f"Brown-Forsythe W = {W:.6f}, p-value = {p:.6f}")
```

For comparison, the mean-centered Levene test is:

```python
W_mean, p_mean = levene(g1, g2, g3, center='mean')
print(f"Levene (mean) W = {W_mean:.6f}, p-value = {p_mean:.6f}")
```

---

## Interpretation

- A large $W$ statistic (small $p$-value) leads to rejecting $H_0$, concluding that the group variances are not all equal.
- For these three groups, which differ mainly in location (not spread), we expect $W$ to be small and $p$ to be large, meaning we fail to reject equal variances.
- The Brown--Forsythe test is the recommended default for testing homoscedasticity when the normality of the data is uncertain.

---

## Exercises

**Exercise 1.** Compute the Brown--Forsythe test statistic $W$ by hand for two groups: $\mathbf{x}_1 = (2, 4, 6)$ and $\mathbf{x}_2 = (1, 5, 9, 13)$. Show each step: compute group medians, absolute deviations, group means of deviations, and the $F$-ratio.

??? success "Solution to Exercise 1"

    Group medians: $\tilde{x}_1 = 4$, $\tilde{x}_2 = 7$.

    Absolute deviations from medians:

    - Group 1: $z_{11} = |2-4| = 2$, $z_{12} = |4-4| = 0$, $z_{13} = |6-4| = 2$
    - Group 2: $z_{21} = |1-7| = 6$, $z_{22} = |5-7| = 2$, $z_{23} = |9-7| = 2$, $z_{24} = |13-7| = 6$

    Group means: $\bar{z}_1 = (2+0+2)/3 = 4/3$, $\bar{z}_2 = (6+2+2+6)/4 = 4$.

    Overall mean: $\bar{z} = (2+0+2+6+2+2+6)/7 = 20/7 \approx 2.857$.

    Between-group sum of squares:

    $$
    \text{SS}_B = 3(4/3 - 20/7)^2 + 4(4 - 20/7)^2 = 3(1.333 - 2.857)^2 + 4(4 - 2.857)^2
    $$

    $$
    = 3(2.322) + 4(1.306) = 6.966 + 5.224 = 12.19
    $$

    Within-group sum of squares:

    $$
    \text{SS}_W = (2-1.333)^2 + (0-1.333)^2 + (2-1.333)^2 + (6-4)^2 + (2-4)^2 + (2-4)^2 + (6-4)^2
    $$

    $$
    = 0.444 + 1.778 + 0.444 + 4 + 4 + 4 + 4 = 18.667
    $$

    $$
    W = \frac{12.19 / 1}{18.667 / 5} = \frac{12.19}{3.733} \approx 3.27
    $$

    With $F(1, 5)$, a critical value of about 6.61 at $\alpha = 0.05$, so we fail to reject $H_0$. $\square$

---

**Exercise 2.** Run a simulation study: generate three groups of size 30 from a standard lognormal distribution (all with equal variance). Apply both Bartlett's test and the Brown--Forsythe test at $\alpha = 0.05$ over 3000 replications. Compare the false-positive rates.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy.stats import bartlett, levene

    rng = np.random.default_rng(42)
    rej_bart, rej_bf = 0, 0
    n_sims = 3000

    for _ in range(n_sims):
        g1 = rng.lognormal(0, 1, 30)
        g2 = rng.lognormal(0, 1, 30)
        g3 = rng.lognormal(0, 1, 30)
        _, p_b = bartlett(g1, g2, g3)
        _, p_bf = levene(g1, g2, g3, center='median')
        if p_b < 0.05:
            rej_bart += 1
        if p_bf < 0.05:
            rej_bf += 1

    print(f"Bartlett false-positive rate:       {rej_bart/n_sims:.3f}")
    print(f"Brown-Forsythe false-positive rate:  {rej_bf/n_sims:.3f}")
    ```

    Bartlett's test will have a false-positive rate well above 0.05 (often 0.15--0.25) because the lognormal distribution has high kurtosis. The Brown--Forsythe test stays much closer to 0.05 due to median centering. $\square$

---

**Exercise 3.** Prove that the Brown--Forsythe test is invariant under location shifts. That is, if we add a constant $c_i$ to all observations in group $i$, the test statistic $W$ does not change.

??? success "Solution to Exercise 3"

    Let $x_{ij}' = x_{ij} + c_i$ for all $j$ in group $i$. The group median shifts by the same amount: $\tilde{x}_i' = \tilde{x}_i + c_i$. The transformed values are:

    $$
    z_{ij}' = |x_{ij}' - \tilde{x}_i'| = |(x_{ij} + c_i) - (\tilde{x}_i + c_i)| = |x_{ij} - \tilde{x}_i| = z_{ij}
    $$

    Since all $z_{ij}$ values are unchanged, the test statistic $W$ remains identical. This confirms that the Brown--Forsythe test depends only on the spread within each group, not on the group means. $\square$

---

**Exercise 4.** Apply the Brown--Forsythe test to four groups generated from $N(0, 1)$, $N(0, 1.5)$, $N(0, 2)$, and $N(0, 3)$ with $n = 25$ each. Report $W$ and the $p$-value. Then increase each group to $n = 100$ and repeat. How does sample size affect the power of the test?

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy.stats import levene

    rng = np.random.default_rng(42)

    for n in [25, 100]:
        g1 = rng.normal(0, 1.0, n)
        g2 = rng.normal(0, 1.5, n)
        g3 = rng.normal(0, 2.0, n)
        g4 = rng.normal(0, 3.0, n)
        W, p = levene(g1, g2, g3, g4, center='median')
        print(f"n={n:3d}: W={W:.4f}, p={p:.6f}")
    ```

    With $n = 25$, the test may or may not reject depending on the random draw, since power is moderate for this effect size. With $n = 100$, the $p$-value should be very small, indicating strong evidence against equal variances. Power increases with sample size because the sampling variability of each $S_i^2$ decreases, making the differences in population variances easier to detect. $\square$

---

**Exercise 5.** The Brown--Forsythe test uses the median for centering. An alternative is to use the **trimmed mean** (e.g., 10% trim). Implement a version that computes $z_{ij} = |x_{ij} - \bar{x}_{i,\text{trim}}|$ and performs a one-way ANOVA $F$-test on the $z$-values. Compare the false-positive rate to the median-based version under data from a $t(3)$ distribution.

??? success "Solution to Exercise 5"

    ```python
    import numpy as np
    from scipy.stats import levene, trim_mean, f_oneway

    rng = np.random.default_rng(0)
    rej_med, rej_trim = 0, 0
    n_sims = 3000

    for _ in range(n_sims):
        g1 = rng.standard_t(3, 30)
        g2 = rng.standard_t(3, 30)
        g3 = rng.standard_t(3, 30)

        # Median centering (Brown-Forsythe)
        _, p_med = levene(g1, g2, g3, center='median')
        if p_med < 0.05:
            rej_med += 1

        # Trimmed mean centering
        groups = [g1, g2, g3]
        z_groups = []
        for g in groups:
            tm = trim_mean(g, 0.1)
            z_groups.append(np.abs(g - tm))
        _, p_trim = f_oneway(*z_groups)
        if p_trim < 0.05:
            rej_trim += 1

    print(f"Median centering FPR:       {rej_med/n_sims:.3f}")
    print(f"Trimmed mean centering FPR: {rej_trim/n_sims:.3f}")
    ```

    Both methods control the false-positive rate near 0.05 for the $t(3)$ distribution. The trimmed mean version may be slightly more efficient (higher power under alternatives) when the data are symmetric with moderate tails, because the trimmed mean is a more efficient estimator of location than the median for such distributions. However, the median is more robust to extreme skewness. $\square$
