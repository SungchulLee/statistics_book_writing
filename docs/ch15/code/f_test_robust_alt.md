# F-Test Normality Sensitivity and Robust Alternatives

## Overview

The F-test for equality of variances assumes that both populations are normally distributed. When this assumption is violated, the test can produce misleading results. This page demonstrates how to check the normality assumption using the Shapiro--Wilk test and presents robust alternatives -- Levene's test, the Brown--Forsythe test, and the Fligner--Killeen test -- that maintain valid Type I error rates under non-normality.

## Checking Normality

Before applying the F-test, it is prudent to assess whether the normality assumption holds. The Shapiro--Wilk test is one of the most powerful tests for normality:

$$
H_0: \text{data come from a normal distribution} \quad \text{vs.} \quad H_1: \text{data are not normal}.
$$

A small $p$-value suggests non-normality, in which case robust alternatives to the F-test should be used.

## Robust Alternatives

| Test | Implementation | Key property |
|---|---|---|
| Levene (mean) | `levene(x1, x2, center='mean')` | ANOVA on absolute deviations from mean |
| Brown--Forsythe | `levene(x1, x2, center='median')` | ANOVA on absolute deviations from median |
| Fligner--Killeen | `fligner(x1, x2)` | Nonparametric, based on ranks of absolute deviations |

The Brown--Forsythe test replaces the group mean with the group median, making it more robust to skewness. The Fligner--Killeen test uses a rank-based approach and is the most robust of the three.

## Code

```python
import numpy as np
from scipy.stats import levene, fligner, shapiro

x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)

# Step 1: Check normality with Shapiro-Wilk
W1, p1 = shapiro(x1)
W2, p2 = shapiro(x2)
print(f"Shapiro-Wilk x1: W={W1:.4f}, p={p1:.4f}")
print(f"Shapiro-Wilk x2: W={W2:.4f}, p={p2:.4f}")

# Step 2: Apply robust alternatives
Wm, pm = levene(x1, x2, center='mean')
print(f"Levene (mean-centered):         W={Wm:.4f}, p={pm:.6f}")

Wmed, pmed = levene(x1, x2, center='median')
print(f"Brown-Forsythe (median-centered): W={Wmed:.4f}, p={pmed:.6f}")

X2, pF = fligner(x1, x2)
print(f"Fligner-Killeen:                  X2={X2:.4f}, p={pF:.6f}")
```

## Interpretation

- If the Shapiro--Wilk $p$-values are large (e.g., $> 0.05$), the normality assumption is plausible and the F-test may be used.
- If normality is rejected for either sample, the F-test should not be trusted. Use Levene's, Brown--Forsythe, or Fligner--Killeen instead.
- In the example above, all robust tests yield large $p$-values, suggesting no significant difference in variances between the two groups.
- In practice, many analysts skip the F-test entirely and default to Levene's or Brown--Forsythe, since these are valid under normality as well (with only a modest loss of power).

## Exercises

**Exercise 1.** Apply the Shapiro--Wilk test to the two samples in the code above. Based on the results, is it appropriate to use the F-test? Justify your answer.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy.stats import shapiro

    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)

    W1, p1 = shapiro(x1)
    W2, p2 = shapiro(x2)
    print(f"x1: W={W1:.4f}, p={p1:.4f}")
    print(f"x2: W={W2:.4f}, p={p2:.4f}")
    ```

    Both $p$-values are large (well above 0.05), so we fail to reject normality for either sample. The F-test is appropriate in this case. However, with only $n = 8$ observations per group, the Shapiro--Wilk test has low power to detect non-normality, so the result should be interpreted cautiously.

---

**Exercise 2.** Generate 50 observations from a $\chi^2(3)$ distribution and 50 from a $\chi^2(5)$ distribution. Apply the F-test, Levene's test, and Fligner--Killeen test. Compare the conclusions.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy.stats import levene, fligner, f as fdist

    rng = np.random.default_rng(42)
    x1 = rng.chisquare(df=3, size=50)
    x2 = rng.chisquare(df=5, size=50)

    # F-test
    F = np.var(x1, ddof=1) / np.var(x2, ddof=1)
    p_f = 2 * min(fdist(49, 49).cdf(F), fdist(49, 49).sf(F))
    print(f"F-test:         F={F:.3f}, p={p_f:.4f}")

    # Levene
    W, p_l = levene(x1, x2, center='mean')
    print(f"Levene (mean):  W={W:.3f}, p={p_l:.4f}")

    # Fligner-Killeen
    X2, p_fk = fligner(x1, x2)
    print(f"Fligner-Killeen: X2={X2:.3f}, p={p_fk:.4f}")
    ```

    The true variances are $2 \times 3 = 6$ and $2 \times 5 = 10$ (since $\operatorname{Var}(\chi^2(d)) = 2d$), so they do differ. The F-test $p$-value may be misleading due to the skewness of chi-squared data, while Levene and Fligner--Killeen provide more trustworthy $p$-values.

---

**Exercise 3.** Explain the difference between the Levene test with `center='mean'` and `center='median'`. Under what conditions does the choice matter?

??? success "Solution to Exercise 3"

    With `center='mean'`, the absolute deviations are computed as $Z_{ij} = |X_{ij} - \bar{X}_i|$. With `center='median'`, they are $Z_{ij} = |X_{ij} - \tilde{X}_i|$ where $\tilde{X}_i$ is the group median.

    The choice matters primarily when the data are **skewed**. For a right-skewed distribution, the mean is pulled toward the right tail, making $Z_{ij}$ larger for observations in the tail and smaller for observations below the mean. This asymmetry can inflate the within-group variability of $Z_{ij}$, affecting the calibration of the test. The median is resistant to skewness, so the median-centered deviations are more symmetric and the resulting test statistic is better calibrated under $H_0$.

    For symmetric distributions (including normal), the mean and median are close, and the two versions perform similarly. Under normality, the mean-centered version may have slightly higher power. $\square$

---

**Exercise 4.** The Fligner--Killeen test uses a normal-scores transformation of the absolute deviations. Describe this procedure in steps and explain why it achieves robustness.

??? success "Solution to Exercise 4"

    The Fligner--Killeen procedure:

    1. Compute absolute deviations from group medians: $Z_{ij} = |X_{ij} - \tilde{X}_i|$.
    2. Rank all $Z_{ij}$ values across all groups, obtaining ranks $R_{ij}$.
    3. Transform ranks to normal scores: $a_{ij} = \mathcal{N}^{-1}\!\left(\frac{1 + R_{ij}/(N+1)}{2}\right)$, where $\mathcal{N}^{-1}$ is the standard normal quantile function.
    4. Compute the Bartlett-like (or ANOVA-like) test statistic on the $a_{ij}$ scores.

    Robustness arises because the rank transformation discards information about the magnitudes of the deviations, retaining only their relative ordering. Extreme outliers receive large ranks but not disproportionately large scores (the normal-scores transformation is bounded). This makes the test insensitive to heavy tails, skewness, or individual outliers. $\square$

---

**Exercise 5.** Design a workflow for a data analyst who must check variance equality before running ANOVA. Include decision points for normality checking and test selection. Implement this workflow as a Python function.

??? success "Solution to Exercise 5"

    ```python
    import numpy as np
    from scipy.stats import shapiro, levene, fligner

    def check_equal_variances(*groups, alpha=0.05):
        """
        Workflow for checking equality of variances.
        Returns a dict with test results and recommendation.
        """
        # Step 1: Check normality of each group
        normality_ok = True
        shapiro_results = []
        for i, g in enumerate(groups):
            W, p = shapiro(g)
            shapiro_results.append((W, p))
            if p < alpha:
                normality_ok = False

        # Step 2: Select appropriate test
        if normality_ok:
            test_name = "Levene (mean-centered)"
            stat, pval = levene(*groups, center='mean')
        else:
            test_name = "Brown-Forsythe (median-centered)"
            stat, pval = levene(*groups, center='median')

        # Step 3: Also run Fligner-Killeen as backup
        fk_stat, fk_p = fligner(*groups)

        return {
            "normality_assumed": normality_ok,
            "shapiro_results": shapiro_results,
            "primary_test": test_name,
            "statistic": stat,
            "p_value": pval,
            "fligner_killeen_p": fk_p,
            "equal_variances": pval >= alpha,
        }

    # Example usage
    rng = np.random.default_rng(0)
    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(0, 1.5, 30)
    g3 = rng.normal(0, 2, 30)
    result = check_equal_variances(g1, g2, g3)
    for k, v in result.items():
        print(f"{k}: {v}")
    ```

    The function first checks normality via Shapiro--Wilk, then selects the mean-centered Levene test under normality or the median-centered (Brown--Forsythe) variant otherwise. The Fligner--Killeen result is included as a secondary check. The final recommendation is based on the primary test's $p$-value.
