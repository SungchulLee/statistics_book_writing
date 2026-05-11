# Handling Assumption Violations


## Overview

When diagnostic checks reveal that one or more ANOVA assumptions are violated, it is important to take corrective actions to ensure valid conclusions. The appropriate response depends on the nature and severity of the violation. This section provides a systematic guide to addressing each type of violation.

## Step-by-Step Approach

1. **Identify the source:** Determine which assumption is violated and the extent of the violation using the diagnostic tools described in previous sections.
2. **Assess severity:** Minor violations may have negligible impact on results, especially with large, balanced samples. Severe violations require corrective action.
3. **Choose a remedy:** Select from the options below based on the specific violation.
4. **Verify the fix:** After applying a correction, re-run the diagnostics to confirm the assumption is now met.

## Non-Parametric Alternatives

### Kruskal-Wallis Test

When the normality assumption is violated, the Kruskal-Wallis test is a non-parametric alternative to one-way ANOVA. It compares the medians (more precisely, the mean ranks) rather than the means across groups and does not assume normality of residuals.

```python
from scipy.stats import kruskal

stat, p_value = kruskal(group1, group2, group3)
print(f"Kruskal-Wallis: H = {stat:.4f}, p-value = {p_value:.4f}")
```

The Kruskal-Wallis test is less sensitive to outliers and skewed distributions but assumes that the distributions have the same shape (differing only in location). For a detailed treatment, see [Kruskal-Wallis Test](../../ch16/multi_group_nonparametric/kruskal_wallis.md).

## Data Transformations

Transformations can simultaneously address violations of normality and homoscedasticity by changing the scale of the data.

### Log Transformation

Used when the data is positively skewed or when the variance increases with the mean:

$$
Y' = \log(Y) \quad \text{or} \quad Y' = \log(Y + c) \text{ if } Y \text{ contains zeros}
$$

```python
import numpy as np

data['log_response'] = np.log(data['response'])
```

### Square Root Transformation

Useful for count data that follow a Poisson-like distribution:

$$
Y' = \sqrt{Y}
$$

```python
data['sqrt_response'] = np.sqrt(data['response'])
```

### Box-Cox Transformation

A family of power transformations parameterized by $\lambda$ that can be optimized to achieve the best approximation to normality:

$$
Y'(\lambda) = \begin{cases} \frac{Y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \log(Y) & \text{if } \lambda = 0 \end{cases}
$$

```python
from scipy.stats import boxcox

transformed_data, best_lambda = boxcox(data['response'])
print(f"Optimal λ = {best_lambda:.4f}")
```

!!! note "Interpretation After Transformation"
    After transforming the data, the ANOVA tests hypotheses about the transformed means, not the original means. Be careful when interpreting and reporting results—back-transform estimates when possible and clearly state what scale the analysis was conducted on.

## Robust ANOVA Methods

### Welch's ANOVA

Welch's ANOVA does not assume equal variances across groups. It adjusts the degrees of freedom of the F-test using the Welch-Satterthwaite approximation:

```python
from scipy.stats import f_oneway
# Or use pingouin for Welch's ANOVA directly
import pingouin as pg

welch_result = pg.welch_anova(dv='response', between='group', data=data)
print(welch_result)
```

For a full treatment, see [Welch's One-Way ANOVA](../anova_welch/welch_one_way.md).

### Robust Estimators

Methods like Huber or M-estimators can provide ANOVA-like results that are less sensitive to outliers:

```python
import statsmodels.api as sm

rlm_model = sm.RLM.from_formula('response ~ group', data=data, M=sm.robust.norms.HuberT())
result = rlm_model.fit()
print(result.summary())
```

## Permutation Tests

Permutation tests make minimal distributional assumptions. They work by:

1. Computing the observed F-statistic.
2. Randomly shuffling the group labels many times.
3. Recomputing the F-statistic for each permutation.
4. Comparing the observed F-statistic to the permutation distribution.

```python
import numpy as np
from scipy.stats import f_oneway

# Observed F-statistic
observed_f, _ = f_oneway(group1, group2, group3)

# Permutation test
all_data = np.concatenate([group1, group2, group3])
group_sizes = [len(group1), len(group2), len(group3)]
n_permutations = 10000
perm_f_stats = []

rng = np.random.default_rng(42)
for _ in range(n_permutations):
    shuffled = rng.permutation(all_data)
    g1 = shuffled[:group_sizes[0]]
    g2 = shuffled[group_sizes[0]:group_sizes[0]+group_sizes[1]]
    g3 = shuffled[group_sizes[0]+group_sizes[1]:]
    f_stat, _ = f_oneway(g1, g2, g3)
    perm_f_stats.append(f_stat)

p_value = np.mean(np.array(perm_f_stats) >= observed_f)
print(f"Permutation test p-value: {p_value:.4f}")
```

For a detailed treatment, see [Permutation Tests](../../ch17/permutation/foundations.md).

## Summary of Remedies by Violation

| Violation | Recommended Remedies |
|-----------|---------------------|
| Non-normality | Transformations, Kruskal-Wallis, bootstrapping |
| Heteroscedasticity | Welch's ANOVA, transformations, robust standard errors |
| Non-independence | Mixed-effects models, repeated-measures ANOVA, GEE |
| Nonlinearity | Polynomial terms, transformations, GAMs |
| Outliers/Influential points | Robust estimators, sensitivity analysis, transformations |
## Exercises

**Exercise 1.**
A one-way ANOVA with four groups yields a significant F-test ($p = 0.008$), but Levene's test rejects the null of equal variances ($p = 0.003$) and the Shapiro-Wilk test on the residuals is non-significant ($p = 0.34$). Outline a step-by-step plan for obtaining valid inference.

??? success "Solution to Exercise 1"

    1. **Normality:** The Shapiro-Wilk test is non-significant, so normality is not a concern. No action needed.

    2. **Homoscedasticity:** Levene's test strongly rejects equal variances. The standard ANOVA F-test results are unreliable.

    3. **Recommended action:** Re-run the analysis using **Welch's one-way ANOVA**, which does not assume equal variances. If Welch's ANOVA is still significant, follow up with the **Games-Howell post-hoc test** (designed for unequal variances) rather than Tukey's HSD.

    4. **Optional:** Try a variance-stabilizing transformation (e.g., log) and check whether it resolves the heteroscedasticity. If so, the standard ANOVA on the transformed data may be used.

---

**Exercise 2.**
Both the normality and homoscedasticity assumptions are violated in a dataset with three groups of sizes $n = 12, 15, 10$. Recommend an analysis strategy, justifying each choice.

??? success "Solution to Exercise 2"
    With both assumptions violated, the options in order of preference are:

    1. **Kruskal-Wallis test.** This non-parametric alternative to one-way ANOVA does not assume normality or equal variances. It compares median ranks rather than means and is appropriate for ordinal or skewed data.

    2. **Bootstrap ANOVA.** Use resampling to obtain the null distribution of the F-statistic without distributional assumptions. This preserves the mean-comparison framework while relaxing assumptions.

    3. **Transformation + Welch's ANOVA.** If a transformation (e.g., log or Box-Cox) can approximately normalize the data, Welch's ANOVA handles the remaining heteroscedasticity.

    The unequal sample sizes make the standard ANOVA particularly sensitive to heteroscedasticity, further supporting the use of Welch's ANOVA or non-parametric methods.

---

**Exercise 3.**
Explain why simply removing outliers detected by Cook's distance is not always the best strategy in ANOVA diagnostics. What should a researcher do instead?

??? success "Solution to Exercise 3"
    Removing outliers can introduce **selection bias** and reduce sample size, potentially eliminating valid observations that represent genuine population variability. The researcher should instead:

    1. **Investigate the outlier.** Determine whether it results from a data entry error, measurement malfunction, or a legitimately extreme observation.

    2. **Perform a sensitivity analysis.** Run the ANOVA with and without the outlier and compare results. If conclusions are the same, the outlier is not influential.

    3. **Use robust methods.** Trimmed means, Winsorized ANOVA, or M-estimators down-weight extreme observations without discarding them.

    4. **Report both analyses.** If conclusions differ, report results with and without the outlier and discuss the discrepancy.
