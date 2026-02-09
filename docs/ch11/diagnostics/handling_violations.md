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

The Kruskal-Wallis test is less sensitive to outliers and skewed distributions but assumes that the distributions have the same shape (differing only in location). For a detailed treatment, see [Kruskal-Wallis Test](../../ch19/two_sample_nonparametric/two_sample.md).

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

For a detailed treatment, see [Permutation Tests](../../ch20/permutation/permutation.md).

## Summary of Remedies by Violation

| Violation | Recommended Remedies |
|-----------|---------------------|
| Non-normality | Transformations, Kruskal-Wallis, bootstrapping |
| Heteroscedasticity | Welch's ANOVA, transformations, robust standard errors |
| Non-independence | Mixed-effects models, repeated-measures ANOVA, GEE |
| Nonlinearity | Polynomial terms, transformations, GAMs |
| Outliers/Influential points | Robust estimators, sensitivity analysis, transformations |
