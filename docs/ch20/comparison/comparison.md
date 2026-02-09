# Comparison: Bootstrap vs Permutation Tests

## Overview

The **Bootstrap** and **Permutation Test** are two widely used resampling techniques. While they share the principle of using observed data to construct sampling distributions, they differ fundamentally in purpose, methodology, and application.

---

## Side-by-Side Comparison

### 1. Purpose

| Aspect | Bootstrap | Permutation Test |
|---|---|---|
| **Goal** | Estimate the distribution of a statistic for confidence intervals or variability | Test hypotheses by comparing the observed statistic to a null distribution |
| **Primary Use** | Confidence intervals, standard errors, model validation | Hypothesis testing (e.g., testing differences between groups) |

### 2. Methodology

| Aspect | Bootstrap | Permutation Test |
|---|---|---|
| **How It Works** | Repeatedly resamples the observed data **with replacement** | Repeatedly **rearranges (permutes)** the data labels **without replacement** |
| **Resampling Type** | With replacement | Without replacement |
| **Key Idea** | Mimics drawing new samples from the population using the observed sample as a proxy | Breaks the association between data and groups under the null hypothesis |

### 3. Assumptions

| Aspect | Bootstrap | Permutation Test |
|---|---|---|
| **Distribution** | None. Assumes the observed data is representative of the population | Assumes exchangeability under the null hypothesis |
| **Independence** | Assumes data points are independent (extensions exist for dependent data) | Assumes the null hypothesis holds, and labels can be shuffled |

### 4. Applications

| Aspect | Bootstrap | Permutation Test |
|---|---|---|
| **Confidence Intervals** | Widely used | Not typically used |
| **Hypothesis Testing** | Can be adapted, though less intuitive | Primary use |
| **Complex Statistics** | Handles complex statistics well (regression coefficients, skewness) | Best suited for simpler statistics (mean differences, correlation) |

### 5. Computational Cost

| Aspect | Bootstrap | Permutation Test |
|---|---|---|
| **Intensity** | High (resampling + recalculating for thousands of iterations) | Moderate to high (number of permutations grows factorially, but approximations are possible) |
| **Small Samples** | Efficient and accurate | Exact results possible (all permutations feasible) |

---

## Practical Example: Testing Difference in Means

### Bootstrap Approach

1. Resample the two groups **with replacement**.
2. Compute the mean difference for each bootstrap sample.
3. Estimate the **confidence interval** for the mean difference.

```python
import numpy as np

group_a = np.array([8, 7, 9, 10, 6])
group_b = np.array([5, 6, 4, 3, 7])

n_resamples = 10000
boot_diffs = []
for _ in range(n_resamples):
    a_boot = np.random.choice(group_a, size=len(group_a), replace=True)
    b_boot = np.random.choice(group_b, size=len(group_b), replace=True)
    boot_diffs.append(a_boot.mean() - b_boot.mean())

boot_diffs = np.array(boot_diffs)
ci_lower = np.percentile(boot_diffs, 2.5)
ci_upper = np.percentile(boot_diffs, 97.5)
print(f"Bootstrap 95% CI for mean difference: ({ci_lower:.2f}, {ci_upper:.2f})")
```

### Permutation Approach

1. Combine the two groups into a single dataset.
2. Randomly shuffle the group labels **without replacement**.
3. Compute the mean difference for each permutation.
4. Compare the observed mean difference to the distribution of permuted differences to calculate a **p-value**.

```python
import numpy as np

group_a = np.array([8, 7, 9, 10, 6])
group_b = np.array([5, 6, 4, 3, 7])

combined = np.concatenate([group_a, group_b])
observed_diff = group_a.mean() - group_b.mean()

n_permutations = 10000
perm_diffs = []
for _ in range(n_permutations):
    np.random.shuffle(combined)
    perm_diffs.append(combined[:len(group_a)].mean() - combined[len(group_a):].mean())

perm_diffs = np.array(perm_diffs)
p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
print(f"Permutation p-value: {p_value:.4f}")
```

---

## Summary Table

| Feature | **Bootstrap** | **Permutation Test** |
|---|---|---|
| **Primary Purpose** | Confidence intervals, variability estimation | Hypothesis testing |
| **Resampling** | With replacement | Without replacement |
| **Key Output** | Confidence intervals, standard errors | p-value |
| **Assumptions** | Data representative of population | Exchangeability under $H_0$ |
| **Computational Intensity** | High | Moderate to high |
| **Flexibility** | Very flexible (any statistic, complex models) | Less flexible, focuses on simpler tests |

---

## Decision Guide

### Use Bootstrap when:

- You need **confidence intervals** or **standard errors**.
- The statistic is complex (e.g., regression coefficients, quantiles, ratios).
- You want to estimate variability in addition to testing hypotheses.
- You need to assess uncertainty for a novel or unusual estimator.

### Use Permutation Test when:

- You need to **test hypotheses** about differences or associations.
- You have small datasets where all permutations are feasible.
- You want a precise, assumption-free alternative to parametric tests.
- You need an **exact** p-value (for small samples).

### Use Both when:

- You want comprehensive inference: use the **permutation test** for the p-value and the **bootstrap** for the confidence interval.
- This combination provides both a decision (reject/fail to reject) and an estimate of effect size with uncertainty.

---

## Relationship to Other Methods

| Method | Chapter | Resampling? | Key Difference |
|---|---|---|---|
| z-test, t-test | Ch 9 | No | Relies on theoretical distributions |
| Wilcoxon, Mann-Whitney | Ch 15 | No | Uses ranks, not raw values |
| Bootstrap | Ch 16 | Yes (with replacement) | Estimates any sampling distribution |
| Permutation Test | Ch 16 | Yes (without replacement) | Tests hypotheses via label shuffling |
