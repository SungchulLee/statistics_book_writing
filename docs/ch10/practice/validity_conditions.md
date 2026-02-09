# Expected Cell Counts and Validity Conditions

## Overview

The chi-square test statistic is an **approximation** based on the asymptotic behavior of the multinomial distribution. For this approximation to be reliable, certain conditions must be met. When these conditions are violated, the p-values produced by the chi-square test may be inaccurate, potentially leading to incorrect conclusions.

## Rule of Thumb: Expected Frequency Threshold

The most widely cited validity condition is:

> **All expected cell frequencies should be at least 5.**

This rule of thumb ensures that the chi-square approximation to the true multinomial distribution is sufficiently accurate. The condition applies to **expected** frequencies, not observed frequencies.

### Why Expected, Not Observed?

The expected frequencies determine the shape of the sampling distribution under $H_0$. When expected counts are small, the discrete multinomial distribution is poorly approximated by the continuous chi-square distribution, leading to inflated Type I error rates.

## Conditions for Each Test

### Goodness-of-Fit Test

1. **Random Sampling**: The observations must be randomly sampled from the population.
2. **Independence**: Each observation is independent of others.
3. **Expected Frequency**: Each category should have an expected frequency of at least 5.
4. **Mutually Exclusive Categories**: Each observation falls into exactly one category.

### Test of Independence and Homogeneity

1. **Random Sampling**: Observations are randomly sampled (one sample for independence; separate samples for homogeneity).
2. **Independence**: Observations are independent within and across samples.
3. **Expected Frequency**: Each cell in the contingency table should have an expected frequency of at least 5.
4. **Mutually Exclusive Categories**: Each observation is classified into exactly one cell.

## What to Do When Conditions Are Violated

### Small Expected Frequencies

When some expected cell counts fall below 5:

1. **Combine categories**: Merge adjacent or related categories to increase expected counts. For example, combine "strongly agree" and "agree" into a single category.

2. **Fisher's Exact Test**: For $2 \times 2$ contingency tables with small samples, Fisher's Exact Test computes the exact p-value without relying on the chi-square approximation.

3. **Simulation-based tests**: Use Monte Carlo simulation or permutation tests to obtain p-values that do not depend on the chi-square approximation.

4. **Yates' continuity correction**: For $2 \times 2$ tables, apply the correction:

$$
\chi^2_{\text{Yates}} = \sum \frac{(|O_{ij} - E_{ij}| - 0.5)^2}{E_{ij}}
$$

This correction reduces the chi-square statistic slightly, producing a more conservative (larger) p-value.

### Non-Independence

If observations are not independent (e.g., repeated measures on the same subjects), the chi-square test is not appropriate. Consider alternatives such as McNemar's test for paired categorical data.

## Practical Guidelines

- Check expected frequencies **before** conducting the test.
- The rule of 5 is a guideline, not a strict cutoff. Some textbooks suggest that the test is acceptable if no more than 20% of expected frequencies are below 5, and none are below 1.
- For very large samples, the chi-square test will detect even trivially small deviations from the null hypothesis. In such cases, supplement the test with a measure of **effect size** (see Cramér's V).
- For very small samples, prefer exact tests over asymptotic chi-square tests.
