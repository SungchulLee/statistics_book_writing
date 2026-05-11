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

## Exercises

**Exercise 1.**
Plant heights under Conditions A: $[15, 20, 25]$, B: $[10, 15, 35]$. Suitable for chi-square test?

??? success "Solution to Exercise 1"
    **No.** Chi-square requires categorical/count data; heights are continuous measurements.

    For comparing continuous distributions: use $t$-test (if approximately normal) or Mann-Whitney U test (non-parametric). The chi-square test would require binning the heights into categories first, losing information.

    Common chi-square applications: independence in contingency tables, goodness-of-fit for discrete distributions, homogeneity across populations.

---

**Exercise 2.**
**Expected cell count rule.** State the standard rule and what happens when violated.

??? success "Solution to Exercise 2"
    **Cochran's rule:** all expected cell counts should be $\ge 5$; ideally $\ge 80\%$ of cells.

    More lenient version: at least $20\%$ of cells have expected count $\ge 5$, none $< 1$.

    **When violated:** the chi-square distribution is a poor approximation. Type I error inflated.

    **Remedies:**

    - **Combine cells:** merge categories with low counts.
    - **Fisher's exact test:** for $2 \times 2$ tables, gives exact p-value.
    - **Monte Carlo simulation:** simulate null distribution.
    - **Exact tests** for larger tables (computationally intensive).

---

**Exercise 3.**
**$2 \times 2$ table.** $\{\{10, 15\}, \{20, 25\}\}$. Compute expected counts. Test independence at $\alpha = 0.05$.

??? success "Solution to Exercise 3"
    Row totals: 25, 45. Col totals: 30, 40. Grand total: 70.

    Expected: $E_{ij} = (\text{row}_i \cdot \text{col}_j)/N$.

    $E_{11} = 25 \cdot 30/70 = 10.71$. $E_{12} = 25 \cdot 40/70 = 14.29$. $E_{21} = 45 \cdot 30/70 = 19.29$. $E_{22} = 45 \cdot 40/70 = 25.71$.

    Chi-square: $\sum (O - E)^2/E = 0.067 + 0.050 + 0.037 + 0.028 = 0.182$.

    df = (2-1)(2-1) = 1. Critical $\chi^2_{1, 0.05} = 3.841$. **Fail to reject.** No evidence of dependence.

---

**Exercise 4.**
**Yates' continuity correction** for $2 \times 2$. When is it applied?

??? success "Solution to Exercise 4"
    Modified statistic: $\chi^2_Y = \sum (|O - E| - 0.5)^2/E$ — subtracts 0.5 from absolute deviation before squaring.

    **Purpose:** discrete counts approximated by continuous chi-square. Correction shifts each |O - E| toward zero, accounting for the gap between integer counts.

    **When to apply:** $2 \times 2$ tables only. Conservative — reduces Type I error but also power.

    **Modern view:** Yates correction is often considered too conservative. Fisher's exact test is preferred for small $2 \times 2$ tables. Yates is a legacy correction; many statisticians omit it.

---

**Exercise 5.**
**Sample size determination** for chi-square test of independence.

??? success "Solution to Exercise 5"
    Power for chi-square depends on the **effect size** $w$ (Cohen's $w$): $w = \sqrt{\sum (p_o - p_e)^2/p_e}$ summed over cells.

    Sample size formula (approximate): $n = \lambda_{\alpha, \beta, df}/w^2$.

    Common values from non-central chi-square tables:

    - df = 1, 80% power, $\alpha = 0.05$: $\lambda \approx 7.85$.
    - df = 4: $\lambda \approx 11.94$.

    Small effect ($w = 0.1$): $n \approx 785$ for df = 1.
    Medium ($w = 0.3$): $n \approx 87$.
    Large ($w = 0.5$): $n \approx 32$.

    Use `statsmodels.stats.power.GofChisquarePower` or similar for precise computation.

---

**Exercise 6.**
**Common misuses** of chi-square test.

??? success "Solution to Exercise 6"
    1. **Continuous data not categorized:** can't apply directly; bin first or use different test.
    2. **Non-independent observations:** chi-square assumes independent cells. Repeated measures, paired data need McNemar's test instead.
    3. **Small expected counts ignored:** violation of Cochran's rule → inflated Type I error.
    4. **Multiple post-hoc cell comparisons:** without correction, false discovery inflation.
    5. **Conclusion of "no association" from large p-value:** failing to reject ≠ proving independence. May be inadequate power.
    6. **Effect size ignored:** highly significant $\chi^2$ with $n = 10^6$ may reflect trivial pattern.

    Always: check assumptions, report effect size (Cramer's V or odds ratio), consider exact tests for small tables.
