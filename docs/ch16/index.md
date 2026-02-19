# Chapter 16: Non-Parametric Tests

## Overview

Non-parametric tests are statistical methods that do not assume a specific parametric form (such as normality) for the underlying population distribution. They are also called **distribution-free tests** because their validity does not depend on the data following a particular distribution. This chapter provides a comprehensive treatment of rank-based and sign-based tests for one-sample, paired-sample, two-sample, and multi-group settings, along with non-parametric correlation measures.

---

## Chapter Structure

### 16.1 Foundations

This section establishes the theoretical groundwork for non-parametric testing, covering when and why to use these methods, and how they compare to their parametric counterparts:

- **When and Why to Use Non-Parametric Tests** --- Motivates the use of distribution-free methods when normality is violated, data are ordinal, sample sizes are small, or outliers are present.
- **Ranks and Rank Transformations** --- Introduces the core mechanism underlying most non-parametric tests: replacing raw observations with their ranks to reduce the influence of extreme values.
- **Power Comparison with Parametric Tests** --- Quantifies the efficiency trade-off between non-parametric and parametric tests, including the asymptotic relative efficiency (ARE) of rank-based procedures.

### 16.2 One-Sample Non-Parametric Tests

Tests designed for a single sample or for assessing properties of a univariate sequence:

- **Runs Test for Randomness** --- Tests whether a binary sequence is random by counting the number of maximal consecutive runs of identical elements (Wald--Wolfowitz test).
- **Sign Test** --- A simple test for the population median that uses only the signs of deviations from a hypothesized value, requiring minimal assumptions.
- **Wilcoxon Signed-Rank Test** --- A more powerful alternative to the sign test that incorporates both the signs and magnitudes of deviations from the hypothesized median.
- **Binomial Test** --- An exact test for whether a proportion matches a hypothesized value, based on the binomial distribution.
- **One-Sample Tests Overview** --- A unified summary of all one-sample non-parametric tests with detailed procedures, formulas, and worked examples including the runs test, sign test, and Wilcoxon signed-rank test.

### 16.3 Paired-Sample Non-Parametric Tests

Non-parametric alternatives to the paired t-test for dependent or matched data:

- **Paired Sign Test** --- Applies the sign test to paired differences, testing whether the median difference is zero using only the direction of change.
- **Wilcoxon Signed-Rank for Paired Data** --- Applies the Wilcoxon signed-rank procedure to paired differences, exploiting both sign and magnitude information for greater power.
- **Paired Permutation Test** --- Uses resampling-based permutation logic to test for a treatment effect in paired designs without distributional assumptions.
- **Paired Tests Overview** --- A comprehensive summary of paired-sample non-parametric methods with step-by-step procedures, decision criteria, and practical examples.

### 16.4 Two-Sample Non-Parametric Tests

Distribution-free tests for comparing two independent groups:

- **Wilcoxon Rank-Sum Test** --- Tests whether two independent samples come from the same distribution by comparing the sum of ranks assigned to each group.
- **Mann-Whitney U Test** --- An equivalent formulation of the rank-sum test that counts the number of pairwise wins between groups, with explicit handling of tied observations.
- **Kolmogorov-Smirnov Two-Sample Test** --- Tests whether two samples come from the same continuous distribution by measuring the maximum difference between their empirical CDFs.
- **Two-Sample Tests Overview** --- A unified reference covering the Mann-Whitney U test and rank-sum test procedures, including the U-statistic computation, normal approximation, and interpretation guidelines.

### 16.5 Multi-Group Non-Parametric Tests

Extensions to three or more independent or related groups:

- **Kruskal-Wallis Test** --- The non-parametric counterpart to one-way ANOVA, testing whether multiple independent groups share the same distribution by comparing mean ranks.
- **Friedman Test (Repeated Measures)** --- The non-parametric counterpart to repeated-measures ANOVA, testing for differences across related groups using within-block rankings.
- **Mood's Median Test** --- A simple chi-square-based test for whether multiple groups share the same median, using counts above and below the grand median.
- **Post-Hoc Dunn's Test** --- A pairwise multiple-comparison procedure used after a significant Kruskal-Wallis test, with p-value adjustments for multiple testing.

### 16.6 Non-Parametric Correlation

Rank-based measures of association that do not require bivariate normality:

- **Spearman's rho (Revisited)** --- A rank-based correlation coefficient that measures the monotonic association between two variables by computing the Pearson correlation on their ranks.
- **Kendall's tau (Revisited)** --- A concordance-based correlation coefficient that measures association by counting concordant and discordant pairs of observations.

### 16.7 Code

Complete Python implementations:

- **runs_test.py** --- Implementation of the Wald-Wolfowitz runs test for randomness.
- **sign_test.py** --- Implementation of the sign test for a population median.
- **wilcoxon_tests.py** --- Implementation of Wilcoxon signed-rank and rank-sum tests.
- **two_sample_tests.py** --- Two-sample and multi-group non-parametric test implementations.
- **nonparametric_suite.py** --- A comprehensive test suite combining all non-parametric methods.

### 16.8 Exercises

Practice problems covering the application and interpretation of non-parametric tests across one-sample, paired-sample, two-sample, and multi-group scenarios.

---

## Prerequisites

This chapter builds on:

- **Chapter 9** (Hypothesis Testing) --- Null and alternative hypotheses, p-values, test statistics, significance levels, and Type I/II errors.
- **Chapter 5** (Sampling Distributions) --- Normal approximations for large-sample versions of rank-based test statistics.
- **Chapter 11** (ANOVA) --- One-way and repeated-measures ANOVA as parametric counterparts to Kruskal-Wallis and Friedman tests.
- **Chapter 12** (Correlation and Causation) --- Pearson, Spearman, and Kendall correlation foundations.
- **Chapter 14** (Normality Tests) --- Methods for determining when parametric assumptions fail and non-parametric alternatives are needed.

---

## Key Takeaways

1. Non-parametric tests replace raw data with ranks, making them robust to outliers and violations of normality, at the cost of a modest loss of power when parametric assumptions hold.
2. For one-sample and paired problems, the sign test requires the fewest assumptions, while the Wilcoxon signed-rank test is more powerful when the symmetry assumption is met.
3. The Mann-Whitney U test and Wilcoxon rank-sum test are equivalent formulations for comparing two independent groups without assuming equal variances or normality.
4. The Kruskal-Wallis test extends two-sample rank-based testing to multiple groups, with Dunn's test available for post-hoc pairwise comparisons.
5. Spearman's rho and Kendall's tau provide robust alternatives to Pearson's r for measuring monotonic association in the presence of non-linearity or non-normality.
