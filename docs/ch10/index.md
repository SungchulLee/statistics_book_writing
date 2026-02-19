# Chapter 10: Chi-Square Tests

## Overview

Chi-square tests are the primary tools for analyzing categorical data. This chapter develops the connection between the chi-square distribution and categorical frequency data, then applies it to three major tests: the goodness-of-fit test (does an observed distribution match a hypothesized one?), the test of independence (are two categorical variables associated?), and the test of homogeneity (do multiple populations share the same distribution?). Practical considerations such as expected cell count requirements and effect size measurement with Cramer's V are also covered.

---

## Chapter Structure

### 10.1 Chi-Square Distribution and Asymptotics

The theoretical foundation connecting the chi-square distribution to categorical data analysis:

- **Chi-Square Distribution** -- Derives the chi-square distribution as the sum of squared standard normal variables, establishes its connection to multinomial count data, and shows how the Pearson chi-square statistic arises from the normal approximation to cell counts.
- **Degrees of Freedom and Asymptotic Theory** -- Explains how degrees of freedom are computed for goodness-of-fit tests (k minus 1), independence tests ((r minus 1) times (c minus 1)), and homogeneity tests, and discusses the asymptotic validity of the chi-square approximation.

### 10.2 Chi-Square Tests for Categorical Data

The three major applications of the chi-square test to categorical frequency data:

- **Goodness-of-Fit Test** -- Tests whether an observed frequency distribution matches a hypothesized distribution across k categories, using a frequency table of observed versus expected counts and the Pearson chi-square statistic.
- **Test of Independence** -- Tests whether two categorical variables are associated within a single sample, using a contingency table to compare observed cell frequencies with expected frequencies computed under the independence assumption.
- **Test of Homogeneity** -- Tests whether multiple independent populations share the same distribution of a categorical variable, using the same chi-square computation as the independence test but with a different sampling design and interpretation.

### 10.3 Practical Considerations

Guidelines for ensuring the validity and interpretability of chi-square test results:

- **Expected Cell Counts and Validity Conditions** -- States the rule of thumb that all expected cell frequencies should be at least 5, explains why the condition applies to expected (not observed) counts, and describes alternatives (combining categories, Fisher's exact test) when the condition is violated.
- **Effect Size and Cramer's V** -- Introduces Cramer's V as a standardized measure of association strength, ranging from 0 (no association) to 1 (perfect association), and provides interpretation guidelines for small, medium, and large effect sizes.

### 10.4 Code

Complete Python implementations:

- **gof_manual.py** -- Manual (step-by-step) computation of the chi-square goodness-of-fit test.
- **gof_scipy.py** -- Goodness-of-fit test using scipy.stats.chisquare.
- **independence_manual.py** -- Manual computation of the independence test with a visualization of observed versus expected counts.
- **independence_template.py** -- Reusable template function for performing chi-square independence tests.
- **homogeneity_basic.py** -- Homogeneity test using scipy.stats.chi2_contingency.
- **homogeneity_residuals.py** -- Residual heatmap visualization for diagnosing which cells drive a significant homogeneity test result.
- **fisher_exact.py** -- Fisher's exact test for 2x2 tables when expected counts are too small for the chi-square approximation.
- **mcnemar_test.py** -- McNemar's test for paired binary data.
- **cochran_q.py** -- Cochran's Q test for comparing k related binary outcomes.

### 10.5 Exercises

Practice problems covering goodness-of-fit tests for distributional assumptions, independence tests on contingency tables, validity condition checks, and test statistic computation.

---

## Prerequisites

This chapter builds on:

- **Chapter 5** (Sampling Distributions) -- The chi-square distribution, its properties, and its role as a sampling distribution.
- **Chapter 9** (Hypothesis Testing) -- The general framework of null and alternative hypotheses, test statistics, p-values, and decision rules.

---

## Key Takeaways

1. The Pearson chi-square statistic measures the discrepancy between observed and expected categorical counts and follows an approximate chi-square distribution under the null hypothesis when expected counts are sufficiently large.
2. The goodness-of-fit test assesses whether a single categorical variable follows a hypothesized distribution, while the independence and homogeneity tests assess relationships between two categorical variables under different sampling designs.
3. The test of independence and the test of homogeneity use identical computations but differ in the research question and data collection design: one sample classified by two variables versus multiple samples compared on one variable.
4. Validity of the chi-square approximation requires all expected cell counts to be at least 5; when this condition is violated, Fisher's exact test or category collapsing should be used instead.
5. Statistical significance alone does not indicate practical importance; Cramer's V provides a standardized effect size that quantifies the strength of the association independent of sample size.
