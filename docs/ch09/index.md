# Chapter 9: Hypothesis Testing

## Overview

Hypothesis testing provides a formal framework for making decisions about population parameters based on sample data. This chapter covers the complete hypothesis testing pipeline: formulating null and alternative hypotheses, computing test statistics and p-values, performing one-sample, two-sample, and paired-sample tests for means, proportions, and variances, analyzing errors and power, and applying multiple testing corrections when many hypotheses are tested simultaneously.

---

## Chapter Structure

### 9.1 Foundations

The conceptual and mathematical framework underlying all hypothesis tests:

- **Null and Alternative Hypotheses** -- Introduces the hypothesis testing paradigm using a courtroom analogy, defines simple and composite hypotheses, and distinguishes one-tailed from two-tailed alternatives.
- **Test Statistics and p-values** -- Defines test statistics (z, t, chi-square, F), explains how p-values quantify the strength of evidence against the null, and covers how to compute and interpret them.
- **Significance Level and Decision Rules** -- Describes the significance level alpha as the Type I error threshold, and compares the p-value approach with the critical value approach for making rejection decisions.

### 9.2 One-Sample Tests

Tests for a single population parameter against a hypothesized value:

- **Z-Test for the Mean (Known sigma)** -- Tests whether the population mean equals a specified value when the population standard deviation is known, using the standard normal distribution.
- **t-Test for the Mean (Unknown sigma)** -- Tests the population mean when sigma is unknown, using the t-distribution with n minus 1 degrees of freedom and discussing robustness to non-normality.
- **Z-Test for a Proportion** -- Tests whether a population proportion equals a hypothesized value using the normal approximation, with conditions on the minimum expected counts.
- **Chi-Square Test for the Variance** -- Tests whether the population variance equals a hypothesized value using the chi-square distribution, with a warning about its sensitivity to non-normality.
- **One-Sample Tests Overview** -- A comprehensive reference page consolidating all one-sample test procedures with detailed derivations and worked examples.

### 9.3 Two-Sample Tests

Tests for comparing parameters from two independent populations:

- **Two-Sample Z-Test for the Difference of Means** -- Tests whether two population means differ when both population variances are known.
- **Two-Sample t-Test (Pooled and Welch)** -- Compares means of two independent samples using either the pooled t-test (equal variances assumed) or Welch's t-test (unequal variances), with formulas for the pooled standard deviation and Welch-Satterthwaite degrees of freedom.
- **Two-Sample Z-Test for the Difference of Proportions** -- Tests whether two population proportions differ using a pooled proportion under the null hypothesis.
- **F-Test for the Ratio of Two Variances** -- Tests equality of two population variances using the F-distribution, with a caution about its high sensitivity to non-normality.
- **Two-Sample Tests Overview** -- A comprehensive reference page consolidating all two-sample test procedures with detailed derivations, decision rules, and worked examples.

### 9.4 Paired-Sample Tests

Tests for data where observations are naturally paired:

- **Paired t-Test for the Mean Difference (mu_D)** -- Applies the one-sample t-test to paired differences, testing whether the mean difference is zero, with formulas for the test statistic and degrees of freedom.
- **When to Use Paired vs Two-Sample Tests** -- Provides guidance on selecting between paired and independent designs based on study design, the presence of natural pairing, and the goal of reducing within-subject variability.

### 9.5 Errors and Power

The consequences of incorrect decisions and how to design studies with adequate sensitivity:

- **Type I and Type II Errors** -- Defines false positives (rejecting a true null) and false negatives (failing to reject a false null), with a summary decision table and practical examples.
- **Power Analysis** -- Defines statistical power as 1 minus beta, identifies the four factors that determine power (alpha, sample size, effect size, and population variability), and explains how to conduct a priori power calculations.
- **CI and Test Duality** -- Establishes the equivalence between a two-sided hypothesis test at level alpha and a (1 minus alpha) confidence interval, showing that each can be derived from the other.

### 9.6 Multiple Testing

Corrections for the inflation of false positives when many hypotheses are tested at once:

- **Family-Wise Error Rate (FWER)** -- Defines FWER as the probability of at least one false rejection among m tests and shows how it grows rapidly with the number of tests.
- **Bonferroni and Holm Corrections** -- Presents the Bonferroni correction (reject at alpha/m) and the uniformly more powerful Holm step-down procedure.
- **False Discovery Rate (Benjamini-Hochberg)** -- Introduces FDR as the expected proportion of false discoveries and describes the Benjamini-Hochberg procedure for controlling it.

### 9.7 Code

Complete Python implementations:

- **hypothesis_tests.py** -- Comprehensive demonstrations of hypothesis testing across multiple settings.
- **power_analysis.py** -- Power analysis and sample size determination utilities.
- **multiple_testing.py** -- Implementations of Bonferroni, Holm, and Benjamini-Hochberg corrections.
- **test_mean.py** -- One-sample mean test (z-test and t-test).
- **test_proportion.py** -- One-sample proportion z-test.
- **test_variance.py** -- One-sample chi-square variance test.
- **test_var_ratio.py** -- F-test for comparing two variances.
- **test_paired.py** -- Paired-sample t-test.
- **test_two_means.py** -- Two-sample t-test (pooled and Welch).
- **test_two_props.py** -- Two-sample proportion z-test.

### 9.8 Exercises

Practice problems covering hypothesis formulation, Type I and Type II error identification, test statistic computation, p-value interpretation, power analysis, and multiple testing scenarios.

---

## Prerequisites

This chapter builds on:

- **Chapter 5** (Sampling Distributions) -- The normal, t, chi-square, and F distributions used as reference distributions for test statistics.
- **Chapter 7** (Estimation of mu and sigma squared) -- Properties of the sample mean and sample variance as estimators.
- **Chapter 8** (Confidence Intervals) -- The construction of confidence intervals, which are dual to hypothesis tests via the CI-test duality.

---

## Key Takeaways

1. Hypothesis testing is a systematic procedure for deciding whether sample data provide sufficient evidence to reject a claim about a population parameter, using the null hypothesis as the default assumption.
2. The p-value measures the probability of observing data as extreme as or more extreme than what was observed, assuming the null hypothesis is true; it is not the probability that the null is true.
3. One-sample, two-sample, and paired-sample tests share the same logical structure but differ in the test statistic and reference distribution used.
4. Type I and Type II errors represent the two ways a test can go wrong; increasing sample size or effect size increases power (reduces Type II error) without inflating the Type I error rate.
5. When testing multiple hypotheses simultaneously, corrections such as Bonferroni, Holm, or Benjamini-Hochberg are essential to control the overall error rate and avoid false discoveries.
