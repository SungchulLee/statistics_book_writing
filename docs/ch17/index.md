# Chapter 17: Resampling Methods

## Overview

Resampling methods are computationally intensive approaches to statistical inference that construct the sampling distribution of a statistic empirically by repeatedly drawing samples from the observed data. Unlike parametric methods that rely on theoretical distributional assumptions, and unlike rank-based non-parametric methods that replace values with ranks, resampling methods work directly with the original data values. This chapter covers the two primary resampling paradigms --- the bootstrap (for estimation and confidence intervals) and permutation tests (for hypothesis testing) --- along with practical guidance on choosing between them.

---

## Chapter Structure

### 17.1 Bootstrap Foundations

The theoretical basis and core algorithms for bootstrap resampling:

- **The Bootstrap Principle** --- Introduces the key insight that the empirical distribution of the sample is a reasonable proxy for the unknown population distribution, enabling distribution-free inference.
- **Non-Parametric Bootstrap** --- Details the standard bootstrap algorithm of sampling with replacement from the observed data and computing replicate statistics to approximate the sampling distribution.
- **Parametric Bootstrap** --- Describes the variant where bootstrap samples are drawn from a fitted parametric model rather than the empirical distribution, useful when a distributional form is assumed.
- **Resampling Method** --- Provides a comprehensive walkthrough of the bootstrap resampling algorithm, including standard error estimation, the 63.2% unique observation property, and practical implementation steps.
- **Bootstrap Overview** --- A detailed reference covering the bootstrap principle, the substitution of the empirical distribution for the population, the nonparametric bootstrap algorithm, and bootstrap standard error computation.

### 17.2 Bootstrap Confidence Intervals

Methods for constructing confidence intervals from the bootstrap distribution:

- **Percentile Method** --- The simplest approach, using quantiles of the bootstrap distribution directly as confidence interval endpoints.
- **BCa (Bias-Corrected and Accelerated)** --- An improved method that corrects for bias and skewness in the bootstrap distribution, providing better coverage for non-symmetric statistics.
- **Bootstrap-t Method** --- Studentizes the bootstrap distribution by estimating the standard error within each bootstrap sample, yielding intervals with higher-order accuracy.
- **Visualization of Confidence Levels** --- Demonstrates how to visually interpret and compare bootstrap confidence intervals at different confidence levels (e.g., 90% vs. 95%), clarifying the coverage property.
- **Comparison of Bootstrap CI Methods** --- A side-by-side evaluation of the percentile, BCa, and bootstrap-t methods in terms of coverage accuracy, computational cost, and applicability.

### 17.3 Bootstrap Hypothesis Testing

Using bootstrap resampling to perform hypothesis tests:

- **Bootstrap Test for a Single Mean** --- Tests whether a population mean equals a hypothesized value by centering the bootstrap distribution under the null hypothesis.
- **Bootstrap Test for Two Means** --- Tests whether two populations have the same mean by bootstrapping under the null hypothesis of no difference.
- **Bootstrap Test for Correlation** --- Tests whether a population correlation coefficient is zero by resampling paired observations under the null.

### 17.4 Permutation Tests

Exact and approximate tests based on random rearrangement of data labels:

- **Permutation Test Foundations** --- Introduces the logic of permutation testing: under the null hypothesis, group labels are exchangeable, and shuffling them generates the null distribution of any test statistic.
- **Permutation Test for Two-Sample Location** --- Tests whether two independent groups differ in location (typically means) by repeatedly shuffling group labels and computing the difference statistic, with applications to A/B testing.
- **Permutation Test for Correlation** --- Tests the significance of a correlation coefficient by permuting one variable while holding the other fixed, breaking any true association.
- **Permutation Test for Paired Data** --- Adapts permutation logic to paired designs by randomly flipping the signs of paired differences under the null hypothesis of no treatment effect.
- **Permutation Tests Overview** --- A comprehensive reference covering the general permutation test framework, hypotheses, step-by-step algorithm, and the distinction between exact and approximate permutation tests.

### 17.5 Comparison and Practical Guidance

When and how to choose between resampling approaches:

- **Bootstrap vs. Permutation Tests** --- Contrasts the two methods across purpose (estimation vs. testing), resampling mechanism (with vs. without replacement), assumptions (representativeness vs. exchangeability), and typical outputs.
- **Number of Resamples and Convergence** --- Provides guidance on choosing the number of bootstrap replicates or permutations (typically 1,000--10,000) and how to assess whether results have stabilized.
- **When Resampling Fails** --- Discusses limitations and failure modes of resampling, including small samples, extreme quantiles, dependent data, and non-representative samples.
- **Comparison Overview** --- A detailed side-by-side comparison table covering purpose, methodology, assumptions, outputs, strengths, and weaknesses of bootstrap and permutation test approaches.

### 17.6 Code

Complete Python implementations:

- **bootstrap_ci.py** --- Bootstrap confidence interval methods (percentile, BCa, bootstrap-t).
- **bootstrap_tests.py** --- Bootstrap hypothesis tests for means and correlations.
- **permutation_tests.py** --- Permutation test implementations for two-sample, paired, and correlation settings.
- **resampling_methods.py** --- Unified comparison of resampling methods across different scenarios.
- **ab_testing_permutation.py** --- A/B testing application using permutation tests.
- **bootstrap_ci_visualization.py** --- Visualization of bootstrap confidence intervals at multiple confidence levels.
- **bootstrap_median.py** --- Bootstrap inference for the median, a statistic with no simple parametric standard error.

### 17.7 Exercises

Practice problems covering conceptual understanding (why sampling with replacement, exchangeability assumptions, BCa vs. percentile intervals) and computational exercises (bootstrap CIs for skewed distributions, permutation tests for A/B experiments, convergence analysis).

---

## Prerequisites

This chapter builds on:

- **Chapter 8** (Confidence Intervals) --- Parametric confidence interval construction, coverage probability, and interpretation.
- **Chapter 9** (Hypothesis Testing) --- Null and alternative hypotheses, p-values, Type I and II errors, and test statistic logic.
- **Chapter 5** (Sampling Distributions) --- The concept of a sampling distribution and the distinction between a statistic and a parameter.
- **Chapter 6** (Statistical Estimation) --- Maximum likelihood estimation and properties of estimators (bias, variance, consistency).
- **Chapter 16** (Non-Parametric Tests) --- Rank-based distribution-free methods as an alternative approach to non-parametric inference.

---

## Key Takeaways

1. The bootstrap approximates the sampling distribution of any statistic by resampling with replacement from the observed data, enabling standard errors and confidence intervals without closed-form derivations.
2. The BCa bootstrap confidence interval corrects for bias and skewness and generally provides better coverage than the simple percentile method, especially for skewed statistics like the median.
3. Permutation tests provide exact p-values under the null hypothesis of exchangeability by generating the null distribution through random rearrangement of group labels.
4. Bootstrap is primarily for estimation (confidence intervals, standard errors), while permutation tests are primarily for hypothesis testing (p-values), though both can serve overlapping roles.
5. Resampling methods can fail when the sample is too small to represent the population, when estimating extreme quantiles, or when observations are dependent, and the number of resamples must be large enough for results to converge.
