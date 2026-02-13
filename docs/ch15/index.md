# Chapter: Variance Tests

## Overview

Variance tests evaluate whether observed differences in variability across groups or populations are statistically significant. These tests are fundamental tools in statistical inference, playing a critical role in validating assumptions for other methods such as ANOVA and regression analysis.

This chapter covers a comprehensive suite of variance testing methods, ranging from classical parametric approaches to robust and non-parametric alternatives, as well as advanced computational and Bayesian techniques.

## Chapter Contents

### Introduction to Variance Testing
A survey of the major variance tests—Chi-Square, F-test, Bartlett's, Levene's, Brown–Forsythe, and Fligner–Killeen—including their assumptions, strengths, and appropriate use cases.

### Chi-Square Test for Variance
A one-sample test for determining whether a population variance equals a hypothesized value, with derivation of the test statistic, critical regions, worked examples, and confidence intervals for variance.

### F-Test for Comparing Two Variances
A two-sample test based on the ratio of sample variances, with full derivation showing the connection to the chi-square and F distributions, critical region analysis, and Python implementation.

### Bartlett's Test for Equality of Variances
A multi-group parametric test for homogeneity of variances under the assumption of normality, with the pooled-variance-based test statistic, limitations, and both library-based and manual Python implementations.

### Robust Tests for Equality of Variances
Levene's test (mean-based), the Brown–Forsythe test (median-based), and the non-parametric Fligner–Killeen test (rank-based), all designed to handle non-normal data and outliers more gracefully than parametric alternatives.

### Advanced Methods for Variance Testing
Bootstrap resampling and Bayesian inference approaches for variance comparison, useful when parametric assumptions are violated, sample sizes are small, or prior knowledge is available.

### Applications in Regression and ANOVA
Practical application of variance tests to check homoscedasticity in regression (Breusch–Pagan test) and homogeneity of variances in ANOVA (Levene's test), with solutions for detected violations including Welch's ANOVA and robust standard errors.

### Exercises
Practice problems covering F-tests, Levene's test, and interpretation of conflicting test results.

## Prerequisites

- Chi-square, F, and normal distributions (Chapter 5)
- Hypothesis testing framework (Chapter 9)
- Confidence intervals (Chapter 8)
- Basic Python with NumPy and SciPy
