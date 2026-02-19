# Chapter 15: Variance Tests

## Overview

Variance tests evaluate whether observed differences in variability across groups or populations are statistically significant. These tests are fundamental tools in statistical inference, playing a critical role in validating assumptions for other methods such as ANOVA and regression analysis. This chapter covers a comprehensive suite of variance testing methods, ranging from classical parametric approaches (chi-square, F-test, Bartlett's) through robust alternatives (Levene's, Brown-Forsythe, Fligner-Killeen) to advanced computational and Bayesian techniques.

---

## Chapter Structure

### 15.1 Introduction to Variance Testing

Motivation and overview of the variance testing landscape:

- **Why Test Variances** -- Explains the practical importance of variance testing for checking homoscedasticity in ANOVA, validating regression assumptions, and comparing volatility across financial instruments.
- **Overview of Variance Tests** -- Surveys the major variance tests (chi-square, F-test, Bartlett's, Levene's, Brown-Forsythe, Fligner-Killeen), including their test statistics, distributional assumptions, and appropriate use cases.
- **Assumptions Common to Variance Tests** -- Discusses the shared requirements of independence, random sampling, and (for parametric tests) normality, and the consequences of violating these assumptions.

### 15.2 Chi-Square Test for Variance

A one-sample test for whether a population variance equals a specified value:

- **One-Sample Chi-Square Variance Test** -- Tests $H_0: \sigma^2 = \sigma_0^2$ using the statistic $\chi^2 = (n-1)s^2 / \sigma_0^2$, with one-tailed and two-tailed formulations and worked examples.
- **Derivation and Distribution Theory** -- Shows how the test statistic arises from the distribution of the sample variance under normality, connecting to the chi-square distribution with $n-1$ degrees of freedom.
- **Confidence Interval for $\sigma^2$** -- Inverts the chi-square test to construct confidence intervals for the population variance and standard deviation.

### 15.3 F-Test for Comparing Two Variances

A two-sample test based on the ratio of sample variances:

- **Two-Sample F-Test** -- Tests $H_0: \sigma_1^2 = \sigma_2^2$ using the statistic $F = s_1^2 / s_2^2$, which follows an $F$-distribution under the null hypothesis when both populations are normal.
- **F-Distribution and Degrees of Freedom** -- Details the properties of the $F$-distribution and how the numerator and denominator degrees of freedom are determined.
- **Sensitivity to Non-Normality** -- Warns that the F-test is highly sensitive to departures from normality, often producing inflated Type I error rates with non-normal data.

### 15.4 Bartlett's Test

A multi-group parametric test for homogeneity of variances:

- **Bartlett's Test for Equality of Variances** -- Tests $H_0: \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2$ using a pooled-variance-based statistic that follows a $\chi^2_{k-1}$ distribution under normality.
- **Derivation and Chi-Square Approximation** -- Derives the Bartlett test statistic from the ratio of pooled to individual variances with the correction factor for small samples.
- **Limitations Under Non-Normality** -- Emphasizes that Bartlett's test is highly sensitive to non-normality, making it unreliable when the normality assumption is questionable.

### 15.5 Robust Tests

Distribution-free and outlier-resistant alternatives for comparing variances:

- **Levene's Test** -- Tests equality of variances by performing a one-way ANOVA on the absolute deviations of observations from their group means; robust to moderate departures from normality.
- **Brown-Forsythe Test** -- A variant of Levene's test that uses deviations from group medians instead of means, providing additional robustness to skewed distributions and outliers.
- **Fligner-Killeen Test** -- A non-parametric rank-based test that uses ranks of absolute deviations from group medians, offering the strongest robustness among the three methods.
- **Comparison of Robust Methods** -- Side-by-side evaluation of Levene's, Brown-Forsythe, and Fligner-Killeen in terms of Type I error control, power, and robustness across different distribution shapes.

### 15.6 Advanced Methods

Computational and Bayesian approaches for variance comparison:

- **Bootstrap Variance Testing** -- Uses resampling to construct a null distribution for variance ratios or differences, providing valid inference without distributional assumptions.
- **Bayesian Variance Testing** -- Employs prior distributions on variance parameters (typically inverse-gamma) to compute posterior probabilities and Bayes factors for variance hypotheses.
- **Likelihood Ratio Test for Variances** -- Compares the maximized likelihoods under the null and alternative hypotheses, providing an asymptotically chi-squared test statistic.

### 15.7 Applications

Practical use cases where variance testing is essential:

- **Pre-Test for ANOVA Homoscedasticity** -- Uses Levene's or Bartlett's test to verify the equal-variance assumption before conducting ANOVA, with Welch's ANOVA as a fallback when variances differ.
- **Variance Testing in Regression** -- Applies variance tests (e.g., Breusch-Pagan) to regression residuals to detect heteroscedasticity, with remedies including weighted least squares and robust standard errors.
- **Financial Volatility Comparisons** -- Compares the volatility of different assets, portfolios, or time periods to assess risk differences and inform portfolio construction decisions.

### 15.8 Code

Complete Python implementations:

- **Chi-Squared Test for Variance** -- One-sample variance test with critical values and $p$-value computation.
- **F-Test of Equality of Variances** -- Two-sample F-test implementation with visualization of the rejection region.
- **Bartlett's Test** -- Multi-group homogeneity test using scipy and manual computation.
- **Levene's Test** -- Robust variance equality test with mean-based deviations.
- **Chi-Square Distribution** -- Visualization of the chi-square distribution for different degrees of freedom.
- **Robust Variance Tests Comparison** -- Side-by-side comparison of Levene, Brown-Forsythe, and Fligner-Killeen on the same data.
- **F-Test Tail Region Visualization** -- Plots the F-distribution with shaded rejection regions.
- **F-Test Normality Sensitivity and Robust Alternatives** -- Simulation showing how the F-test's Type I error inflates under non-normality, with robust alternatives performing correctly.
- **F-Test Power Simulation** -- Monte Carlo study of F-test power as a function of variance ratio and sample size.
- **Bartlett Test Non-Normality Sensitivity** -- Simulation demonstrating Bartlett's poor performance with skewed or heavy-tailed data.
- **Levene Test Normal vs Skewed Simulation** -- Compares Levene's test performance across normal and non-normal distributions.
- **Brown-Forsythe Test (scipy)** -- Implementation using scipy's median-based Levene variant.
- **Fligner-Killeen Test (scipy)** -- Non-parametric rank-based variance test.
- **Bootstrap Variance Test** -- Resampling-based approach for comparing variances without distributional assumptions.
- **Bayesian Variance Test** -- Posterior inference on variance parameters using conjugate priors.

### 15.9 Exercises

Practice problems covering F-test computation and interpretation, Levene's test application, comparison of classical and robust test results, chi-square confidence intervals for variance, and practical decision-making when test results conflict.

---

## Prerequisites

This chapter builds on:

- **Chapter 5** (Sampling Distributions) -- The chi-square, $F$, and normal distributions and their roles in inference about variances.
- **Chapter 8** (Confidence Intervals) -- Confidence interval construction, particularly intervals for $\sigma^2$ using the chi-square distribution.
- **Chapter 9** (Hypothesis Testing) -- The general framework of null and alternative hypotheses, test statistics, $p$-values, and decision rules.
- **Chapter 14** (Normality Tests) -- Methods for checking whether the normality assumption required by classical variance tests is satisfied.

---

## Key Takeaways

1. Classical variance tests (chi-square, F-test, Bartlett's) are powerful under normality but highly sensitive to non-normal data, often producing misleading results when the normality assumption is violated.
2. Robust alternatives (Levene's, Brown-Forsythe, Fligner-Killeen) maintain proper Type I error rates across a wider range of distributions and should be preferred when normality is uncertain.
3. The Brown-Forsythe test (median-based Levene) offers the best balance of robustness and power for most practical applications, while Fligner-Killeen provides the strongest non-parametric guarantees.
4. Bootstrap and Bayesian methods provide modern alternatives that avoid distributional assumptions entirely or incorporate prior information, respectively.
5. Variance testing is not just an end in itself -- it serves as a prerequisite check for ANOVA (homoscedasticity), regression (constant error variance), and financial analysis (volatility comparison), making it an essential step in many analysis workflows.
