# Chapter 8: Confidence Intervals

## Overview

Confidence intervals quantify the uncertainty in estimating population parameters by providing a range of plausible values together with a specified level of confidence. This chapter develops the theory and construction of confidence intervals for means, proportions, and variances across one-sample, two-sample, and paired-sample designs. It also addresses the practical question of how large a sample is needed to achieve a desired margin of error.

---

## Chapter Structure

### 8.1 Foundations

The foundational theory behind confidence intervals, including what they are, how to interpret them correctly, and the most common misconceptions:

- **Confidence Level and Coverage** -- Defines confidence intervals formally, explains the role of the confidence level (90%, 95%, 99%), and derives the general form as point estimate plus or minus margin of error.
- **Interpretation and Common Pitfalls** -- Clarifies the repeated-sampling interpretation of confidence intervals and debunks the common misconception that a 95% CI means there is a 95% probability that the parameter lies in the interval.

### 8.2 One-Sample Intervals

Confidence intervals for a single population parameter estimated from one sample:

- **CI for the Mean (mu)** -- Constructs z-intervals (known variance) and t-intervals (unknown variance) for the population mean, including validity conditions and the role of the Central Limit Theorem.
- **CI for a Proportion (p)** -- Derives the Wald z-interval for a population proportion, discusses the Wilson and Clopper-Pearson alternatives, and states the normal approximation conditions.
- **CI for the Variance (sigma squared)** -- Uses the chi-square distribution to build a confidence interval for the population variance, requiring the normality assumption.

### 8.3 Two-Sample Intervals

Confidence intervals for comparing parameters from two independent populations:

- **CI for the Difference of Means (mu1 minus mu2)** -- Covers the z-interval (known variances), pooled t-interval (equal variances), and Welch t-interval (unequal variances) using the Welch-Satterthwaite degrees of freedom.
- **CI for the Difference of Proportions (p1 minus p2)** -- Presents the Wald interval for the difference of two proportions and compares it with the Newcombe and Clopper-Pearson alternatives.
- **CI for the Variance Ratio (sigma1 squared over sigma2 squared)** -- Constructs an F-distribution-based confidence interval for the ratio of two population variances.

### 8.4 Paired-Sample Intervals

Confidence intervals for data where observations are naturally paired:

- **CI for the Mean of Differences (mu_D)** -- Applies the one-sample t-interval to the paired differences, estimating the mean difference between two related measurements.
- **When to Use Paired vs Independent Designs** -- Provides guidance on choosing between paired and independent designs based on expected within-pair correlation and study logistics.
- **Paired Interval for Proportions (McNemar)** -- Addresses paired binary data using McNemar's approach, focusing on discordant pairs to construct a confidence interval for the difference in paired proportions.

### 8.5 Sample Size Determination

Formulas for determining the minimum sample size required to achieve a target precision:

- **Sample Size for Desired Margin of Error** -- Derives sample size formulas for estimating a mean and a proportion with a specified margin of error and confidence level.
- **Sample Size for Comparing Two Groups** -- Extends sample size calculations to two-sample designs for comparing means and proportions, incorporating effect size and desired power.

### 8.6 Code

Complete Python implementations:

- **confidence_intervals.py** -- Comprehensive demonstrations of confidence interval construction across multiple settings.
- **sample_size.py** -- Sample size calculation utilities for one-sample and two-sample designs.
- **ci_mean_sim.py** -- Monte Carlo simulation verifying coverage of the mean confidence interval.
- **ci_prop_sim.py** -- Coverage simulation for the proportion confidence interval.
- **ci_var_sim.py** -- Coverage simulation for the variance confidence interval using the chi-square distribution.
- **ci_paired_sim.py** -- Coverage simulation for the paired-sample mean difference interval.
- **ci_diff_means_sim.py** -- Coverage simulation for the two-sample difference of means interval.
- **ci_mean_calc.py** -- One-sample mean CI computation with step-by-step output.
- **ci_prop_calc.py** -- One-sample proportion CI computation with step-by-step output.

### 8.7 Exercises

Practice problems covering CI interpretation, construction of one-sample and two-sample intervals, sample size determination, and the relationship between confidence level and interval width.

---

## Prerequisites

This chapter builds on:

- **Chapter 5** (Sampling Distributions) -- The normal, t, chi-square, and F sampling distributions that serve as the basis for constructing pivotal quantities.
- **Chapter 6** (Statistical Estimation) -- Maximum likelihood and method of moments estimators, including properties such as bias and consistency.
- **Chapter 7** (Estimation of mu and sigma squared) -- The sample mean and sample variance as estimators, Bessel's correction, and their sampling distributions.

---

## Key Takeaways

1. A confidence interval provides a range of plausible values for an unknown parameter; the confidence level refers to the long-run coverage rate under repeated sampling, not the probability that any single interval contains the parameter.
2. The choice between z-intervals and t-intervals depends on whether the population variance is known and the sample size; t-intervals are more appropriate when sigma is unknown and the sample is small.
3. Two-sample intervals allow comparison of means, proportions, and variances across independent groups, with Welch's approach providing robustness when variances are unequal.
4. Paired-sample intervals exploit the within-pair correlation to reduce variability and produce narrower intervals than independent-sample designs when pairing is meaningful.
5. Sample size formulas translate a desired margin of error and confidence level into the minimum number of observations required, enabling researchers to plan studies with sufficient precision.
