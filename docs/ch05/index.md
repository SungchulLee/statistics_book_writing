# Chapter 5: Sampling Distributions

## Overview

This chapter bridges probability theory and statistical inference by showing that statistics computed from random samples are themselves random variables with their own distributions. It introduces the four fundamental sampling distributions (Normal, Student's $t$, chi-square, and $F$), derives the sampling distributions of the most commonly used statistics, and demonstrates how sample size governs the precision of estimation through the standard error.

---

## Chapter Structure

### 5.1 Foundations

The conceptual groundwork for understanding why and how statistics vary from sample to sample:

- **Statistics as Random Variables** --- Establishes that any function of observed data is a random variable whose value changes from sample to sample, forming the conceptual foundation of all sampling-distribution theory.
- **Repeated Sampling Concept** --- Explains how drawing multiple random samples and computing a statistic for each one produces a sampling distribution, which is fundamental to estimating population parameters, constructing confidence intervals, and performing hypothesis tests.

### 5.2 The Four Fundamental Sampling Distributions

The four probability distributions that serve as reference distributions for most classical inference procedures:

- **Normal Distribution (Z)** --- The standard normal distribution arises from standardizing normally distributed statistics and serves as the large-sample approximation for a wide variety of estimators via the Central Limit Theorem.
- **Student's t Distribution** --- Arises when estimating the mean of a normal population using the sample standard deviation instead of the known population standard deviation, accounting for the additional uncertainty with heavier tails that converge to the normal as degrees of freedom increase.
- **Chi-Square Distribution** --- The distribution of a sum of squared standard normal random variables, playing a central role in inference about population variance, goodness-of-fit tests, and tests of independence.
- **F Distribution** --- Defined as the ratio of two independent chi-square random variables divided by their degrees of freedom, fundamental for comparing variances between populations and for Analysis of Variance (ANOVA).

### 5.3 Applications to Common Statistics

How the fundamental distributions connect to the sampling behavior of statistics used in everyday practice:

- **Sampling Distribution of the Mean** --- Derives the distribution of $\bar{X}$ across repeated samples, showing it is unbiased for $\mu$ with variance $\sigma^2/n$, and is normally distributed when the population is normal (or approximately so by the CLT for large $n$).
- **Standard Error** --- Clarifies the distinction between standard deviation (spread of individual observations) and standard error (spread of a sample statistic), showing that SE quantifies how much a statistic varies from sample to sample.
- **Sampling Distribution of Proportions** --- Derives the distribution of the sample proportion $\hat{p}$ for binary data, establishing its mean $p$, variance $p(1-p)/n$, and normal approximation for large samples.
- **Sampling Distribution of the Variance** --- Shows how $S^2$ behaves across repeated samples, connecting the scaled sample variance to the chi-square distribution under normality and explaining Bessel's correction for unbiasedness.
- **Difference of Two Sample Means** --- Derives the sampling distribution of $\bar{X}_1 - \bar{X}_2$ under various scenarios (known variances, unknown equal variances, unknown unequal variances via Welch's approximation).
- **Difference of Two Sample Proportions** --- Establishes the sampling distribution of $\hat{p}_1 - \hat{p}_2$ for comparing proportions from two independent populations, with the normal approximation and pooled standard error for hypothesis testing.

### 5.4 Visualization

Graphical demonstrations that build intuition for sampling distributions:

- **Sampling Distribution Visualization** --- Demonstrates how the sampling distribution of the sample mean becomes more concentrated as sample size increases, using realistic income data to distinguish between the population distribution, a single sample distribution, and the sampling distribution.

### 5.5 Code

Complete Python implementations for simulating and visualizing sampling distributions:

- **Sampling Distribution of X-bar (Uniform)** --- Simulates the sampling distribution of the mean from a uniform population.
- **Sampling Distribution of X-bar (Exponential)** --- Simulates the sampling distribution of the mean from an exponential population.
- **Sampling Distribution of X-bar (Normal)** --- Simulates the sampling distribution of the mean from a normal population.
- **Sampling Distribution of X-bar (Bernoulli)** --- Simulates the sampling distribution of the mean from a Bernoulli population.
- **Sampling Distribution of S-squared (Normal)** --- Simulates the sampling distribution of the sample variance from a normal population.
- **Standard Error of X-bar** --- Demonstrates how the standard error of the sample mean decreases with sample size.
- **Standard Error of S-squared** --- Illustrates the variability of the sample variance estimator.
- **Sampling Distribution Income Visualization** --- Uses real loan income data to visualize the three-way distinction between population, sample, and sampling distributions.

### 5.6 Exercises

Practice problems covering the capture-recapture method, properties of sampling distributions, standard error calculations, and applications of the fundamental sampling distributions.

---

## Prerequisites

This chapter builds on:

- **Chapter 3** (Foundations of Probability) --- Random variables, expectation, variance, the Law of Large Numbers, and the Central Limit Theorem.
- **Chapter 4** (Distributions) --- Properties of the normal, binomial, Poisson, and exponential distributions, as well as joint distributions and independence.

---

## Key Takeaways

1. A statistic is a random variable; its probability distribution across repeated samples is its sampling distribution, which is the basis for all inferential procedures.
2. The four fundamental sampling distributions (Normal, $t$, $\chi^2$, $F$) arise naturally from normal random samples and serve as the reference distributions for confidence intervals, hypothesis tests, and ANOVA.
3. The sampling distribution of $\bar{X}$ is centered at $\mu$ with standard error $\sigma/\sqrt{n}$, and is approximately normal for large $n$ regardless of the population shape (by the CLT).
4. The standard error measures precision of estimation: it decreases at rate $1/\sqrt{n}$, meaning quadrupling the sample size halves the standard error.
5. The sampling distributions of differences ($\bar{X}_1 - \bar{X}_2$ and $\hat{p}_1 - \hat{p}_2$) extend these ideas to two-sample comparisons, forming the foundation for two-sample inference in later chapters.
