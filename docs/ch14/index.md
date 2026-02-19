# Chapter 14: Normality Tests

## Overview

Many statistical methods -- $t$-tests, ANOVA, linear regression, confidence intervals -- assume that data (or residuals) follow a normal distribution. Before applying these methods, practitioners need tools to check whether the normality assumption is reasonable. This chapter provides a complete toolkit for assessing normality, from visual diagnostics and descriptive statistics through formal hypothesis tests, along with strategies for dealing with data that fails the normality assumption.

---

## Chapter Structure

### 14.1 Introduction to Normality

Why normality matters and where it arises in statistical practice:

- **What Is Normality and Why It Matters** -- Defines the normal distribution, explains its symmetric bell-shaped density, and describes why so many statistical methods depend on it through the Central Limit Theorem.
- **Central Role in Statistical Inference** -- Explains how normality underpins confidence intervals, hypothesis tests, and prediction intervals, and what goes wrong when the assumption is violated.
- **Normality in Financial Data** -- Discusses why financial models (Black-Scholes, VaR, portfolio optimization) assume normality and the practical consequences when asset returns deviate from this assumption.

### 14.2 Graphical Methods

Visual approaches for assessing distributional shape:

- **Histogram and Density Plots** -- Overlays the empirical histogram with a kernel density estimate and the theoretical normal PDF to judge whether the data's shape matches the bell curve.
- **Q-Q Plots** -- Compares sample quantiles against theoretical normal quantiles; points falling along a straight diagonal line indicate normality, while systematic deviations reveal skewness or heavy tails.
- **Boxplots and Their Interpretation** -- Uses the median, quartiles, and whisker symmetry of a boxplot to detect skewness and outliers that suggest departures from normality.
- **Q-Q Plots for Financial Returns** -- Applies Q-Q plots specifically to asset return data, illustrating the characteristic heavy tails and the implications for tail risk estimation.

### 14.3 Descriptive Statistics as Normality Indicators

Numerical summaries that quantify departures from the normal shape:

- **Skewness and Kurtosis** -- Defines sample skewness (asymmetry) and kurtosis (tail heaviness) and explains their expected values under normality (0 and 3, respectively).
- **Skewtest and Kurtosistest** -- Formal $z$-score-based tests that evaluate whether the observed skewness or kurtosis significantly deviates from the normal distribution's theoretical values.
- **D'Agostino's K-Squared Test** -- Combines the skewness and kurtosis $z$-scores into a single chi-squared test statistic with 2 degrees of freedom to assess overall normality.
- **Jarque-Bera Test** -- A widely used test in econometrics that combines skewness and excess kurtosis into the statistic $JB = \frac{n}{6}(S^2 + \frac{(K-3)^2}{4})$, following a $\chi^2_2$ distribution under the null.

### 14.4 Formal Tests for Normality

Rigorous hypothesis tests based on the empirical distribution function or order statistics:

- **Kolmogorov-Smirnov and Lilliefors Tests** -- The K-S test compares the empirical CDF to a fully specified normal CDF; the Lilliefors correction adjusts for the case where mean and variance are estimated from the data.
- **Anderson-Darling Test** -- An enhancement of the K-S test that places greater weight on the tails of the distribution, making it especially sensitive to tail departures from normality.
- **Shapiro-Wilk Test** -- Evaluates normality by comparing ordered sample values to their expected values under normality using optimally weighted linear combinations; widely regarded as the most powerful test for small to moderate sample sizes.

### 14.5 Limitations and Pitfalls

Understanding what normality tests can and cannot tell you:

- **Sample Size Effects on Power** -- With small samples, tests may lack power to detect real departures; with very large samples, tests reject normality for trivially small deviations.
- **Sensitivity vs Practical Significance** -- A statistically significant test result does not necessarily mean the departure from normality is large enough to invalidate downstream analyses.
- **Choosing the Right Test** -- Guidance on selecting among the available tests based on sample size, the type of departure expected, and the analysis context.

### 14.6 Dealing with Non-Normal Data

Strategies for proceeding when the normality assumption fails:

- **Transformations to Achieve Normality** -- Applies log, square root, or Box-Cox transformations to reduce skewness and bring data closer to a normal shape.
- **Bootstrapping as an Alternative** -- Uses resampling with replacement to construct confidence intervals and perform hypothesis tests without requiring distributional assumptions.
- **Non-Parametric Methods** -- Employs rank-based tests (Mann-Whitney U, Kruskal-Wallis, Wilcoxon signed-rank) that do not assume a specific distribution for the data.

### 14.7 Applications

Where normality assessment is required in practice:

- **Normality in t-Tests and ANOVA** -- Explains the normality requirement for $t$-tests and ANOVA, the robustness of these methods to moderate violations, and when alternatives are needed.
- **Normality in Regression (Residual Diagnostics)** -- Applies normality tests to regression residuals to validate the assumptions underlying confidence intervals and prediction intervals.
- **Normality of Financial Returns** -- Examines the empirical evidence that asset returns exhibit heavy tails and excess kurtosis relative to the normal distribution, with implications for risk management.

### 14.8 Code

Complete Python implementations:

- **Graphical Normality Checks** -- Histogram, density plot, and Q-Q plot generation for visual assessment.
- **Formal Normality Test Suite** -- Runs Shapiro-Wilk, Anderson-Darling, K-S, and Lilliefors tests on a dataset and summarizes results.
- **Transformation Demonstrations** -- Log, square root, and Box-Cox transformations with before-and-after normality tests.
- **Q-Q Plot with Normality Tests** -- Combined Q-Q plot with annotated test statistics.
- **Q-Q Plot Confidence Band Simulation** -- Simulated confidence envelopes for Q-Q plots to distinguish significant departures from sampling noise.
- **Distribution Shapes via Boxplots** -- Side-by-side boxplots comparing normal, skewed, and heavy-tailed distributions.
- **Skewness Test** -- Implementation and interpretation of `scipy.stats.skewtest`.
- **Kurtosis Test** -- Implementation and interpretation of `scipy.stats.kurtosistest`.
- **D'Agostino $K^2$ Test** -- Combined skewness-kurtosis normality test.
- **Jarque-Bera Test** -- Step-by-step manual computation alongside the scipy implementation.
- **Kolmogorov-Smirnov Test** -- One-sample K-S test with specified parameters.
- **Lilliefors Test** -- K-S test corrected for estimated parameters.
- **Anderson-Darling Test** -- Tail-weighted goodness-of-fit test.
- **Shapiro-Wilk Test** -- The recommended test for small to moderate samples.
- **Shapiro-Wilk Power Simulation** -- Monte Carlo simulation studying how power varies with sample size and departure type.
- **Q-Q Plot Financial Returns** -- Q-Q analysis applied to real or simulated stock return data.

### 14.9 Exercises

Practice problems covering Jarque-Bera test computation, graphical normality assessment, comparison of formal tests on different distribution shapes, transformations for non-normal data, and residual normality diagnostics in regression.

---

## Prerequisites

This chapter builds on:

- **Chapter 4** (Distributions) -- The normal distribution's properties, PDF, CDF, and its central role among continuous distributions.
- **Chapter 2** (Descriptive Statistics) -- Histograms, boxplots, skewness, and kurtosis as summary measures of distributional shape.
- **Chapter 9** (Hypothesis Testing) -- The framework of null and alternative hypotheses, $p$-values, significance levels, and Type I/II errors.
- **Chapter 5** (Sampling Distributions) -- The chi-square and $t$ distributions that underlie many normality-dependent procedures.

---

## Key Takeaways

1. Normality is a prerequisite for many classical statistical methods; verifying it (or understanding when violations are tolerable) is an essential part of any analysis workflow.
2. Graphical methods (histograms, Q-Q plots, boxplots) provide intuitive first assessments, while formal tests (Shapiro-Wilk, Anderson-Darling, Jarque-Bera) offer rigorous quantitative evidence.
3. No single normality test is universally best: Shapiro-Wilk is generally most powerful for small samples, Anderson-Darling is sensitive to tail departures, and Jarque-Bera is standard in econometrics.
4. With very large samples, every normality test will reject -- the key question becomes whether the departure is practically significant enough to affect downstream inference.
5. When normality fails, practitioners have three main strategies: transform the data, use resampling (bootstrapping), or switch to non-parametric methods that do not require distributional assumptions.
6. Financial return data consistently violates normality through heavy tails and excess kurtosis, making normality testing particularly important in risk management and portfolio analysis.
