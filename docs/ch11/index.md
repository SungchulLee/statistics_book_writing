# Chapter 11: ANOVA

## Overview

Analysis of Variance (ANOVA) extends hypothesis testing to compare means across three or more groups simultaneously, avoiding the inflated Type I error rate that would result from performing multiple pairwise t-tests. This chapter covers one-way and two-way ANOVA models, a comprehensive suite of post-hoc comparison methods, Welch's ANOVA for heteroscedastic data, assumption checking and diagnostic tools, and practical applications including A/B testing and financial analysis.

---

## Chapter Structure

### 11.1 One-Way ANOVA

The foundational ANOVA framework for comparing means across multiple independent groups:

- **Model and Assumptions** -- Introduces one-way ANOVA as a method for partitioning total variability into between-group and within-group components, defines the F-statistic as the ratio of these variances, and states the assumptions of normality, independence, and homoscedasticity.
- **F-Test Procedure** -- Provides a step-by-step procedure for conducting one-way ANOVA: formulating hypotheses, computing group means, calculating the total, between-group, and within-group sums of squares, constructing the ANOVA table, and interpreting the F-statistic and p-value.

### 11.2 Two-Way ANOVA

Extends the ANOVA framework to examine the effects of two factors simultaneously:

- **Main Effects and Blocking** -- Introduces two-way ANOVA for assessing the individual effects of two factors on a continuous outcome, partitioning total variance into components due to Factor A, Factor B, and residual error, and explains the use of blocking to control nuisance variability.
- **Interaction Effects** -- Defines interaction effects as the combined influence of two factors beyond their individual main effects, provides formulas for the interaction sum of squares and F-test, and demonstrates interpretation using interaction plots.

### 11.3 Post-Hoc Comparisons

Methods for identifying which specific group pairs differ after a significant ANOVA result:

- **Tukey HSD** -- The most widely used post-hoc test for all pairwise comparisons, controlling the family-wise error rate using the Studentized range distribution.
- **Bonferroni and Scheffe Methods** -- Bonferroni adjusts the significance level by the number of comparisons (suitable for a small number of planned comparisons), while Scheffe controls the FWER for all possible linear contrasts (more conservative but applicable to any set of comparisons).
- **Dunnett's Test (vs Control)** -- A specialized test for comparing each of k minus 1 treatment groups against a single control group, using the multivariate t-distribution to account for the correlation between comparisons.
- **Games-Howell (Unequal Variances)** -- A post-hoc procedure that does not assume equal variances or equal sample sizes, using Welch-Satterthwaite degrees of freedom for each pairwise comparison.

### 11.4 Welch's ANOVA

Robust alternatives to classical ANOVA when the equal-variance assumption is violated:

- **Welch's One-Way ANOVA** -- Tests equality of group means without assuming homoscedasticity, using weighted means and an adjusted F-statistic with modified degrees of freedom.
- **Welch's Two-Way ANOVA** -- Extends the Welch approach to two-factor designs, allowing analysis of main effects and interactions under heteroscedasticity using robust (HC3) standard errors.

### 11.5 Assumptions

A systematic treatment of the four assumptions underlying ANOVA and how to verify each one:

- **Assumptions Overview** -- Summarizes the four key assumptions (normality, independence, homoscedasticity, linearity) and explains why each matters for the validity of the F-test.
- **Checking Normality of Residuals** -- Describes Q-Q plots and the Shapiro-Wilk test for assessing whether residuals are approximately normally distributed, with guidance on robustness for large samples.
- **Checking Independence of Observations** -- Emphasizes that independence is primarily ensured through proper study design (random sampling and assignment) and describes the Durbin-Watson test for detecting serial correlation.
- **Checking Homoscedasticity** -- Uses Levene's test and residual-vs-fitted plots to detect unequal variances across groups, with guidance on when to switch to Welch's ANOVA.
- **Checking Linearity** -- Discusses the relevance of the linearity assumption when ANOVA includes continuous covariates, using scatter plots and residual plots to detect nonlinear patterns.

### 11.6 Diagnostics

Tools for evaluating model adequacy and identifying problematic observations:

- **Residual Analysis** -- Examines residual-vs-fitted plots, standardized residuals, and histograms of residuals to detect patterns indicating assumption violations such as heteroscedasticity, non-normality, and model misspecification.
- **Influential Data Points** -- Identifies observations with disproportionate influence on ANOVA results using Cook's distance, leverage values, and DFFITS, with threshold guidelines and visualization.
- **Handling Assumption Violations** -- Provides a systematic decision framework for addressing violations, including data transformations (log, square root, Box-Cox), non-parametric alternatives (Kruskal-Wallis), and robust methods (Welch's ANOVA).

### 11.7 Practical Applications

Real-world applications demonstrating the complete ANOVA workflow:

- **A/B Testing and Experimental Design** -- Connects ANOVA to A/B testing with more than two groups, covering randomization, control groups, and the relationship between A/B tests and the one-way F-test.
- **Financial Applications of ANOVA** -- Applies ANOVA to comparing portfolio returns, sector analysis, factor model testing, and trading strategy evaluation across market regimes.
- **Case Studies** -- Complete worked examples (e.g., Iris species morphology) demonstrating the full ANOVA pipeline from model fitting through assumption checking and diagnostics using Python.

### 11.8 Code

Complete Python implementations:

- **anova_diagnostics.py** -- Comprehensive ANOVA diagnostic toolkit including residual plots, normality tests, and homoscedasticity checks.
- **post_hoc_comparisons.py** -- Implementations of Tukey HSD, Bonferroni, Scheffe, Dunnett, and Games-Howell post-hoc tests.
- **oneway_pipeline.py** -- End-to-end one-way ANOVA pipeline from data loading through assumption checking and post-hoc analysis.
- **oneway_scipy.py** -- One-way ANOVA using scipy with accompanying visualizations.
- **twoway_pipeline.py** -- End-to-end two-way ANOVA pipeline with main effects and interaction analysis.
- **interaction_plot.py** -- Visualization of interaction effects between two factors.
- **welch_simulation.py** -- Monte Carlo simulation comparing Type I error rates and power of classical versus Welch ANOVA.
- **welch_twoway_robust.py** -- Two-way Welch ANOVA implementation using robust HC3 standard errors.

### 11.9 Exercises

Practice problems covering one-way ANOVA computation, assumption checking, post-hoc test selection, two-way ANOVA with interaction interpretation, and complete analysis workflows on real datasets.

---

## Prerequisites

This chapter builds on:

- **Chapter 5** (Sampling Distributions) -- The F-distribution, which is the reference distribution for the ANOVA F-test.
- **Chapter 9** (Hypothesis Testing) -- The general hypothesis testing framework, including significance levels, p-values, Type I and Type II errors, and power.
- **Chapter 8** (Confidence Intervals) -- Confidence intervals for means, which are used in post-hoc comparisons to construct simultaneous intervals for pairwise differences.

---

## Key Takeaways

1. ANOVA compares group means by partitioning total variability into between-group and within-group components; a large F-statistic indicates that group means differ more than would be expected by chance alone.
2. A significant ANOVA result indicates that at least one group mean differs but does not identify which groups differ; post-hoc tests (Tukey, Bonferroni, Scheffe, Dunnett, Games-Howell) are required to pinpoint the specific pairwise differences.
3. Two-way ANOVA simultaneously assesses the effects of two factors and their interaction, where an interaction effect means the influence of one factor depends on the level of the other.
4. The four ANOVA assumptions (normality, independence, homoscedasticity, linearity) should be checked systematically; when homoscedasticity is violated, Welch's ANOVA provides a robust alternative.
5. Diagnostic tools such as residual plots, Cook's distance, and formal tests (Shapiro-Wilk, Levene's) are essential for validating the ANOVA model before interpreting results.
6. ANOVA has broad practical applications including A/B testing with multiple treatments, comparing financial portfolio returns, and any experimental design comparing three or more groups.
