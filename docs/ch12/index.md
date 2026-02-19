# Chapter 12: Correlation and Causation

## Overview

This chapter explores the fundamental concepts of correlation and causation -- two ideas that are central to statistical reasoning and data interpretation. While correlation quantifies the strength and direction of the relationship between two variables, causation implies that changes in one variable directly produce changes in another. The chapter covers how to measure, test, and visualize correlations, how to recognize pitfalls such as ecological fallacy and confounding, and how to reason about causal relationships using modern frameworks like directed acyclic graphs.

---

## Chapter Structure

### 12.1 Correlation

Measures of association between two variables, covering both classical and rank-based approaches:

- **Pearson Correlation Coefficient** -- Quantifies the strength and direction of the linear relationship between two continuous variables, ranging from $-1$ to $+1$.
- **Spearman Rank Correlation** -- A non-parametric measure based on the ranks of observations, capturing monotonic (not necessarily linear) relationships.
- **Kendall's Tau** -- A rank-based correlation coefficient that counts concordant and discordant pairs, offering robustness to outliers and ties.
- **Partial Correlation** -- Measures the association between two variables after controlling for the effect of one or more additional variables.
- **Point-Biserial and Phi Coefficients** -- Specialized correlation measures for situations involving binary variables (one continuous and one binary, or both binary).
- **Understanding Correlation** -- A conceptual overview of what correlation is, how to visualize it across a range of values, and common misconceptions.

### 12.2 Ecological Correlation

How aggregated data can mislead about individual-level relationships:

- **Ecological Fallacy** -- The error of inferring individual-level associations from group-level (aggregated) correlation data.
- **Simpson's Paradox** -- A phenomenon where a trend that appears in aggregated data reverses or disappears when the data is separated into subgroups.
- **Aggregation Bias** -- Systematic distortion introduced when individual-level data is summarized at a higher level, altering the apparent strength or direction of associations.

### 12.3 Correlation, Causation, and Confounding

Why correlation alone does not imply causation, and the role of hidden variables:

- **Confounding Variables** -- Third variables that influence both the predictor and the outcome, creating a spurious association between them.
- **Spurious Correlations** -- Statistically significant correlations that arise from chance, confounding, or data mining rather than a genuine causal relationship.
- **Lurking Variables and Common Causes** -- Unobserved variables that drive observed associations, making it appear that two variables are directly related when they are not.

### 12.4 Causation

Frameworks and methods for establishing genuine causal relationships:

- **Criteria for Causal Inference** -- Classical criteria (temporal precedence, covariation, elimination of confounders) and modern perspectives on what constitutes evidence for causation.
- **Randomized Experiments and Causation** -- How random assignment eliminates confounding and provides the strongest evidence for causal claims.
- **Instrumental Variables (Introduction)** -- A technique for estimating causal effects in observational data by leveraging a variable that affects the treatment but not the outcome directly.
- **Directed Acyclic Graphs (DAGs)** -- A graphical framework for encoding causal assumptions, identifying confounders, and determining which variables to control for in an analysis.

### 12.5 Correlation Tests

Formal hypothesis tests for the significance of different correlation measures:

- **Testing Pearson's $r$ (t-Test for Correlation)** -- A $t$-based test to determine whether the population Pearson correlation is significantly different from zero.
- **Testing Spearman's $\rho$** -- Hypothesis testing for the significance of Spearman rank correlation using either exact or approximate methods.
- **Testing Kendall's $\tau$** -- Hypothesis testing for the significance of Kendall's tau with normal approximation for larger samples.
- **Comparing Two Correlations** -- Methods (such as Fisher's $z$-transformation) for testing whether two correlation coefficients differ significantly from each other.

### 12.6 Correlation Matrix and Visualization

Tools for exploring multivariate relationships visually:

- **Correlation Heatmaps** -- Two-dimensional color-coded visualizations of correlation matrices that reveal patterns of association across many variables simultaneously.
- **Pair Plots and Scatter Matrices** -- Grid displays of bivariate scatter plots for all variable pairs, useful for identifying non-linear patterns and outliers.

### 12.7 Code

Complete Python implementations:

- **Correlation Analysis Demonstrations** -- End-to-end computation and comparison of Pearson, Spearman, and Kendall correlations.
- **Causal Inference Simulations** -- Simulations illustrating confounding, spurious correlation, and the effect of randomization.
- **Correlation Visualization** -- Scripts for generating heatmaps, scatter matrices, and annotated correlation plots.
- **Correlation Ellipse Plot** -- Visualization of correlation strength using ellipses whose shape encodes the magnitude of association.

### 12.8 Exercises

Practice problems covering correlation computation and comparison, visualization of associations, identification of confounders and ecological fallacies, and application of correlation tests to real datasets.

---

## Prerequisites

This chapter builds on:

- **Chapter 3** (Foundations of Probability) -- Random variables, expectation, and covariance.
- **Chapter 4** (Distributions) -- Joint distributions, covariance, and the distinction between independence and zero correlation.
- **Chapter 2** (Descriptive Statistics) -- Scatter plots, numerical summaries, and exploratory data analysis.
- **Chapter 9** (Hypothesis Testing) -- Null and alternative hypotheses, $p$-values, and significance levels.

---

## Key Takeaways

1. Correlation measures the strength and direction of association between variables, but it does not imply causation -- confounders, lurking variables, and aggregation effects can all produce misleading correlations.
2. Pearson captures linear association, while Spearman and Kendall capture monotonic association and are more robust to outliers and non-normal data.
3. The ecological fallacy and Simpson's paradox demonstrate that group-level patterns can differ dramatically from individual-level patterns, making careful data disaggregation essential.
4. Establishing causation requires more than correlation: randomized experiments, instrumental variables, and directed acyclic graphs provide rigorous frameworks for causal inference.
5. Formal correlation tests (with appropriate null hypotheses and $p$-values) are necessary to distinguish genuine associations from sampling variability, and Fisher's $z$-transformation allows comparison of correlations across samples.
