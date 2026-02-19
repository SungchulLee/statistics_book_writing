# Chapter 2: Descriptive Statistics

## Overview

This chapter develops the tools for exploring, summarizing, and visualizing data before any formal modeling or inference. It covers graphical methods for examining distributions (histograms, ECDFs, Q-Q plots), numerical measures of shape, center, and spread, and a range of visualization techniques for comparing distributions across groups. Together, these tools form the foundation of exploratory data analysis (EDA).

---

## Chapter Structure

### 2.1 Exploratory Data Analysis

Graphical tools for understanding the shape and structure of a dataset:

- **Histograms and Density Plots** -- Covers how histograms divide a continuous variable into bins to reveal distributional features (center, spread, skewness, modality, outliers) and how kernel density estimates provide a smooth approximation to the underlying PDF.
- **ECDF, Quantiles, and Q-Q Plots** -- Introduces the empirical cumulative distribution function as a bin-free alternative to histograms, defines quantiles and percentiles, and explains how Q-Q plots compare an observed distribution to a theoretical reference distribution.

### 2.2 Shape of Distributions

Characterizing the geometry of a distribution:

- **Modality** -- Explains unimodal, bimodal, and multimodal distributions, showing how the number of peaks reveals whether data comes from a single population or a mixture of distinct subgroups.
- **Skewness and Kurtosis** -- Defines skewness (asymmetry of the distribution) and kurtosis (tail heaviness relative to a normal distribution), with formulas, visualizations, and practical interpretation for identifying departures from normality.
- **Outliers and Leverage** -- Covers univariate and multivariate outliers, their causes (measurement error, natural variation, sampling), their effects on central tendency, variability, and regression models, and methods for detection (IQR rule, Z-scores).

### 2.3 Numerical Summaries

Quantitative measures of center and spread:

- **Mean, Median, Mode** -- Defines and compares the three main measures of central tendency, including their formulas, sensitivity to outliers, and appropriate use cases for symmetric, skewed, and categorical data.
- **Variance and Standard Deviation** -- Covers population and sample variance (with Bessel's correction), standard deviation, and their interpretation as measures of dispersion around the mean.
- **IQR and Robust Measures** -- Introduces the range, interquartile range, and percentiles as measures of spread that are resistant to outlier influence, complementing variance-based measures.
- **Median Absolute Deviation** -- Defines MAD as a robust alternative to standard deviation, covering its computation, the standardization constant for comparability with the standard deviation under normality, and its use for outlier-resistant dispersion measurement.

### 2.4 Visualization

Advanced plotting techniques for comparing distributions and groups:

- **Boxplots** -- Explains the anatomy of box-and-whisker plots (five-number summary, whiskers, outlier markers) and their use for compact visual summaries of center, spread, skewness, and outliers.
- **Violin Plots** -- Combines box plots with kernel density estimates to reveal the full distributional shape, including multimodality and density variations that box plots cannot show.
- **Group Comparisons** -- Covers scatter plots, line plots, bar plots, pie charts, pair plots, stem-and-leaf plots, dot plots, frequency tables, and mosaic plots for comparing distributions, frequencies, and relationships across groups.

### 2.5 Code

Complete Python implementations:

- **Group Comparison Examples** -- Python script demonstrating group comparison visualization techniques using real-world datasets.

### 2.6 Exercises

Practice problems covering histograms, ECDFs, measures of center and spread, skewness, kurtosis, outlier detection, and group comparison visualization.

---

## Prerequisites

This chapter builds on:

- **Chapter 0** (Prerequisites) -- Python basics, NumPy arrays, pandas DataFrames, and Matplotlib plotting for running the code examples and generating visualizations.
- **Chapter 1** (Data Collection) -- Understanding of populations, samples, and the distinction between designed and observational data.

---

## Key Takeaways

1. Exploratory data analysis with histograms, density plots, and ECDFs should precede any formal modeling to reveal distributional features and potential issues in the data.
2. The shape of a distribution (modality, skewness, kurtosis) determines which summary statistics and inferential methods are appropriate.
3. The mean is sensitive to outliers while the median is robust; choosing the right measure of center depends on the distribution shape.
4. Variance and standard deviation measure spread around the mean, while IQR and MAD provide robust alternatives that resist outlier influence.
5. Boxplots provide compact distributional summaries, while violin plots reveal the full density shape including multimodality.
6. Effective group comparisons require choosing the right visualization (scatter, bar, box, violin, mosaic) for the type of data and the comparison being made.
