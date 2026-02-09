# Chapter: Normality Tests

## Overview

Many statistical methods—$t$-tests, ANOVA, linear regression, confidence intervals—assume that data (or residuals) follow a **normal distribution**. Before applying these methods, we need tools to check whether the normality assumption is reasonable.

This chapter covers:

- **What normality means** and why it matters for statistical inference
- **Graphical methods** (histograms, Q-Q plots, boxplots) for visual assessment
- **Descriptive statistics** (skewness, kurtosis) and their formal tests (skewtest, kurtosistest, D'Agostino's $K^2$, Jarque-Bera)
- **Formal hypothesis tests** (Kolmogorov-Smirnov / Lilliefors, Anderson-Darling, Shapiro-Wilk)
- **Limitations and pitfalls** of normality tests (sample size sensitivity, power, practical significance)
- **Strategies for non-normal data** (transformations, bootstrapping, non-parametric methods)
- **Applications** in $t$-tests, regression, and ANOVA

## Prerequisites

- Probability distributions (Chapter 4), especially the normal distribution
- Hypothesis testing framework (Chapter 9): null/alternative hypotheses, $p$-values, significance levels
- Familiarity with Python (`numpy`, `scipy.stats`, `matplotlib`)

## Key Python Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors, normal_ad
```
