# Resampling Methods: Overview

## What Are Resampling Methods?

Resampling methods are modern, computationally intensive approaches to statistical inference that construct the sampling distribution of a statistic **empirically** by repeatedly drawing samples from the observed data, rather than relying on theoretical distributional assumptions.

Unlike rank-based non-parametric methods (Chapter 15), which replace values with ranks, resampling methods work directly with the original data values. They leverage computational power to approximate distributions that would be difficult or impossible to derive analytically.

## The Two Primary Resampling Methods

| Method | Resampling Type | Primary Use | Key Output |
|---|---|---|---|
| **Bootstrap** | With replacement | Estimation | Confidence intervals, standard errors |
| **Permutation Test** | Without replacement (shuffling labels) | Hypothesis testing | p-values |

## When to Use Resampling Methods

Resampling methods are especially valuable when:

- The theoretical sampling distribution is unknown or difficult to derive.
- The statistic of interest is complex (e.g., median, ratio of variances, regression coefficients).
- Sample sizes are small and asymptotic approximations are unreliable.
- No parametric or rank-based test exists for the specific problem.

## Historical Context

- The **bootstrap** was introduced by Bradley Efron in 1979 and revolutionized statistical practice by making distribution-free inference accessible for virtually any statistic.
- **Permutation tests** date back to R.A. Fisher in the 1930s but became practical only with modern computing power.

## Chapter Structure

- **16.1 Bootstrap**: Confidence intervals, standard errors, hypothesis testing via bootstrap
- **16.2 Permutation Tests**: Hypothesis testing via label shuffling
- **16.3 Comparison**: When to use bootstrap vs permutation tests
