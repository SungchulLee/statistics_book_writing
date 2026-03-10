# MoM vs MLE Comparison


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

## Overview

We compare the Method of Moments and Maximum Likelihood Estimation across several criteria.

## Comparison

| Criterion | MoM | MLE |
|---|---|---|
| Computation | Often closed-form | May require optimization |
| Efficiency | Generally less efficient | Asymptotically efficient |
| Consistency | Yes (under regularity) | Yes (under regularity) |
| Invariance | No | Yes |
| Robustness | Moderate | Sensitive to model misspecification |

## When to Use Each

- **MoM**: Quick estimates, starting values for MLE optimization, when likelihood is intractable
- **MLE**: When efficiency matters, when model is correctly specified, when asymptotic theory is needed
