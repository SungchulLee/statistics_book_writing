# Splines and Generalized Additive Models

## Overview

Beyond polynomial regression, splines and generalized additive models (GAMs) provide flexible methods for capturing non-linear relationships while maintaining interpretability and computational efficiency.

## Contents

This section covers:

- **Splines** — Piecewise polynomial functions that provide smooth, local adaptability
- **Generalized Additive Models (GAMs)** — Flexible semi-parametric regression using smooth functions of predictors

## Why These Methods Matter

Traditional polynomial regression requires choosing the degree of the polynomial globally for all data. Splines and GAMs overcome this limitation:

1. **Local flexibility** — Splines adapt their shape locally to the data rather than imposing a global polynomial form
2. **Reduced overfitting** — Regularization (smoothness penalties) prevents spurious wiggles
3. **Automatic smoothness** — Many algorithms optimize smoothing parameters automatically
4. **Interpretability** — Each predictor's effect is visualizable and understandable independently

## Key Concepts

- **Basis functions** — Splines are linear combinations of basis functions (B-splines, thin-plate splines)
- **Smoothness penalty** — Regularization term that penalizes roughness, controlling the bias-variance tradeoff
- **Effective degrees of freedom** — Accounts for the penalty when assessing model complexity
- **Additivity** — In GAMs, the response is an additive combination of univariate smooth functions

## Practical Applications

- **Economics**: Modeling non-linear price elasticity across product ranges
- **Finance**: Capturing volatility smile in options pricing
- **Medicine**: Dose-response relationships that plateau at high doses
- **Environmental science**: Non-linear effects of pollutant concentrations on health outcomes
