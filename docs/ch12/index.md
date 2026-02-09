# Chapter 12: Linear Regression

## Overview

This chapter provides a comprehensive treatment of linear regression, from simple models with a single predictor through multiple regression with interaction and polynomial terms. The chapter covers the full modeling workflow: estimation, inference, diagnostics, performance evaluation, model selection, and practical implementation in Python.

---

## Chapter Structure

### 12.1 Linear Regression

The foundational regression models:

- **Simple Linear Regression** — modeling the relationship between a single predictor and a response variable using the least squares line.
- **Multiple Linear Regression** — extending to multiple predictors, with discussion of coefficients, $R^2$, adjusted $R^2$, and information criteria (AIC, BIC).

### 12.2 Estimation and Inference

How regression coefficients are estimated and tested:

- **Least Squares Estimation** — deriving the OLS estimator and its properties.
- **Sampling Distributions** — the distributions of OLS estimators under classical assumptions.
- **Confidence Intervals for Coefficients** — interval estimation for individual regression parameters.
- **Hypothesis Tests** — $t$-tests for individual coefficients and $F$-tests for overall model significance.

### 12.3 Interaction and Polynomial Extensions

Capturing complex relationships beyond additivity and linearity:

- **Interaction Terms** — modeling how the effect of one predictor depends on the level of another.
- **Polynomial Regression** — modeling non-linear (curved) relationships using powers of predictors.

### 12.4 Diagnostics

Verifying model assumptions and identifying problems:

- **Residual Analysis** — checking normality, homoscedasticity, and independence of residuals using graphical and formal tests.
- **Multicollinearity and Influence** — detecting collinear predictors (VIF, tolerance) and influential observations (Cook's distance, leverage, standardized residuals).

### 12.5 Performance Metrics

Quantifying model quality:

- **$R^2$, Adjusted $R^2$, MAE, MSE, RMSE** — a suite of metrics for assessing fit and prediction accuracy.

### 12.6 Model Selection Criteria

Choosing among competing models:

- **AIC and BIC** — information criteria that balance fit against complexity, with guidance on when to prefer each.

### 12.7 Package Usage

Practical tools for implementation:

- **sklearn vs statsmodels Comparison** — when to use each library, with side-by-side examples and a combined workflow.

### 12.8 Code

Complete Python implementations:

- **CI for Slope** — confidence interval computation for the slope in a caffeine study example.
- **CI and Prediction Bands** — confidence and prediction intervals for regression lines.
- **OLS Regression Output Reproduction** — reproducing the full OLS summary table from scratch.
- **Multiple Regression Diagnostics** — complete implementation of multiple regression with VIF, residual analysis, influence diagnostics, interaction terms, and polynomial regression.

### 12.9 Exercises

Practice problems covering regression modeling, diagnostics, and interpretation.

---

## Prerequisites

This chapter builds on:

- **Chapter 5** (Sampling Distributions) — understanding the $t$, $\chi^2$, and $F$ distributions.
- **Chapter 8** (Confidence Intervals) — constructing interval estimates.
- **Chapter 9** (Hypothesis Testing) — formulating and conducting statistical tests.

---

## Key Takeaways

1. Multiple regression models how several predictors jointly influence a response, with each coefficient representing the effect of one predictor holding others constant.
2. Interaction and polynomial terms extend the model to capture non-additive and non-linear relationships.
3. Model diagnostics (residual analysis, multicollinearity checks, influence diagnostics) are essential to validate assumptions.
4. AIC and BIC provide principled criteria for model selection that balance fit against complexity.
5. `statsmodels` and `sklearn` serve complementary roles: inference and diagnostics versus prediction and deployment.
