# Chapter 13: Linear Regression

## Overview

This chapter provides a comprehensive treatment of linear regression, from simple models with a single predictor through multiple regression with interaction, polynomial, and spline-based extensions. The chapter covers the full modeling workflow: estimation via ordinary least squares, inference on coefficients, assumption checking and diagnostics, performance evaluation, model selection, and practical implementation in Python using both statsmodels and scikit-learn.

---

## Chapter Structure

### 13.1 Linear Regression

The foundational regression models:

- **Simple Linear Regression** -- Models the relationship between a single predictor $X$ and a response $Y$ as a straight line $Y = \beta_0 + \beta_1 X + \varepsilon$, with geometric intuition from scatter plots and fitted lines.
- **Multiple Linear Regression** -- Extends to multiple predictors using matrix notation $\mathbf{y} = X\boldsymbol{\beta} + \boldsymbol{\varepsilon}$, including interaction terms and implementation with scikit-learn.

### 13.2 Estimation and Inference

How regression coefficients are estimated and their statistical properties:

- **Least Squares Estimation** -- Derives the OLS estimator from three perspectives: the least squares criterion, maximum likelihood under Gaussian errors, and the normal equation with vector calculus.
- **Sampling Distributions (Simple OLS)** -- Derives the $t$-distributions of the slope estimator, mean response, and predicted response under classical assumptions.
- **Confidence Intervals for Coefficients** -- Constructs interval estimates for the slope, expected response, and individual predictions, including the characteristic "bowtie" shape of confidence bands.
- **Sampling Distributions and Tests (General OLS)** -- Extends inferential results to the multiple regression setting using matrix notation, deriving the distribution of $\hat{\beta}$, $s^2$, and the $t$-statistic for each coefficient.

### 13.3 Testing Coefficients

Evaluating the statistical significance of individual predictors:

- **Hypothesis Tests (t-tests)** -- Formulates and conducts $t$-tests for individual regression coefficients to assess whether each predictor has a significant effect on the response.
- **p-values and Confidence Intervals** -- Interprets $p$-values and confidence intervals for coefficients, with practical examples showing how to read regression output tables.
- **Interpretation of Significance** -- Distinguishes between statistical significance and practical significance, with guidance on threshold conventions and common misinterpretations.

### 13.4 Interaction and Polynomial Extensions

Capturing complex relationships beyond additivity and linearity:

- **Interaction Terms** -- Models how the effect of one predictor depends on the level of another by including product terms $X_1 \times X_2$ in the regression equation.
- **Polynomial Regression** -- Models non-linear (curved) relationships using powers of predictors $X, X^2, \ldots, X^d$, while remaining a linear model in the parameters.

### 13.5 Assumptions and Diagnostics

The four key assumptions (LINE) and how to verify them:

- **Assumptions Overview (LINE)** -- Summarizes the four requirements: Linearity, Independence, Normality of residuals, and Equal variance (homoscedasticity), with a diagnostic workflow.
- **Linearity Assumption** -- What linearity means and why violations lead to biased predictions.
- **Independence Assumption** -- Why residual independence matters, especially in time-series contexts where autocorrelation may occur.
- **Homoscedasticity Assumption** -- Constant error variance and the consequences of heteroscedasticity for inference.
- **Normality Assumption** -- Why normally distributed residuals are needed for valid confidence intervals and hypothesis tests.
- **Checking Linearity / Independence / Homoscedasticity / Normality** -- Visual diagnostics (residual plots, Q-Q plots) and formal tests (Durbin-Watson, Breusch-Pagan, Shapiro-Wilk) for each assumption.

### 13.6 Diagnostics

Identifying problems with the fitted model:

- **Residual Analysis** -- Examines residuals (observed minus predicted) to check for patterns indicating model misspecification, non-constant variance, or non-normality.
- **Multicollinearity and VIF** -- Detects highly correlated predictors using the Variance Inflation Factor, with discussion of consequences for coefficient stability and interpretation.
- **Influence and Leverage (Cook's Distance, DFFITS)** -- Identifies observations that disproportionately affect regression results, using Cook's distance, leverage values, and DFFITS.

### 13.7 Performance Metrics

Quantifying how well the model fits and predicts:

- **$R^2$ and Adjusted $R^2$** -- Measures the proportion of variance explained by the model, with adjusted $R^2$ penalizing for the number of predictors.
- **MAE, MSE, and RMSE** -- Absolute and squared error metrics for assessing prediction accuracy on the original scale of the response.
- **MAPE and Other Relative Metrics** -- Scale-independent measures useful for comparing models across different datasets or response scales.

### 13.8 Model Selection Criteria

Choosing among competing models:

- **AIC (Akaike Information Criterion)** -- Balances goodness of fit against model complexity, favoring models that predict well out of sample.
- **BIC (Bayesian Information Criterion)** -- Similar to AIC but with a stronger penalty for additional parameters, tending to select simpler models.
- **Cross-Validation for Model Selection** -- Uses hold-out or $k$-fold strategies to estimate out-of-sample prediction error directly.
- **Stepwise and Best-Subset Selection** -- Automated procedures for searching the space of possible predictor subsets.

### 13.9 Splines and GAMs

Flexible non-parametric extensions to linear regression:

- **Splines and GAMs Overview** -- Introduces step functions, piecewise polynomial splines, and their advantages over global polynomials for capturing local patterns.
- **Generalized Additive Models** -- Extends regression by replacing linear terms with smooth non-parametric functions $f_j(X_j)$ learned from data, maintaining interpretability through the additive structure.

### 13.10 Package Usage

Practical tools for implementing regression in Python:

- **statsmodels OLS Interface** -- The go-to library for statistical inference, providing detailed summaries with $p$-values, confidence intervals, and diagnostic tests.
- **sklearn LinearRegression Interface** -- The go-to library for prediction workflows, with built-in cross-validation, pipelines, and regularization.
- **Comparison and When to Use Which** -- Side-by-side feature comparison showing that statsmodels excels at inference while sklearn excels at prediction and machine learning pipelines.

### 13.11 Code

Complete Python implementations:

- **CI for Slope (Caffeine Example)** -- Confidence interval computation for the regression slope.
- **CI and Prediction Bands** -- Confidence and prediction intervals plotted alongside the regression line.
- **OLS Regression Output Reproduction** -- Reproducing the full statsmodels OLS summary table from scratch.
- **Multiple Regression Diagnostics** -- VIF, residual analysis, influence diagnostics, and interaction terms.
- **Testing Coefficients Examples** -- Demonstrations of $t$-tests and $F$-tests for regression coefficients.
- **Model Selection Comparison** -- AIC, BIC, and cross-validation applied to competing models.
- **Weighted Least Squares** -- Regression with non-constant variance using observation weights.
- **GAM Housing Analysis** -- Generalized additive model fitted to housing data with smooth terms.
- **Regression Diagnostics (Housing)** -- Full diagnostic suite on the California Housing dataset.
- **RSS Surface Visualization** -- 3D visualization of the residual sum of squares as a function of $\beta_0$ and $\beta_1$.
- **3D Regression Plane** -- Visualization of a multiple regression surface in three dimensions.
- **CV Polynomial Model Selection** -- Cross-validation to select the optimal polynomial degree.
- **Step Functions** -- Piecewise constant regression implementations.
- **Splines with Patsy** -- B-spline and natural spline fitting using the patsy formula interface.

### 13.12 Exercises

Practice problems covering regression modeling, OLS output interpretation, diagnostics, coefficient testing, model selection, and polynomial and interaction extensions.

---

## Prerequisites

This chapter builds on:

- **Chapter 5** (Sampling Distributions) -- Understanding the $t$, $\chi^2$, and $F$ distributions and how they arise from normal samples.
- **Chapter 8** (Confidence Intervals) -- Constructing interval estimates for population parameters.
- **Chapter 9** (Hypothesis Testing) -- Formulating null and alternative hypotheses, computing $p$-values, and making decisions at given significance levels.
- **Chapter 12** (Correlation and Causation) -- Pearson correlation as the foundation for simple linear regression, and the distinction between association and causation.

---

## Key Takeaways

1. Simple linear regression models a straight-line relationship between one predictor and a response; multiple regression generalizes this to any number of predictors, with each coefficient representing the effect of one predictor holding all others constant.
2. OLS estimation minimizes the sum of squared residuals and, under Gaussian errors, coincides with maximum likelihood estimation; the resulting estimators have known $t$-distributions that enable inference.
3. Interaction and polynomial terms extend the linear model to capture non-additive and non-linear relationships without abandoning the OLS framework.
4. The LINE assumptions (Linearity, Independence, Normality, Equal variance) must be verified through residual diagnostics; violations invalidate inference and may require transformations, robust methods, or alternative models.
5. Model selection criteria (AIC, BIC, cross-validation) provide principled approaches to choosing among competing models, balancing fit against complexity.
6. Splines and GAMs offer flexible alternatives when the true relationship is non-linear, providing smooth, locally adaptive fits while maintaining interpretability.
7. statsmodels and scikit-learn serve complementary roles: use statsmodels for inference and diagnostics, and scikit-learn for prediction pipelines and deployment.
