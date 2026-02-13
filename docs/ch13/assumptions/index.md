# Assumptions and Diagnostics for Linear Regression

Linear regression is a fundamental statistical method used to model the relationship between a dependent variable and one or more independent variables. However, for the results of a linear regression model to be valid, certain assumptions must be met. These assumptions ensure that the model is appropriately specified and that the statistical inferences made from the model are reliable.

## The Four Key Assumptions

The four key assumptions underlying linear regression are commonly summarized by the acronym **LINE**:

1. **Linearity** — A straight-line relationship exists between the dependent variable and each independent variable.
2. **Independence** — The residuals (errors) of the regression model are independent of each other.
3. **Normality** — The residuals are normally distributed with a mean of zero.
4. **Equal Variance (Homoscedasticity)** — The variance of the residuals is constant across all levels of the independent variables.

## Why These Assumptions Matter

Understanding and verifying these assumptions is essential because:

- **Violated linearity** leads to biased predictions and invalid inferences, as the model either overestimates or underestimates the true relationship.
- **Violated independence** produces biased coefficient estimates, incorrect standard errors, and invalid hypothesis tests — particularly critical in time-series data where autocorrelation may be present.
- **Violated homoscedasticity** (heteroscedasticity) yields inefficient coefficient estimates and biased standard errors, ultimately compromising hypothesis tests.
- **Violated normality** undermines the validity of confidence intervals, hypothesis tests (t-tests for coefficients, F-tests for overall significance), and prediction intervals.

## Diagnostic Workflow

A typical diagnostic workflow for checking regression assumptions involves:

1. Fit the regression model and compute residuals.
2. Use **visual diagnostics** (scatterplots, residual plots, Q-Q plots, histograms) for initial assessment.
3. Apply **formal statistical tests** (Durbin-Watson, Breusch-Pagan, Shapiro-Wilk, etc.) for rigorous evaluation.
4. If violations are detected, apply **remedies** such as variable transformations, weighted least squares, robust regression, or alternative model specifications.

The following sections provide detailed coverage of each assumption, including diagnostic methods, formal tests, and remedies for violations.

## Section Overview

| Section | Topic | Key Diagnostics |
|---------|-------|----------------|
| [Linearity](linearity_overview.md) | Straight-line relationship assumption | Scatterplots, residual plots |
| [Independence](independence_overview.md) | Uncorrelated residuals assumption | Durbin-Watson test, residual plots |
| [Homoscedasticity](homoscedasticity_overview.md) | Constant variance assumption | Residual vs. fitted plot, Breusch-Pagan test |
| [Normality](normality_overview.md) | Normal residuals assumption | Q-Q plot, Shapiro-Wilk test |
| [Checking Linearity](checking_linearity.md) | Methods for linearity assessment | Scatterplots, CPR plots, polynomial terms |
| [Checking Independence](checking_independence.md) | Methods for independence assessment | Durbin-Watson, Breusch-Godfrey, clustering |
| [Checking Homoscedasticity](checking_homoscedasticity.md) | Methods for homoscedasticity assessment | Breusch-Pagan, White test, scale-location plot |
| [Checking Normality](checking_normality.md) | Methods for normality assessment | Histogram, Q-Q plot, Shapiro-Wilk, Anderson-Darling, Jarque-Bera |
