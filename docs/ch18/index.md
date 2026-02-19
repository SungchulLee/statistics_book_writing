# Chapter 18: Regularization Techniques

## Overview

Regularization techniques address the fundamental problems of overfitting, multicollinearity, and instability in linear regression by adding penalty terms to the loss function. By deliberately introducing a small amount of bias, regularized estimators can substantially reduce variance and improve out-of-sample prediction. This chapter covers Ridge regression (L2), Lasso regression (L1), Elastic Net (combined L1+L2), hyperparameter tuning strategies, and dimensionality reduction alternatives including Principal Components Regression and Partial Least Squares.

---

## Chapter Structure

### 18.1 Motivation for Regularization

The problems that motivate moving beyond ordinary least squares:

- **Overfitting and the Bias-Variance Tradeoff** --- Explains how models with too many parameters capture noise rather than signal, and how regularization navigates the bias-variance tradeoff to minimize prediction error.
- **Ill-Conditioned Design Matrices** --- Describes how near-singular design matrices cause OLS coefficient estimates to become numerically unstable, with small data perturbations leading to large changes in estimated coefficients.
- **Multicollinearity and Regularization** --- Shows how highly correlated predictors inflate the variance of OLS estimates and how regularization stabilizes the solution by shrinking or selecting coefficients.

### 18.2 Ridge Regression

L2-penalized regression that shrinks all coefficients toward zero without eliminating any:

- **Ridge Formulation and Closed-Form Solution** --- Derives the Ridge estimator as the minimizer of the residual sum of squares plus an L2 penalty, with the closed-form solution involving the regularized inverse of the design matrix.
- **Geometric Interpretation (L2 Penalty)** --- Visualizes Ridge regression as the first contact point between elliptical OLS contours and a spherical constraint region centered at the origin.
- **Bayesian Interpretation (Gaussian Prior)** --- Shows that the Ridge estimator is equivalent to the posterior mode under a Gaussian prior on the coefficients, connecting frequentist regularization to Bayesian inference.
- **Ridge Trace and Choosing lambda** --- Introduces the Ridge trace plot (coefficients vs. lambda) and discusses strategies for selecting the regularization parameter.
- **Ridge Regression Overview** --- A comprehensive reference covering motivation, formulation, closed-form solution, the Gauss-Markov tradeoff, geometric and Bayesian interpretations, and practical guidance.

### 18.3 Lasso Regression

L1-penalized regression that performs simultaneous shrinkage and variable selection:

- **Lasso Formulation and Sparsity** --- Derives the Lasso estimator and explains why the L1 penalty drives some coefficients to exactly zero, producing sparse models.
- **Geometric Interpretation (L1 Penalty)** --- Visualizes the Lasso as the contact between elliptical OLS contours and a diamond-shaped (cross-polytope) constraint, with corners on the axes explaining the sparsity property.
- **Bayesian Interpretation (Laplace Prior)** --- Shows that the Lasso estimator corresponds to the posterior mode under a Laplace (double-exponential) prior, placing more mass near zero than the Gaussian prior.
- **Coordinate Descent Algorithm** --- Describes the iterative optimization algorithm used to fit Lasso, which cycles through coordinates applying soft-thresholding updates.
- **Lasso for Feature Selection** --- Discusses how the sparsity property of Lasso enables automatic feature selection and interpretable models in high-dimensional settings.
- **Lasso Regression Overview** --- A detailed reference covering the Lasso formulation, the L1 sparsity mechanism, the geometric argument for exact zeros, and comparisons with Ridge.

### 18.4 Elastic Net

A hybrid regularization method combining L1 and L2 penalties:

- **Elastic Net Formulation** --- Defines the Elastic Net objective function as a convex combination of L1 and L2 penalties controlled by two hyperparameters (or equivalently, a mixing ratio and overall penalty strength).
- **Advantages over Pure Ridge and Lasso** --- Explains how Elastic Net overcomes Lasso's limitations with correlated predictors and Ridge's inability to perform variable selection.
- **Grouping Effect for Correlated Features** --- Demonstrates how Elastic Net tends to select or exclude groups of correlated features together, rather than arbitrarily picking one and dropping the rest as Lasso does.
- **Elastic Net Overview** --- A comprehensive reference covering the objective function, key features, the grouping effect, handling of correlated predictors, and practical comparison with Ridge and Lasso.

### 18.5 Hyperparameter Tuning

Strategies for selecting the regularization parameter(s):

- **Cross-Validation for lambda Selection** --- Describes k-fold cross-validation as the standard method for choosing the regularization strength, including the one-standard-error rule for parsimonious models.
- **Regularization Path** --- Traces how model coefficients evolve as lambda varies, illustrating feature activation order, the bias-variance tradeoff, and the connection to model complexity.
- **Information Criteria for Regularized Models** --- Discusses the use of AIC, BIC, and related criteria as computationally cheaper alternatives to cross-validation for lambda selection.

### 18.6 Dimensionality Reduction Methods

Alternative approaches that reduce the predictor space rather than shrinking coefficients:

- **PCR and PLS Overview** --- Motivates dimensionality reduction for settings where the number of predictors exceeds the number of observations or where severe multicollinearity exists.
- **Principal Components Regression** --- Combines PCA with regression by extracting principal components that maximize variance in the predictor space, then regressing the response on a subset of these components.
- **Partial Least Squares** --- A supervised dimensionality reduction method that constructs components by maximizing the covariance between predictors and the response, often requiring fewer components than PCR.

### 18.7 Overview and Comparison

Guidance for choosing the right regularization approach:

- **Ridge vs. Lasso vs. Elastic Net** --- A side-by-side comparison of the three methods across penalty type, sparsity, multicollinearity handling, computational cost, and typical use cases.
- **Guidelines for Choosing a Method** --- Practical decision rules based on data characteristics (number of predictors, correlation structure, interpretability requirements).
- **Overview** --- A summary of all regularization methods and their applications, including Ridge, Lasso, Elastic Net, and dimensionality reduction approaches, with practical recommendations.

### 18.8 Code

Complete Python implementations:

- **ridge_examples.py** --- Ridge regression fitting, coefficient paths, and cross-validation.
- **lasso_examples.py** --- Lasso regression fitting, sparsity demonstration, and feature selection.
- **elastic_net_examples.py** --- Elastic Net fitting and comparison with Ridge and Lasso.
- **reg_compare.py** --- Side-by-side comparison of Ridge, Lasso, and Elastic Net on the same dataset.
- **cv_tuning.py** --- Cross-validation and lambda tuning with visualization of the CV error curve.
- **lasso_housing_regularization_path.py** --- Regularization path visualization for Lasso on housing data showing feature activation order.
- **pcr_pls_examples.py** --- Principal Components Regression and Partial Least Squares implementations with cross-validated component selection.

### 18.9 Exercises

Practice problems covering both conceptual topics (Ridge and Lasso closed-form solutions for orthonormal design, the grouping effect of Elastic Net, effective degrees of freedom) and computational exercises (fitting regularized models, comparing coefficient estimates, cross-validation tuning, and regularization path analysis).

---

## Prerequisites

This chapter builds on:

- **Chapter 13** (Linear Regression) --- Ordinary least squares estimation, the normal equations, residual analysis, and model selection criteria (AIC, BIC, cross-validation).
- **Chapter 6** (Statistical Estimation) --- Bias-variance tradeoff, mean squared error decomposition, and maximum likelihood estimation.
- **Chapter 0** (Prerequisites) --- Linear algebra foundations including matrix inverses, eigenvalues, singular value decomposition, and positive definite matrices.
- **Chapter 12** (Correlation and Causation) --- Understanding of multicollinearity and its effects on regression estimates.

---

## Key Takeaways

1. Regularization introduces bias to reduce variance, deliberately trading a small increase in bias for a large reduction in variance to minimize overall prediction error.
2. Ridge regression (L2) shrinks all coefficients toward zero but never eliminates any, making it ideal for handling multicollinearity when all predictors are potentially relevant.
3. Lasso regression (L1) produces sparse models by driving irrelevant coefficients to exactly zero, enabling simultaneous regularization and feature selection.
4. Elastic Net combines L1 and L2 penalties to overcome Lasso's instability with correlated predictors while retaining the ability to perform variable selection through the grouping effect.
5. Cross-validation is the standard method for selecting regularization hyperparameters, with the regularization path providing visual insight into how model complexity changes with penalty strength.
6. Dimensionality reduction methods (PCR, PLS) offer alternatives when the number of predictors is very large, constructing a small number of latent components that capture the essential structure in the predictors.
