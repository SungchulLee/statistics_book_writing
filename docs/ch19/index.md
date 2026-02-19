# Chapter 19: Logistic Regression

## Overview

Logistic regression models the probability of a binary outcome as a function of predictor variables. Unlike linear regression, which predicts a continuous response, logistic regression maps the linear predictor through the sigmoid function to produce probabilities in the interval (0, 1). This chapter covers the logistic model formulation and interpretation, maximum likelihood estimation and inference procedures, a comprehensive treatment of classification evaluation metrics, and regularization techniques adapted for the logistic setting.

---

## Chapter Structure

### 19.1 Logistic Regression

The model formulation, link function, and coefficient interpretation:

- **Logit Link and Odds** --- Introduces the sigmoid (logistic) function as the mapping from the linear predictor to probabilities, defines the logit as the log-odds, and derives the sigmoid derivative used throughout gradient-based estimation.
- **Odds Ratios and Interpretation** --- Shows that each coefficient represents the change in log-odds per unit increase in the corresponding feature, so that exponentiating the coefficient gives the multiplicative effect on the odds (the odds ratio).
- **Likelihood for Logistic Regression** --- Derives the Bernoulli likelihood for binary outcomes, takes the negative log to obtain the cross-entropy (log-loss) objective function, and establishes the optimization problem for logistic regression.

### 19.2 Estimation and Inference

How coefficients are estimated and tested:

- **Maximum Likelihood Estimation** --- Derives the gradient of the cross-entropy loss, showing it simplifies to a clean matrix expression involving the difference between predicted probabilities and observed labels, and explains why no closed-form solution exists.
- **Newton-Raphson and IRLS Algorithms** --- Describes the iterative optimization algorithms used to find the MLE, including Newton-Raphson (using the Hessian) and Iteratively Reweighted Least Squares (IRLS), which reframes logistic regression as a sequence of weighted least squares problems.
- **Wald and Likelihood Ratio Tests** --- Presents the two classical approaches for testing whether individual coefficients or groups of coefficients are significantly different from zero, using the asymptotic normality of the MLE and the Fisher information matrix.
- **Deviance and Goodness-of-Fit** --- Defines the deviance as twice the difference between the saturated and fitted model log-likelihoods, and discusses its use for assessing overall model fit and comparing nested models.

### 19.3 Model Evaluation

Comprehensive treatment of classification performance assessment:

- **Confusion Matrix** --- Defines the 2x2 contingency table (TP, FP, FN, TN) that summarizes all classification outcomes and from which all other metrics are derived.
- **Precision, Recall, and F1-Score** --- Covers precision (positive predictive value), recall (sensitivity/true positive rate), their trade-off, and the F1-score as their harmonic mean, with guidance on when each metric matters most.
- **ROC Curve and AUC** --- Explains the Receiver Operating Characteristic curve as a threshold-free evaluation of classifier performance, plots TPR vs. FPR across all thresholds, and interprets the Area Under the Curve as a measure of discriminative ability.
- **Decision Threshold Tuning** --- Discusses why the default 0.5 threshold is often suboptimal, how to choose thresholds based on cost considerations, and methods for finding the optimal operating point on the ROC curve.
- **Calibration and Brier Score** --- Addresses whether predicted probabilities are well-calibrated (i.e., a predicted 70% corresponds to a 70% empirical rate), and introduces the Brier score as a measure of probabilistic prediction quality.
- **Handling Imbalanced Data** --- Covers strategies for class imbalance including inverse class frequency weighting, oversampling, undersampling, and threshold adjustment, with practical guidance on when each approach is appropriate.
- **Evaluation Metrics Overview** --- A unified reference covering the confusion matrix, accuracy, precision, recall, F1-score, ROC/AUC, and their interrelationships, with emphasis on the limitations of accuracy for imbalanced classes.

### 19.4 Regularized Logistic Regression

Applying regularization to prevent overfitting in logistic models:

- **L1 and L2 Regularization for Logistic Regression** --- Extends Ridge (L2) and Lasso (L1) penalties to the logistic regression objective, producing regularized log-likelihood optimization problems that reduce overfitting and handle multicollinearity.
- **Feature Selection via Penalized Likelihood** --- Demonstrates how L1-penalized logistic regression drives irrelevant feature coefficients to exactly zero, enabling automatic feature selection in high-dimensional classification problems.

### 19.5 Code

Complete Python implementations:

- **logistic_regression.py** --- End-to-end logistic regression fitting and interpretation using statsmodels and scikit-learn.
- **evaluation_metrics.py** --- ROC curve, AUC, confusion matrix, precision, recall, F1-score, and calibration plot computations.
- **regularized_logistic.py** --- L1 and L2 regularized logistic regression with cross-validated hyperparameter tuning.
- **logistic_vs_linear_visualization.py** --- Visual comparison of logistic and linear regression on binary outcomes, illustrating why linear regression is inappropriate for classification.

### 19.6 Exercises

Practice problems covering logit and odds ratio interpretation, predicted probability computation, MLE derivation, confusion matrix analysis, ROC curve construction, threshold optimization for imbalanced data, and regularized logistic regression applications.

---

## Prerequisites

This chapter builds on:

- **Chapter 13** (Linear Regression) --- Ordinary least squares, coefficient interpretation, and the normal equations, which logistic regression extends to the classification setting.
- **Chapter 6** (Statistical Estimation) --- Maximum likelihood estimation, Fisher information, and asymptotic properties of the MLE.
- **Chapter 3** (Foundations of Probability) --- Bernoulli distribution, conditional probability, and likelihood functions.
- **Chapter 9** (Hypothesis Testing) --- Hypothesis testing framework, test statistics, and p-value interpretation used in Wald and likelihood ratio tests.
- **Chapter 18** (Regularization Techniques) --- Ridge and Lasso penalty concepts, which are extended to the logistic regression objective in the regularization section.

---

## Key Takeaways

1. Logistic regression models the log-odds of a binary outcome as a linear function of predictors, and the sigmoid function maps this linear predictor to a probability between 0 and 1.
2. Coefficients are interpreted through odds ratios: exponentiating a coefficient gives the multiplicative change in odds for a one-unit increase in the corresponding predictor.
3. MLE for logistic regression has no closed-form solution and requires iterative algorithms (Newton-Raphson or IRLS), but the gradient has a clean form involving the residuals (predicted minus observed).
4. Model evaluation requires metrics beyond accuracy --- precision, recall, F1-score, and AUC are essential for understanding classifier performance, especially with imbalanced classes.
5. The decision threshold should be tuned based on the relative costs of false positives and false negatives rather than defaulting to 0.5.
6. L1 and L2 regularization extend naturally to logistic regression, with L1 enabling feature selection and L2 stabilizing estimates when predictors are correlated or the model is overparameterized.
