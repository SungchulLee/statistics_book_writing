# Chapter 13: Logistic Regression

## Overview

Logistic regression models the probability of a binary outcome as a function of predictor variables. Unlike linear regression, which predicts a continuous response, logistic regression predicts the log-odds of belonging to a particular class. It is the foundational model for binary classification and serves as a stepping stone to more complex classification methods.

## Learning Objectives

After completing this chapter, you should be able to:

- Explain why linear regression is inappropriate for binary outcomes
- Define the logit link function and interpret odds and odds ratios
- Write the likelihood function for logistic regression
- Describe how MLE is used to estimate logistic regression coefficients
- Perform Wald and likelihood ratio tests for coefficient significance
- Evaluate model performance using confusion matrix, ROC curve, and AUC
- Implement logistic regression using both sklearn and statsmodels

## Chapter Outline

| Section | Topic |
|---|---|
| 13.1 | Logistic Regression: Logit Link, Odds, Odds Ratios |
| 13.2 | Estimation and Inference: MLE, Wald and LR Tests |
| 13.3 | Model Evaluation: Confusion Matrix, ROC, AUC, Classification Metrics |
| 13.4 | Code |
| 13.5 | Exercises |

## Prerequisites

- Linear regression (Chapter 12)
- Maximum likelihood estimation (Chapter 5–6)
- Probability fundamentals (Chapter 3)
