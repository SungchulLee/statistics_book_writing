# Chapter 21: Survival Models

## Overview

This chapter introduces survival analysis, the branch of statistics concerned with modeling time-to-event data in the presence of censoring. We develop the core mathematical framework---survival and hazard functions---and progress from non-parametric estimators (Kaplan--Meier, Nelson--Aalen) through fully parametric models (exponential, Weibull, log-normal, log-logistic) to the semi-parametric Cox proportional hazards model. The chapter concludes with model comparison strategies and practical applications in finance, medicine, and engineering.

---

## Chapter Structure

### 21.1 Introduction to Survival Analysis

Foundational concepts for analyzing time-to-event outcomes:

- **Time-to-Event Data and Censoring** --- Defines the structure of survival data where the outcome is a duration until an event of interest, and introduces the fundamental challenge of incomplete observation (censoring).
- **Types of Censoring (Right, Left, Interval)** --- Classifies censoring mechanisms: right-censoring (event not yet observed), left-censoring (event occurred before observation began), and interval-censoring (event known only to fall within a time window).
- **Survival and Hazard Functions** --- Develops the key mathematical quantities: the survival function $S(t) = P(T > t)$, the hazard function $h(t)$, and the cumulative hazard $H(t)$, along with their interrelationships.
- **Financial Applications (Default, Churn, Duration)** --- Motivates survival analysis with real-world examples including credit default modeling, customer churn prediction, and duration analysis in economics and finance.

### 21.2 Non-Parametric Methods

Distribution-free approaches to estimating survival:

- **Kaplan--Meier Estimator** --- Derives the product-limit estimator for the survival function, which handles censored observations by adjusting the risk set at each observed event time.
- **Nelson--Aalen Cumulative Hazard** --- Presents an alternative non-parametric estimator that directly estimates the cumulative hazard function and relates it to the Kaplan--Meier curve.
- **Log-Rank Test** --- Introduces the most widely used hypothesis test for comparing survival distributions between two or more groups, based on observed versus expected event counts.
- **Confidence Intervals for Survival Curves** --- Constructs pointwise confidence bands for Kaplan--Meier estimates using Greenwood's formula and log-transformed intervals.

### 21.3 Parametric Survival Models

Fully specified distributional models for survival times:

- **Exponential Model (Constant Hazard)** --- The simplest parametric survival model, assuming a constant hazard rate over time, with connections to the memoryless property and Poisson processes.
- **Weibull Model (Monotone Hazard)** --- Generalizes the exponential model by allowing the hazard to increase or decrease monotonically over time via a shape parameter.
- **Log-Normal and Log-Logistic Models** --- Covers two accelerated failure time models whose hazard functions can be non-monotone (initially rising then falling), suitable for processes with early peak risk.
- **Maximum Likelihood for Censored Data** --- Develops the likelihood function that accounts for both observed events and censored observations, and derives the MLE for parametric survival models.

### 21.4 Cox Proportional Hazards Model

The semi-parametric workhorse of survival analysis:

- **Partial Likelihood and the Cox Model** --- Introduces Cox's proportional hazards formulation, which models the effect of covariates on the hazard without specifying the baseline hazard, and derives the partial likelihood for estimation.
- **Interpreting Hazard Ratios** --- Explains how exponentiated Cox coefficients yield hazard ratios, providing a multiplicative measure of the effect of each covariate on the instantaneous event rate.
- **Proportional Hazards Assumption** --- Discusses the critical assumption that hazard ratios remain constant over time, and describes methods for assessing whether this assumption holds.
- **Model Diagnostics (Schoenfeld Residuals)** --- Presents Schoenfeld residuals as the primary diagnostic tool for detecting violations of the proportional hazards assumption and other model misspecifications.

### 21.5 Model Comparison and Selection

Choosing among competing survival models:

- **Non-Parametric vs Parametric vs Semi-Parametric** --- Compares the three modeling paradigms in terms of assumptions, flexibility, interpretability, and efficiency, providing guidance on when each approach is most appropriate.
- **AIC and Concordance Index** --- Introduces the two main tools for survival model selection: AIC for comparing nested parametric models and the concordance index (C-index) for evaluating discriminative ability across all model types.

### 21.6 Code

Complete Python implementations:

- **Kaplan--Meier Survival Curves and Log-Rank Test** --- Fits Kaplan--Meier estimators, plots survival curves with confidence intervals, and performs log-rank tests for group comparisons.
- **Parametric Survival Models** --- Fits exponential, Weibull, log-normal, and log-logistic models to censored data using maximum likelihood.
- **Cox Proportional Hazards** --- Fits the Cox model, interprets hazard ratios, checks the proportional hazards assumption, and examines Schoenfeld residuals.
- **Survival Model Comparison** --- Compares non-parametric, parametric, and semi-parametric models on the same dataset using AIC and the concordance index.

### 21.7 Exercises

Practice problems covering censoring concepts, Kaplan--Meier estimation, parametric model fitting, Cox regression interpretation, hazard ratio calculations, and model comparison strategies.

---

## Prerequisites

This chapter builds on:

- **Chapter 6** (Statistical Estimation) --- Maximum likelihood estimation, likelihood functions for censored data, and Fisher information for constructing confidence intervals.
- **Chapter 4** (Distributions) --- The exponential distribution, its memoryless property, and the Weibull distribution as a generalization; log-normal and logistic distributions.
- **Chapter 9** (Hypothesis Testing) --- Null and alternative hypotheses, test statistics, and p-values, which underpin the log-rank test and tests of the proportional hazards assumption.
- **Chapter 13** (Linear Regression) --- The regression modeling framework, coefficient interpretation, and model diagnostics, which extend naturally to the Cox model's linear predictor.

---

## Key Takeaways

1. Survival analysis provides a principled framework for modeling time-to-event data where some observations are censored---simply dropping censored subjects or treating them as events leads to biased results.
2. The Kaplan--Meier estimator is a robust, distribution-free method for estimating the survival function, while the log-rank test allows formal comparison of survival curves between groups.
3. Parametric models (exponential, Weibull, log-normal, log-logistic) offer greater efficiency and smoother estimates when their distributional assumptions are met, with the Weibull model providing a flexible default via its shape parameter.
4. The Cox proportional hazards model strikes a balance between flexibility and interpretability: it models covariate effects without specifying the baseline hazard, and its exponentiated coefficients have a direct interpretation as hazard ratios.
5. The proportional hazards assumption must be checked---Schoenfeld residuals are the primary diagnostic---since violations invalidate the standard interpretation of Cox model coefficients.
6. Model selection in survival analysis relies on AIC for parametric model comparison and the concordance index (C-index) for assessing discriminative performance across all model types.
