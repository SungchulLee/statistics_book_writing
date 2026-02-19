# Chapter 6: Statistical Estimation

## Overview

This chapter formalizes the central question of statistical inference: given observed data, how do we construct good estimators, and how do we measure what "good" means? It develops the theoretical framework for evaluating estimator quality --- bias, variance, mean squared error, consistency, efficiency, and sufficiency --- and then presents three systematic methods for constructing estimators: Maximum Likelihood Estimation, the Method of Moments, and Bayesian estimation.

---

## Chapter Structure

### 6.1 Estimator Quality

The theoretical criteria for evaluating and comparing statistical estimators:

- **Bias--Variance Tradeoff** --- Formalizes the fundamental tension between systematic error (bias) and sensitivity to sampling variability (variance), showing that minimizing total estimation error requires balancing these two competing sources of error.
- **Mean Squared Error** --- Defines MSE as the expected squared deviation of an estimator from the true parameter value and establishes the decomposition $\text{MSE} = \text{Variance} + \text{Bias}^2$, providing a single criterion that captures both bias and variance.
- **Consistency and Asymptotic Normality** --- Establishes that a consistent estimator converges in probability to the true parameter as $n \to \infty$, and that asymptotically normal estimators permit approximate confidence intervals and tests for large samples.
- **Efficiency and Cramer--Rao Lower Bound** --- Introduces Fisher information and the CRLB, which provides an absolute lower bound on the variance of any unbiased estimator, defining efficiency as the attainment of this bound.
- **Sufficiency and Minimal Sufficiency** --- Defines sufficient statistics via the Fisher--Neyman factorization theorem and introduces the Rao--Blackwell theorem, showing how sufficiency enables data reduction without loss of information about the parameter.

### 6.2 Maximum Likelihood Estimation

The most widely used method for parameter estimation, based on maximizing the probability of the observed data:

- **Likelihood Function** --- Defines the likelihood as the joint density of the observed data viewed as a function of the parameters, establishes the log-likelihood for computational convenience, and explains that likelihood ratios (not absolute values) are meaningful.
- **Introduction to MLE** --- Presents the MLE principle of finding parameter values that make the observed data most probable, with the formal definition and the log-likelihood maximization procedure.
- **MLE for Bernoulli Distribution** --- Derives the closed-form MLE $\hat{p} = \bar{X}$ for the success probability of Bernoulli trials.
- **MLE for Normal Distribution** --- Derives the joint MLEs $\hat{\mu} = \bar{X}$ and $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$ for the normal distribution parameters.
- **MLE for Poisson Distribution** --- Derives $\hat{\lambda} = \bar{X}$ as the MLE for the Poisson rate parameter and shows it is unbiased and efficient.
- **MLE for Exponential Distribution** --- Derives $\hat{\lambda} = 1/\bar{X}$ as the MLE for the exponential rate parameter and notes it is biased but consistent.
- **Capture-Recapture Method** --- Applies the MLE framework to the ecological problem of estimating population size from mark-recapture data using the hypergeometric likelihood.
- **Asymptotic Properties of MLE** --- States the three key large-sample properties of MLE: consistency, asymptotic normality, and asymptotic efficiency (achieving the CRLB), under regularity conditions.
- **Fisher Information and Standard Errors** --- Shows how Fisher information quantifies the amount of information data carry about a parameter and how it yields asymptotic standard errors for MLEs.

### 6.3 Method of Moments

A classical and intuitive approach to estimation that equates population moments to sample moments:

- **Method of Moments Foundations** --- Defines the MoM procedure of equating the first $p$ population moments to their sample counterparts and solving for $p$ parameters, noting its consistency, simplicity, and potential drawbacks.
- **MoM for Common Distributions** --- Derives closed-form MoM estimators for the normal, gamma, and beta distributions by matching first and second moments.
- **Generalized Method of Moments (GMM)** --- Extends MoM to overidentified models with more moment conditions than parameters, using a weighting matrix to obtain efficient estimators --- the dominant estimation framework in empirical finance and econometrics.
- **MoM vs MLE Comparison** --- Systematically compares MoM and MLE across computation, efficiency, consistency, invariance, and robustness, with guidance on when to use each method.
- **Likelihood and Estimation Overview** --- Provides a comprehensive treatment of the likelihood function as the cornerstone of parametric inference, connecting it to MLE, sufficiency, and model comparison.
- **Method of Moments Overview** --- Offers an extended treatment of MoM from definitions through population and sample moments to practical examples, positioning MoM within the broader estimation landscape.

### 6.4 Bayesian Estimation

A framework that combines prior beliefs with observed data to produce posterior distributions over parameters:

- **Prior, Likelihood, and Posterior** --- Presents Bayes' theorem as the foundation of Bayesian inference, defining the prior, likelihood, posterior, and marginal likelihood, along with posterior point estimates (mean, median, MAP).
- **Conjugate Priors** --- Defines conjugate prior families whose posterior belongs to the same distributional family as the prior, with a table of common conjugate pairs (Beta-Binomial, Gamma-Poisson, Normal-Normal, Gamma-Exponential).
- **MAP Estimation** --- Defines Maximum A Posteriori estimation as the mode of the posterior distribution, connects it to MLE with a log-prior regularization term, and links Gaussian and Laplace priors to Ridge and Lasso regularization respectively.

### 6.5 Code

Complete Python implementations demonstrating estimation methods in practice:

- **Estimation Methods Comparison** --- Side-by-side comparison of MLE, MoM, and Bayesian estimators on the same data.
- **MLE Optimization Examples** --- Numerical optimization of log-likelihood functions using `scipy.optimize`.
- **Fisher Information Computation** --- Computes observed and expected Fisher information for standard distributions.
- **Bayesian Estimation Demonstrations** --- Implements Bayesian updating with conjugate priors and visualizes prior-to-posterior evolution.
- **Capture-Recapture MLE** --- Applies MLE to the capture-recapture population estimation problem.
- **Log-Likelihood Visualization** --- Plots log-likelihood surfaces to build intuition for the MLE optimization landscape.

### 6.6 Exercises

Practice problems covering estimator comparison (bias, variance, MSE), MLE derivations for uniform and normal distributions, MoM vs MLE tradeoffs, Bayesian shrinkage estimators, Fisher information calculations, and sufficiency.

---

## Prerequisites

This chapter builds on:

- **Chapter 3** (Foundations of Probability) --- Random variables, expectation, variance, and moment generating functions.
- **Chapter 4** (Distributions) --- Properties of the normal, Bernoulli, Poisson, and exponential distributions.
- **Chapter 5** (Sampling Distributions) --- The concept of a statistic as a random variable and the behavior of sample means and variances across repeated samples.

---

## Key Takeaways

1. The bias--variance tradeoff is fundamental: minimizing MSE (= Variance + Bias$^2$) often requires accepting some bias in exchange for reduced variance.
2. Maximum Likelihood Estimation is the workhorse of parametric inference --- under regularity conditions it is consistent, asymptotically normal, and asymptotically efficient (achieves the CRLB).
3. The Method of Moments provides a simpler alternative that yields closed-form estimators by matching population and sample moments, though it is generally less efficient than MLE.
4. Bayesian estimation incorporates prior knowledge through Bayes' theorem; conjugate priors yield tractable posteriors, and MAP estimation bridges Bayesian and frequentist approaches.
5. Fisher information quantifies how much data tell us about a parameter, setting a fundamental lower bound (CRLB) on the precision of any unbiased estimator.
6. Sufficiency identifies the minimal data summaries that capture all information about the parameter, enabling optimal estimation via the Rao--Blackwell and Lehmann--Scheffe theorems.
