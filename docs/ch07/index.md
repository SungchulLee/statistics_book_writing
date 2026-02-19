# Chapter 7: Estimation of $\mu$ and $\sigma^2$

## Overview

This chapter applies the general estimation theory developed in Chapter 6 to the two most fundamental population parameters: the mean $\mu$ and the variance $\sigma^2$. It examines the sample mean and sample variance as estimators in detail --- their bias, consistency, efficiency, and MSE --- derives the Gaussian MLEs, and explores robust alternatives for situations where the normality assumption fails, such as heavy-tailed or skewed distributions.

---

## Chapter Structure

### 7.1 Estimation of the Mean

A thorough analysis of the sample mean and its alternatives as estimators of the population mean:

- **Sample Mean as Estimator** --- Presents $\bar{X}$ as the most fundamental estimator in statistics, deriving its expectation, variance, and distribution, and explaining why it is the natural starting point for estimating $\mu$.
- **Bias and Consistency** --- Proves that $\bar{X}$ is exactly unbiased for all sample sizes and consistent under mild conditions (finite variance), with detailed discussion of the minimal assumptions required.
- **Efficiency of the Sample Mean** --- Shows that $\bar{X}$ achieves the Cramer--Rao lower bound for the normal mean, making it the most efficient unbiased estimator, and introduces asymptotic relative efficiency (ARE) for comparing estimators under non-normality.
- **Trimmed and Winsorized Means** --- Introduces robust alternatives that reduce the influence of outliers by removing or capping extreme observations, with a comparison of breakdown points and efficiency under normality.

### 7.2 Estimation of the Variance

A detailed study of variance estimators, the origin of Bessel's correction, and robust alternatives:

- **Naive Variance Estimator** --- Defines the biased estimator $\tilde{S}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$, derives its downward bias of $-\sigma^2/n$ through a fundamental algebraic identity, and explains the intuition for why dividing by $n$ underestimates the true variance.
- **Bessel's Correction** --- Proves that dividing by $n-1$ instead of $n$ yields the unbiased estimator $S^2$, explains the degrees-of-freedom interpretation, and connects it to the standard implementation in statistical software.
- **MSE of Variance Estimators** --- Compares the MLE ($1/n$), Bessel-corrected ($1/(n-1)$), and MSE-optimal ($1/(n+1)$) variance estimators for normal data, demonstrating the bias--variance tradeoff in a concrete setting.
- **Robust Variance Estimators (MAD, IQR-based)** --- Introduces the Median Absolute Deviation and IQR-based estimators as outlier-resistant alternatives, comparing their breakdown points and efficiency relative to the classical sample variance.

### 7.3 Gaussian MLE

The maximum likelihood approach to jointly estimating the mean and variance of a normal population:

- **MLE of $\mu$ and $\sigma^2$** --- Derives the closed-form MLEs $\hat{\mu} = \bar{X}$ and $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$ from the normal log-likelihood, analyzing the properties of each estimator.
- **Bias of Gaussian MLE for $\sigma^2$** --- Shows that the MLE of variance is biased with $E[\hat{\sigma}^2_{MLE}] = \frac{n-1}{n}\sigma^2$, quantifies the bias as $-\sigma^2/n$, and notes that despite being biased, the MLE can have lower MSE than the unbiased estimator.
- **Sufficiency and Completeness** --- Identifies $(\bar{X}, S^2)$ as jointly sufficient and complete for $(\mu, \sigma^2)$ in the normal model, and applies the Lehmann--Scheffe theorem to establish $\bar{X}$ and $S^2$ as the unique UMVUEs for their respective parameters.

### 7.4 Estimation Under Non-Normality

How the performance of standard estimators changes when the normality assumption is violated:

- **Heavy-Tailed Distributions** --- Examines the impact of heavy tails (e.g., Student's $t$, Cauchy, Pareto) on the sample mean, showing that extreme observations can inflate variance and that robust estimators may outperform the mean, with particular relevance to financial return data.
- **Skewed Distributions** --- Discusses how skewness causes the mean and median to diverge, explains when the median may be a more representative measure of center, and introduces transformations (log, Box-Cox) to reduce skewness and improve mean-based inference.

### 7.5 Code

Complete Python implementations for exploring estimator properties through simulation:

- **Sample Mean Properties** --- Simulates the unbiasedness and variance reduction of the sample mean with increasing sample size.
- **Consistency and Convergence** --- Visualizes how the sample mean converges to the population mean as $n$ grows.
- **Variance Estimators** --- Compares the behavior of $1/n$, $1/(n-1)$, and $1/(n+1)$ variance estimators through simulation.
- **Bessel's Correction** --- Demonstrates the bias of the naive estimator and the correction provided by dividing by $n-1$.
- **Gaussian MLE** --- Implements maximum likelihood estimation for the normal distribution and visualizes the log-likelihood surface.
- **Return Estimation** --- Applies mean and variance estimation methods to financial return data.
- **Robust Estimators Comparison** --- Compares classical and robust estimators (trimmed mean, MAD, IQR-based) on clean and contaminated data.

### 7.6 Exercises

Practice problems covering unbiasedness proofs, variance under correlated data, relative efficiency of mean vs median, shrinkage estimators, Bessel's correction derivations, MSE-optimal variance estimators, and robust estimation in finance.

---

## Prerequisites

This chapter builds on:

- **Chapter 4** (Distributions) --- Properties of the normal distribution, heavy-tailed distributions, and skewness.
- **Chapter 5** (Sampling Distributions) --- The sampling distributions of $\bar{X}$ and $S^2$, standard error, and the chi-square connection.
- **Chapter 6** (Statistical Estimation) --- Bias, variance, MSE, consistency, efficiency, the Cramer--Rao lower bound, MLE methodology, and sufficiency.

---

## Key Takeaways

1. The sample mean $\bar{X}$ is unbiased for $\mu$ under minimal assumptions and is the most efficient unbiased estimator of the normal mean (it achieves the CRLB).
2. The naive variance estimator (dividing by $n$) is biased downward by $\sigma^2/n$; Bessel's correction (dividing by $n-1$) removes this bias, though the MSE-optimal estimator divides by $n+1$.
3. The Gaussian MLEs are $\hat{\mu} = \bar{X}$ and $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$; the pair $(\bar{X}, S^2)$ forms a jointly sufficient and complete statistic, yielding the unique UMVUEs for $\mu$ and $\sigma^2$.
4. Under heavy-tailed distributions, the sample mean can perform poorly due to extreme observations; robust alternatives such as trimmed means and the median offer better stability at the cost of some efficiency under normality.
5. For skewed data, the mean and median can differ substantially; log or Box-Cox transformations can reduce skewness and improve the reliability of mean-based inference.
6. The choice between classical and robust estimators depends on the data-generating process: when normality holds, classical estimators are optimal; when it does not, robust methods provide essential protection against outliers and heavy tails.
