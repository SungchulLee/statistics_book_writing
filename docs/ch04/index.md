# Chapter 4: Distributions

## Overview

This chapter catalogs the most important probability distributions used in statistics and data science, covering both discrete and continuous families. It then develops the multivariate framework needed to analyze the joint behavior of multiple random variables, including joint, marginal, and conditional distributions, as well as covariance and correlation. Together, these tools form the distributional foundation for all subsequent chapters on sampling, estimation, and inference.

---

## Chapter Structure

### 4.1 Discrete Distributions

The core discrete probability models built from independent Bernoulli trials and event-counting processes:

- **Bernoulli and Binomial** --- The Bernoulli distribution models a single success/failure trial, while the binomial distribution counts the number of successes in $n$ independent trials, forming the foundation of discrete probability modeling.
- **Geometric and Negative Binomial** --- The geometric distribution models the number of trials until the first success, and the negative binomial generalizes this to the number of trials until the $r$-th success in sequential Bernoulli experiments.
- **Poisson Distribution** --- Models the number of events occurring in a fixed interval of time or space given a known average rate, with applications in finance (trade arrivals, default counts), insurance (claim frequency), and queueing theory.

### 4.2 Continuous Distributions

The fundamental continuous probability models that appear throughout statistics and applied sciences:

- **Uniform Distribution** --- Assigns equal probability to all values in an interval $[a, b]$, serving as the foundation for random number generation, simulation, and probability integral transforms.
- **Exponential Distribution** --- Models the time between events in a Poisson process, possessing the unique memoryless property among continuous distributions, with applications to inter-arrival times, waiting times, and component lifetimes.
- **Normal Distribution** --- The most fundamental probability distribution in statistics, describing continuous data that cluster symmetrically around a central value in a characteristic bell curve, and serving as the basis for the Central Limit Theorem.

### 4.3 Multivariate Structure

The mathematical framework for describing the joint behavior of two or more random variables:

- **Joint Distributions** --- Defines joint PMFs and PDFs that describe the probabilistic behavior of multiple random variables simultaneously, capturing how variables relate to and depend on each other.
- **Marginal and Conditional Distributions** --- Shows how to recover individual distributions by marginalizing over other variables, and how to describe one variable given a specific value of another, with applications to Bayesian reasoning and regression.
- **Covariance and Correlation** --- Quantifies the linear relationship between two random variables, with covariance measuring direction and magnitude of co-movement and correlation normalizing this to a dimensionless quantity between $-1$ and $+1$.
- **Independence vs Zero Correlation** --- Clarifies the important distinction that while independence implies zero correlation, the converse is false in general, with proofs, counterexamples, and the special Gaussian case where the two notions coincide.

### 4.4 Code

Complete Python implementations using `scipy.stats` and `matplotlib`:

- **Normal PDF with scipy.stats** --- Visualizes the normal probability density function for various parameter settings.
- **Normal Random Variates** --- Generates random samples from the normal distribution.
- **Normal CDF and Quantiles** --- Computes cumulative probabilities and quantile values for the normal distribution.
- **Student-t PDF** --- Plots the Student's $t$ density for different degrees of freedom.
- **Chi-Square PDF** --- Visualizes the chi-square density for different degrees of freedom.
- **F-Distribution PDF** --- Plots the F-distribution density for various numerator and denominator degrees of freedom.
- **Normal PPF (Quantile Function)** --- Demonstrates the percent-point (inverse CDF) function for the normal distribution.
- **Normal Survival Function** --- Computes upper-tail probabilities using the survival function.
- **Exponential PDF** --- Visualizes the exponential density for different rate parameters.
- **Uniform PDF** --- Plots the uniform density over various intervals.
- **Logistic PDF (vs Normal)** --- Compares the logistic and normal densities to highlight their similar shapes and differing tail behavior.
- **Log-Normal PDF** --- Visualizes the log-normal distribution and its right-skewed shape.
- **Weibull PDF and Hazard** --- Plots the Weibull density and hazard function for various shape parameters.

### 4.5 Exercises

Practice problems covering discrete and continuous distribution calculations, multivariate probability, and dependence concepts.

---

## Prerequisites

This chapter builds on:

- **Chapter 3** (Foundations of Probability) --- Random variables (discrete and continuous), PMFs, PDFs, CDFs, expectation, variance, covariance, and moment generating functions.

---

## Key Takeaways

1. The Bernoulli, binomial, geometric, negative binomial, and Poisson distributions form a complete toolkit for modeling count and trial-based phenomena.
2. The uniform, exponential, and normal distributions are the foundational continuous models, each with distinct properties (equal likelihood, memorylessness, and the bell curve, respectively).
3. Joint distributions capture the simultaneous behavior of multiple random variables, while marginal and conditional distributions allow us to extract information about individual variables and their dependence structure.
4. Covariance and correlation quantify linear association, but zero correlation does not imply independence except in special cases such as the multivariate normal.
5. Each distribution's properties (mean, variance, MGF) and relationships to other distributions (e.g., Poisson as a limit of binomial, exponential as continuous analogue of geometric) form a coherent network that recurs throughout statistical inference.
