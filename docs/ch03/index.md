# Chapter 3: Foundations of Probability

## Overview

This chapter builds the probability foundation that underpins all of statistical inference. Starting from sample spaces and the axioms of probability, it develops conditional probability, Bayes' theorem, and independence. It then formalizes random variables (discrete and continuous), their distributions (PMF, PDF, CDF), and summary measures (expectation, variance, covariance, moment generating functions). The chapter culminates with the powerful limit theorems -- the Law of Large Numbers and the Central Limit Theorem -- that connect probability theory to practical statistics.

---

## Chapter Structure

### 3.1 Probability Theory

The axiomatic foundation of probability:

- **Sample Spaces and Events** -- Defines the sample space as the set of all possible outcomes, events as subsets of the sample space, and the basic set operations (union, intersection, complement) used to combine events.
- **Axioms of Probability** -- Presents three equivalent formulations of the probability axioms (naive, intermediate, and Kolmogorov's measure-theoretic), establishing the formal rules for assigning probabilities to events.
- **Conditional Probability** -- Defines how the probability of an event changes when we learn that another event has occurred, introducing the formula P(A|B) = P(A and B)/P(B) and the intuition of restricting the sample space.
- **Bayes' Theorem** -- Derives the formula for reversing the direction of conditioning (computing P(A|B) from P(B|A)), identifying the prior, likelihood, posterior, and evidence, with applications to medical diagnosis, spam filtering, and finance.

### 3.2 Independence

The concept of probabilistic independence and its extensions:

- **Independence of Events** -- Defines independence as P(A and B) = P(A)P(B), contrasts it with mutual exclusivity, and extends the definition to mutual independence of multiple events.
- **Conditional Independence** -- Extends independence by introducing a conditioning event, showing that two events can be marginally dependent but conditionally independent (and vice versa), with applications to graphical models and Bayesian networks.

### 3.3 Random Variables

Formalizing the mapping from outcomes to numbers:

- **Discrete Random Variables** -- Defines random variables as functions from the sample space to the real line, introduces discrete random variables with countable support, and develops the brick-on-a-number-line metaphor for understanding distributions.
- **Continuous Random Variables** -- Extends the framework to variables that take values over continuous intervals, introducing the probability density function and the key property that P(X = x) = 0 for any single point.
- **PMF, PDF, and CDF** -- Provides a unified treatment of the three fundamental distribution functions: the probability mass function for discrete variables, the probability density function for continuous variables, and the cumulative distribution function that applies to both.

### 3.4 Expectation and Moments

Summary measures derived from a distribution:

- **Expectation and Linearity** -- Defines the expected value as the long-run average (center of mass of the distribution) and establishes the linearity of expectation, one of the most powerful and widely used properties in probability.
- **Variance and Covariance** -- Defines variance as the expected squared deviation from the mean, introduces covariance and correlation for measuring how two random variables move together, with applications to risk measurement and portfolio theory.
- **Moment Generating Functions** -- Introduces the MGF as a tool that encodes all moments of a distribution into a single function, enabling elegant computation of expectations, proofs of limit theorems, and distribution characterization.

### 3.5 Limit Theorems

The fundamental convergence results connecting probability to statistics:

- **Law of Large Numbers** -- States and proves both the weak and strong forms of the LLN, showing that the sample mean converges to the population mean as the sample size grows, providing the theoretical foundation for estimation.
- **Central Limit Theorem** -- Establishes that the standardized sample mean converges in distribution to a standard normal regardless of the original distribution, justifying the widespread use of normal-based inference methods.
- **Berry-Esseen Theorem** -- Provides an explicit upper bound on the rate of convergence in the CLT, quantifying how quickly the normal approximation becomes accurate as a function of sample size and the third absolute moment.

### 3.6 Exercises

Practice problems covering sample spaces, probability axioms, conditional probability, Bayes' theorem, independence, random variables, expectation, variance, and the limit theorems.

---

## Prerequisites

This chapter builds on:

- **Chapter 0** (Prerequisites) -- Set theory notation, sequences and limits, and summation/integration for defining probabilities and expectations.
- **Chapter 1** (Data Collection) -- The concept of populations and samples, which motivates the need for a formal probability framework.
- **Chapter 2** (Descriptive Statistics) -- Empirical measures of center (mean, median) and spread (variance, IQR) that find their theoretical counterparts in expectation and variance of random variables.

---

## Key Takeaways

1. Probability theory rests on three axioms (non-negativity, normalization, additivity) that provide a rigorous foundation for reasoning about uncertainty.
2. Conditional probability and Bayes' theorem provide the machinery for updating beliefs in light of new evidence.
3. Independence simplifies probability calculations and is a fundamental assumption behind most statistical methods.
4. Random variables formalize the bridge between abstract outcomes and numerical analysis, with PMF, PDF, and CDF providing complete distributional descriptions.
5. Expectation gives the center of a distribution, variance measures its spread, and covariance captures how two variables co-move.
6. Moment generating functions provide a compact encoding of all distributional information and are a powerful tool for proving theoretical results.
7. The Law of Large Numbers guarantees that sample averages converge to population means, justifying the use of data for estimation.
8. The Central Limit Theorem explains why the normal distribution appears so frequently in practice and underpins the construction of confidence intervals and hypothesis tests.
