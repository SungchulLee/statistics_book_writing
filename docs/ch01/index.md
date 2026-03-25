# Chapter 1: Data Collection

## Overview

This chapter explores how data is collected and how we learn from it. It contrasts two fundamental philosophies: the classical approach, where the researcher designs the data collection process before gathering data, and the modern approach, where the analyst works with data that already exists. The chapter concludes by introducing the three major machine learning paradigms -- supervised, unsupervised, and reinforcement learning -- that define how algorithms extract knowledge from data.

---

## Chapter Structure

### 1.1 Classical Approach

The classical statistical approach to designing data collection from the ground up:

- **Populations and Samples** -- Defines the distinction between the entire population of interest and the sample drawn from it, and explains why sampling is necessary for practical and economic reasons.
- **Parameters vs Statistics** -- Distinguishes fixed but unknown population parameters (e.g., mu, sigma) from computable sample statistics (e.g., x-bar, s) and explains why this distinction is the foundation of all inferential procedures.
- **Observational Studies** -- Describes research designs where the investigator observes without intervening, covering cross-sectional, cohort, case-control, and ecological study types.
- **Confounding and Association vs Causation** -- Explains how confounding variables can create spurious associations between variables, using classic examples (ice cream and drowning, coffee and lung cancer) to illustrate why correlation does not imply causation.
- **Controlled Experiments** -- Covers the structure of experiments with treatment and control groups, random assignment, and how controlled designs establish causal relationships.
- **Randomization and Blinding** -- Details how randomization distributes confounders across groups and how blinding (single, double, triple) prevents human expectation and bias from distorting results, including the gold-standard randomized controlled trial.
- **Sample Surveys and Sampling Methods** -- Covers simple random, stratified, cluster, and systematic sampling methods for selecting representative samples from populations.
- **Bias and Nonresponse** -- Examines sampling bias, nonresponse bias, response bias, and selection bias, illustrated by historical examples such as the 1936 Literary Digest disaster.
- **Survivorship Bias** -- Explores the danger of analyzing only "survivors" while ignoring failures, with examples from WWII bomber armor placement and the 2008 financial crisis.
- **Strengths and Limitations** -- Provides a systematic comparison of classical and modern approaches across dimensions such as causality, scalability, interpretability, and cost.
- **Design Your Data Collection** -- Summarizes the classical philosophy of "design first, collect second, analyze third" and its three main study types: observational studies, controlled experiments, and sample surveys.
- **Analyze Available Data** -- Introduces the modern philosophy of "data first, algorithm second, insight third" and its reliance on existing data sources such as transaction logs, sensor readings, and financial market data.

### 1.2 Modern Approach

The modern data-driven approach to learning from existing data:

- **Statistical Models vs Learning Algorithms** -- Contrasts parametric statistical models (explicit assumptions, interpretable parameters, uncertainty quantification) with machine learning algorithms (flexible, data-driven, prediction-focused), marking the key philosophical shift in modern data analysis.
- **Prediction vs Inference** -- Distinguishes between the goal of forecasting outcomes as accurately as possible (prediction) and the goal of understanding variable relationships and testing hypotheses (inference), and explains how each goal leads to different methodological choices.

### 1.3 Three Learning Paradigms

An overview of the three major paradigms in machine learning:

- **Unsupervised Learning (Pattern Discovery)** -- Covers learning without labels, including clustering (K-Means, hierarchical, DBSCAN), dimensionality reduction (PCA, t-SNE), and anomaly detection, with applications to portfolio diversification and market regime detection.
- **Supervised Learning (Prediction with Labels)** -- Covers learning from labeled input-output pairs for regression (continuous targets) and classification (discrete targets), including common methods such as linear regression, random forests, and neural networks.
- **Reinforcement Learning (Sequential Decisions)** -- Introduces agent-environment interaction, reward-based learning, the exploration-exploitation tradeoff, and applications to portfolio management and algorithmic trading.

### 1.4 Exercises

Practice problems covering populations and samples, study design, bias identification, the classical-modern contrast, and the three learning paradigms.

---

## Prerequisites

This chapter builds on:

- **Chapter 0** (Prerequisites) -- Basic mathematical notation and familiarity with Python for running code examples.

---

## Key Takeaways

1. The population is the complete group of interest; the sample is the subset we actually observe, and all of inferential statistics rests on this distinction.
2. Controlled experiments with randomization are the gold standard for establishing causation, while observational studies can only identify associations.
3. Confounding variables can create misleading associations; recognizing and controlling for confounders is essential for valid conclusions.
4. Bias (sampling, nonresponse, survivorship) can systematically distort results even with large sample sizes.
5. The modern approach leverages existing data and algorithms to discover patterns and make predictions at scale, complementing rather than replacing classical methods.
6. Understanding the difference between prediction and inference guides the choice of methods and the interpretation of results.
7. The three learning paradigms -- supervised, unsupervised, and reinforcement learning -- define the major ways algorithms extract knowledge from data.
