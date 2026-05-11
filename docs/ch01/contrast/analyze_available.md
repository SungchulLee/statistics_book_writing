# Analyze the Data You Have (Modern Approach)

## Overview

The **modern approach** to data analysis starts with data that **already exists**—transaction logs, sensor readings, social media posts, financial market data—and asks: *"What can I learn from this?"* Rather than designing a collection process, the analyst applies algorithms to discover patterns, make predictions, and extract insights from available data.

## Core Principle

> **Data first, algorithm second, insight third.**

The modern approach leverages the explosion of digital data and computational power. Data is often collected as a byproduct of operations (e.g., web clicks, trades, medical records) rather than through a deliberate research design.

## Three Learning Paradigms

### 1. Supervised Learning

Learn a mapping from inputs to labeled outputs. The algorithm is trained on historical data where the "answer" is known and then applied to new data.

### 2. Unsupervised Learning

Discover hidden structure in data without labels. Clustering, dimensionality reduction, and anomaly detection fall into this category.

### 3. Reinforcement Learning

Learn optimal sequential decisions through interaction with an environment, guided by reward signals rather than labeled examples.

## Strengths of the Modern Approach

- **Scalability**: Algorithms can process millions or billions of data points that would be impossible to collect through designed studies.
- **Flexibility**: Machine learning and deep learning models can capture highly complex, non-linear relationships without requiring the analyst to specify them in advance.
- **Speed**: Existing data can be analyzed immediately without waiting months or years for data collection.
- **Discovery**: Patterns and relationships that no researcher anticipated can emerge from exploratory analysis.
- **Unstructured data**: Images, text, audio, and video can be analyzed at scale.

## When This Approach Works Best

- Large volumes of data already exist.
- The goal is **prediction** rather than **causal explanation**.
- The data is high-dimensional or unstructured.
- Speed of analysis is important (e.g., real-time trading, recommendation systems).
- The problem is too complex for a simple statistical model.

## Limitations

- **Causality**: Without a designed experiment, it is difficult to distinguish correlation from causation. Associations discovered in observational data may be driven by confounders.
- **Data quality**: The analyst has no control over how the data was collected, leading to potential biases, missing values, and measurement errors.
- **Interpretability**: Complex models (deep networks, ensembles) may provide accurate predictions without explaining *why*.
- **Overfitting**: With flexible models and large feature spaces, there is a risk of fitting noise rather than signal—mitigated by cross-validation and regularization.
- **Ethical and privacy concerns**: Using existing data (especially personal data) raises questions about consent, fairness, and privacy.

## Key Takeaways

- The modern approach takes advantage of abundant existing data and powerful algorithms to extract predictions and insights.
- Its greatest strength is **scalability and flexibility**—handling problems and data types that classical methods were not designed for.
- Its greatest weakness is the difficulty of making **causal claims** without a designed study.
- In practice, the most effective data scientists combine both approaches: using modern algorithms for prediction and classical principles for causal reasoning and uncertainty quantification.

## Exercises

**Exercise 1.**
A data scientist at a tech company has access to clickstream data from 1 million users. They discover that users who view the help page spend 30% more time on the site. Should they recommend making the help page more prominent? Discuss the inferential challenges.

??? success "Solution to Exercise 1"
    The recommendation is premature. The data are observational (not from a randomized experiment), so the association between viewing the help page and engagement could be driven by confounders:

    - **Reverse causation:** Users who are already more engaged (spend more time) are more likely to explore the help page.
    - **User type:** Power users who explore help pages may naturally be heavier users regardless.
    - **Difficulty:** Users who find the product confusing visit help pages *and* spend more time struggling, not because help makes them more engaged.

    To establish a causal effect, the company should run an A/B test: randomly assign users to a version with a prominent help page versus the status quo and compare engagement metrics.

---

**Exercise 2.**
Explain the difference between *exploratory data analysis* (EDA) and *confirmatory data analysis* (CDA). Why is it problematic to use the same dataset for both?

??? success "Solution to Exercise 2"
    **Exploratory data analysis** searches for patterns, generates hypotheses, and builds intuition. It involves visualizations, summary statistics, and flexible model fitting without pre-specified hypotheses.

    **Confirmatory data analysis** tests pre-specified hypotheses with formal statistical procedures (p-values, confidence intervals) whose validity depends on the hypothesis being formulated before seeing the data.

    Using the same dataset for both is problematic because patterns discovered during EDA are, by definition, patterns that happen to appear in this particular sample. Testing those same patterns on the same data inflates the probability of finding "significant" results (data snooping / double dipping). The p-values are no longer valid because they assume the hypothesis was specified independently of the data. The remedy is data splitting (train/test) or pre-registration of hypotheses.

---

**Exercise 3.**
A researcher scrapes social media posts and finds that users who post about exercise have higher self-reported happiness. List three sources of bias specific to this observational web-scraped dataset.

??? success "Solution to Exercise 3"

    1. **Selection bias (non-representative sample):** Social media users are not representative of the general population. They skew younger, more urban, and more tech-savvy.
    2. **Self-presentation bias:** People selectively post positive content (exercise accomplishments, happy moments) while omitting negative experiences. The data reflect curated self-images, not true behavior or mood.
    3. **Measurement bias:** "Self-reported happiness" extracted from text (sentiment analysis) is a noisy proxy for actual well-being. Sarcasm, cultural norms, and language ambiguity introduce systematic errors.

    Additional biases include confounding (socioeconomic status affects both exercise habits and happiness) and missing data (users who are unhappy or sedentary may post less frequently, creating survivorship bias).

---

**Exercise 4.**
Describe the "garden of forking paths" problem when analyzing available data. How does it differ from classical p-hacking?

??? success "Solution to Exercise 4"
    The "garden of forking paths" (Gelman and Loken, 2013) refers to the many researcher degrees of freedom in data analysis: choices about variable definitions, outlier removal, subgroup selection, model specification, and transformation -- each of which could have gone differently. Even without deliberate p-hacking, the researcher makes these choices after seeing the data, guided (consciously or not) by what produces interesting results.

    Classical **p-hacking** involves explicitly trying multiple analyses and reporting only the significant ones. The garden of forking paths is more subtle: the researcher runs a single analysis but the specific analysis was implicitly selected from a large space of possibilities based on the data. The result is the same -- inflated false-positive rates -- but the researcher may honestly believe they did not try multiple analyses because they only reported one. The remedy is pre-registration, where the analysis plan is committed to before data collection.

---

**Exercise 5.**
A modern data team works with administrative records covering an entire population (e.g., all credit-card transactions over a year). Some claim that *with N → ∞, statistical inference becomes unnecessary*. Identify two reasons inferential reasoning still matters even with apparent population data.

??? success "Solution to Exercise 5"
    **(1) The data are still a sample — just from a different superpopulation.** "All transactions in 2024" is a sample from the implicit population "all transactions that could have occurred under similar conditions, including future years." Inference is about generalizing from the observed history to the future or to other plausible scenarios, not about generalizing from a finite sample to a fixed population. The data-generating process produces all the variability of interest.

    **(2) Even with a true census, *causal* and *counterfactual* questions are not answered by descriptive statistics.** Knowing the exact average revenue of every customer says nothing about what would happen if the prices changed. Counterfactuals require either explicit randomized experiments or strong identifying assumptions — the data's size is irrelevant.

    A third reason often cited: even large $N$ does not protect against **selection bias** in who appears in the data. Administrative records reflect what was recorded and survived to be recorded; non-customers and dropped transactions are invisible.

---

**Exercise 6.**
**External validity** is the question of whether findings on one dataset generalize to other contexts. A retailer's recommendation model was trained on US data and now must launch in the EU. List three distinct mechanisms by which the model could fail in the EU, and one practical mitigation strategy for each.

??? success "Solution to Exercise 6"
    **Covariate shift:** the joint distribution of user features differs (age distribution, income, browsing patterns). *Mitigation:* importance weighting, where training examples are reweighted to match the EU covariate distribution; or domain adaptation methods.

    **Concept drift / different mapping $P(Y \mid X)$:** the same features predict different outcomes — EU users with US-equivalent profiles may shop differently due to cultural or regulatory differences. *Mitigation:* deploy with monitoring and quickly re-train on local data once enough EU samples accumulate. Avoid hand-tuned thresholds that assume US-specific feature distributions.

    **Regulatory / structural shift:** GDPR limits what data can be collected; consent UIs change which features are available at inference time; cross-border data restrictions prevent fine-tuning on EU data within US infrastructure. *Mitigation:* model designs that gracefully degrade when features are missing (e.g., feature-dropout training, hierarchical models that can use sparser EU features), and a deployment plan that includes legal and engineering review for regional differences.

    The underlying principle: any model deployed outside the distribution it was trained on requires monitoring, calibration, and often retraining. Treating the US-trained model as ground truth for the EU is exactly the kind of error this paradigm encourages.
