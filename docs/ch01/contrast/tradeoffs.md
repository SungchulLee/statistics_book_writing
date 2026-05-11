# Strengths and Limitations of Each Approach

The classical (designed-collection) and modern (algorithmic-learning) approaches are complementary, not competing. Understanding their trade-offs guides the choice of methodology for any given problem. The most experienced data scientists know which paradigm a question belongs to, recognize the rare hybrid cases that need both, and avoid the common error of using algorithms for causal questions or rigid hypothesis tests for prediction problems.

## Definition

| Dimension | Classical (Designed Collection) | Modern (Algorithmic Learning) |
|---|---|---|
| Starting point | Research question → design → data | Existing data → algorithm → insight |
| Primary goal | Inference, causal understanding | Prediction, pattern discovery |
| Causality | Strong, via randomization or quasi-experiments | Weak; association only without further assumptions |
| Scalability | Limited by data-collection cost | Routinely scales to billions of observations |
| Interpretability | High — parameters have substantive meaning | Often low — many models are black-box |
| Uncertainty quantification | Built in (CIs, p-values, decision rules) | Often retrofitted via bootstrap, calibration, conformal prediction |
| Sample size needs | Small to moderate | Large preferred; works best with $n \gg p$ |

## Explanation

### When the classical approach wins

- **Causal questions**: clinical trials, A/B tests, policy evaluation. Randomization is the only routine way to identify causal effects without invoking strong assumptions.
- **Regulatory contexts**: FDA, EMA, and many regulators require formal experimental designs with pre-registered protocols.
- **Precise uncertainty matters**: confidence intervals with valid coverage, $p$-values with known error rates.
- **Generalizable to the target population**: probability sampling produces estimators with quantifiable bias and variance for population parameters.

### When the modern approach wins

- **Pure prediction**: forecasting churn, demand, defaults, click-through rates. The algorithm with the lowest out-of-sample loss wins; the structure of the model is irrelevant.
- **High-dimensional or unstructured data**: text, images, audio, graphs — domains where flexible function approximators outperform any parametric model.
- **Already-collected data at scale**: web logs, transactional records, sensor streams. Designing a new study would be impossible; the modern approach extracts what value can be extracted.
- **Iterative engineering**: model-deployment cycles where the goal is to push a metric, not to test a theory.

### Hybrids that combine both

- **Double / debiased machine learning** (Chernozhukov et al., 2018): use flexible ML estimators for nuisance functions, then apply orthogonalized score equations to recover unbiased causal effect estimates.
- **Causal forests** (Wager & Athey, 2018): tree-based methods for heterogeneous treatment effects from experimental or observational data.
- **Designed data, modern analysis**: a randomized clinical trial that uses a neural network for the primary endpoint analysis still inherits causal validity from randomization.
- **Post-hoc interpretability**: SHAP, LIME, integrated gradients — make black-box predictions partially interpretable without sacrificing predictive accuracy.

### Failure modes to recognize

- **Classical hammer, modern nail**: insisting on a linear model with a hand-picked set of predictors when the relationships are clearly nonlinear and the sample is huge. The model is interpretable but poorly predictive.
- **Modern hammer, classical nail**: training a deep learning model on observational data and interpreting its feature importances as causal effects. The model may predict well in-distribution but its "explanations" reflect spurious correlations.
- **Pretending one is the other**: reporting $p$-values from a model selected via cross-validation, or claiming "validation accuracy" is the same as "external validity."

## Examples

```python
"""Classical A/B test (inference) vs. modern prediction on the same setup."""

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n = 500
true_effect = 2.0

# === Classical A/B test ===
group = rng.choice([0, 1], size=n)
outcome = 10 + true_effect * group + rng.normal(0, 5, n)
t_stat, p_val = stats.ttest_ind(outcome[group == 1], outcome[group == 0])
print("=== Classical A/B test ===")
print(f"Estimated effect: {outcome[group == 1].mean() - outcome[group == 0].mean():+.2f} "
      f"(true {true_effect})")
print(f"p = {p_val:.4f}")

# === Modern prediction from observational features ===
X = rng.standard_normal((n, 3))
y = 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.standard_normal(n)
train, test = np.arange(n // 2), np.arange(n // 2, n)
beta_train = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
mse_test = np.mean((y[test] - X[test] @ beta_train) ** 2)
print("\n=== Modern prediction ===")
print(f"Test MSE = {mse_test:.3f}")
```

## Exercises

**Exercise 1.**
For each scenario, state whether the **classical** or **modern** approach is more appropriate, and justify your answer.

**(a)** A pharmaceutical company wants to determine whether a new vaccine reduces infection rates.
**(b)** An e-commerce company wants to predict which products a customer is likely to purchase next.
**(c)** A government agency wants to estimate the unemployment rate with a known margin of error.
**(d)** A bank wants to flag potentially fraudulent credit-card transactions in real time.
**(e)** A health system needs to identify the causal effect of a new triage protocol on patient mortality.

??? success "Solution to Exercise 1"
    (a) **Classical** — establishing a causal effect requires a randomized controlled trial with prespecified protocol.
    (b) **Modern** — recommendation is a prediction problem on existing transactional and browsing data.
    (c) **Classical** — known margin of error requires a probability sample with planned size.
    (d) **Modern** — fraud detection at scale demands flexible algorithms (gradient boosting, neural nets) on millions of transactions.
    (e) **Classical** if random assignment is feasible (cluster RCT); otherwise a hybrid causal-inference approach (instrumental variables, difference-in-differences, or stepped-wedge design) drawing on classical identification with modern estimation.

---

**Exercise 2.**
A team has 10 years of customer data and wants to (a) predict which customers will churn next month, and (b) understand *why* customers churn so they can change product policy. Which approach fits each goal, and how would using the wrong one fail?

??? success "Solution to Exercise 2"
    **(a) Prediction → modern.** Train a gradient-boosted model on historical features; the goal is to rank customers by churn risk for retention outreach. Trying to use a hand-built parametric model with only a handful of predictors would likely leave predictive accuracy on the table.

    **(b) Understanding why → classical (or hybrid causal).** Even if the prediction model has accurate features, its feature importances are not causal. Customers who call customer support frequently might predict churn, but reducing customer-support frequency would not reduce churn — both are caused by an underlying dissatisfaction. Establishing causes requires either A/B testing potential interventions or careful causal-inference analysis on observational data with explicit identifying assumptions.

    Using the modern approach for goal (b) leads to "fix the symptom, not the cause" policy mistakes. Using the classical approach for goal (a) leaves predictive accuracy unrealized.

---

**Exercise 3.**
Explain the role of **out-of-sample evaluation** in the modern paradigm. Why is in-sample $R^2$ unreliable for assessing predictive accuracy of a flexible model?

??? success "Solution to Exercise 3"
    A sufficiently flexible model (deep trees, neural nets, kernel methods) can fit *any* training set arbitrarily well — pushing in-sample $R^2$ toward 1 even when there is no real signal. The in-sample fit measures memorization, not generalization.

    **Out-of-sample evaluation** breaks the dependence: a held-out test set (or $k$-fold cross-validation) measures how well the model performs on data not used in training, which is what matters for deployment. This is the modern analog of the classical insistence on unbiased estimators — both are ways to prevent the model-selection process from inflating reported performance.

    The relationship between in-sample and out-of-sample error is governed by **generalization bounds** in statistical learning theory (VC dimension, Rademacher complexity, PAC-Bayes). The practical upshot: never trust an $R^2$ that wasn't measured on data the model didn't see.

---

**Exercise 4.**
Causal inference from observational data with modern flexible methods (double ML, causal forests) requires the **conditional ignorability** assumption. State the assumption and explain why it is the "modern" analog of randomization in classical experiments.

??? success "Solution to Exercise 4"
    **Conditional ignorability:** $(Y(0), Y(1)) \perp T \mid X$. Conditional on observed covariates $X$, treatment assignment $T$ is independent of the potential outcomes.

    **Why it parallels randomization:** randomization in a classical RCT guarantees $(Y(0), Y(1)) \perp T$ *unconditionally* — assignment carries no information about potential outcomes. Observational analysis weakens this to *conditional* independence: assignment is "as-if random" once we condition on $X$.

    The classical version is mechanical (the investigator ensures it by design); the observational version is an *assumption* about the data-generating process that cannot be verified from the data alone. Modern ML methods relax the parametric form of the model but cannot relax this fundamental identifying assumption. They make the analysis *more flexible*, not the assumptions *weaker*.

---

**Exercise 5.**
Give an example of a setting where a classical-paradigm analysis would produce **valid inference but poor prediction**, and another where a modern-paradigm analysis would produce **excellent prediction but invalid inference**.

??? success "Solution to Exercise 5"
    **Valid inference, poor prediction:** an RCT of a new drug with $n = 200$ patients. The treatment-effect estimate is unbiased with a 95% CI, and the $p$-value has correct coverage. But if a clinician wants to *predict* a new patient's outcome under treatment, the model is just a sample mean plus or minus noise — it ignores patient covariates and individual characteristics that a flexible ML model would exploit for more personalized prediction.

    **Excellent prediction, invalid inference:** a gradient-boosted tree predicting loan defaults from 200 features on a million-row dataset. Out-of-sample AUC is 0.92 — excellent. But if the bank asks "is income predictive of default *causally*, or only through correlated variables?", the model cannot answer: among the 200 features, some are confounders, some are mediators, some are colliders, and the SHAP importances confuse all three.

    The two paradigms answer different questions; using either model to answer the *other* question is the most common pitfall in practice.

---

**Exercise 6.**
A startup A/B tests 1,000 small interface changes per year. Each test runs for 7 days at $\alpha = 0.05$. Without multiple-testing correction, how many false positives will they accumulate per year on average, and what does this tell you about uncritically scaling classical inference to industrial volume?

??? success "Solution to Exercise 6"
    Under the null hypothesis of no effect, each test yields a false positive with probability $\alpha = 0.05$. Across 1,000 tests, the expected number of false positives is

    $$
    \mathbb{E}[\text{false positives}] = 1{,}000 \times 0.05 = 50
    $$

    Roughly 50 of 1,000 "winning" tests will have been won by chance. If the startup ships every "significant" change, dozens of useless features land in production each year, occasionally on top of each other — leading to drift, regression, and inscrutable behavior.

    Remedies include **family-wise error control** (Bonferroni: divide $\alpha$ by the number of tests), **false-discovery-rate control** (Benjamini–Hochberg: control the fraction of declared positives that are false), **stricter thresholds** as standard practice (e.g., $\alpha = 0.005$ for confirmatory tests), and **prior judgment**: low-prior changes (small effects, low business value) should require higher evidence than high-prior changes.

    The deeper lesson: classical methods were designed for a small number of pre-specified questions, not for industrial-scale interrogation of the same data set. Naive application at scale silently inflates error rates.
