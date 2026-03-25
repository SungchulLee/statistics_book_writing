# Confounding Variables

A **confounding variable** is a variable that influences both the exposure (or predictor) and the outcome, creating a spurious association between them. Confounding is one of the most important concepts in statistics and epidemiology because it can make a non-causal relationship appear causal, or mask a genuine causal effect. Properly identifying and controlling for confounders is essential for drawing valid conclusions from observational data.

---

## Definition

A variable $Z$ is a **confounder** of the relationship between $X$ (exposure) and $Y$ (outcome) if all three of the following conditions hold:

1. $Z$ is associated with $X$ (the exposure).
2. $Z$ is associated with $Y$ (the outcome), conditional on $X$.
3. $Z$ is not on the causal pathway from $X$ to $Y$ (i.e., $Z$ is not a mediator).

When a confounder is present, the observed association between $X$ and $Y$ reflects both the direct relationship (if any) and the indirect path through $Z$. Failure to account for $Z$ leads to a **confounded** estimate of the $X$-$Y$ relationship.

---

## The Confounding Structure

The causal structure underlying confounding can be represented as a diagram where $Z$ has arrows pointing to both $X$ and $Y$:

$$
X \leftarrow Z \rightarrow Y
$$

This "common cause" structure means that $X$ and $Y$ are associated even if there is no direct causal link between them. The association arises entirely through the backdoor path $X \leftarrow Z \rightarrow Y$.

??? example "Classic example: coffee and cancer"
    Early observational studies found a positive association between coffee consumption ($X$) and lung cancer ($Y$). However, coffee drinkers were more likely to smoke ($Z$). Smoking is associated with both coffee consumption and lung cancer. After controlling for smoking, the coffee-cancer association largely disappeared. Smoking was the confounder.

---

## Positive and Negative Confounding

Confounding can either inflate or mask the true effect:

- **Positive confounding**: the confounder creates or inflates a positive association. The crude association overestimates the true effect.
- **Negative confounding**: the confounder masks or reduces the true association. The crude association underestimates the true effect.

The direction of confounding depends on the signs of the relationships between $Z$ and $X$, and between $Z$ and $Y$.

---

## How to Detect Confounding

In practice, confounding is detected by comparing the crude (unadjusted) association with the adjusted association:

1. Compute the crude association between $X$ and $Y$ (ignoring $Z$).
2. Compute the adjusted association, controlling for $Z$.
3. If the two estimates differ meaningfully (typically by more than 10%), $Z$ is a confounder.

This is sometimes called the "change-in-estimate" criterion. It is a practical heuristic, not a formal statistical test.

---

## Methods for Controlling Confounders

### At the Design Stage

1. **Randomization.** Random assignment of the exposure breaks the association between $X$ and $Z$, eliminating confounding. This is why randomized controlled trials are the gold standard for causal inference. See [Experiments and Causation](../causation/experiments_causation.md).

2. **Restriction.** Limit the study to individuals with the same value of $Z$. For example, study only non-smokers to eliminate smoking as a confounder.

3. **Matching.** For each exposed individual, select an unexposed individual with the same (or similar) value of $Z$.

### At the Analysis Stage

4. **Stratification.** Analyze the data separately within each level of $Z$, then combine using methods like the Mantel-Haenszel estimator.

5. **Regression adjustment.** Include $Z$ as a covariate in a regression model for $Y$ on $X$. The coefficient of $X$ then estimates the effect of $X$ holding $Z$ constant.

6. **Propensity score methods.** Estimate the probability of exposure given $Z$, then use this propensity score for matching, stratification, or weighting.

---

## Example: Exercise, Diet, and Heart Disease

Suppose we observe that people who exercise regularly ($X = 1$) have lower rates of heart disease ($Y$). However, regular exercisers also tend to eat healthier diets ($Z$). Diet is a confounder because:

1. Diet is associated with exercise ($r_{XZ} > 0$).
2. Diet is associated with heart disease, given exercise status ($r_{YZ} \neq 0$).
3. Diet is not on the causal pathway from exercise to heart disease (exercise does not cause diet in this context).

If we fail to control for diet, the observed benefit of exercise may be partly attributable to the healthier diets of exercisers rather than exercise itself.

After adjusting for diet:

- If the exercise-heart disease association **persists**, exercise has an independent protective effect.
- If the association **disappears**, the apparent benefit was driven by diet (confounding).
- If the association **weakens but remains**, exercise has an independent effect, but its crude magnitude was confounded by diet.

---

## Confounding vs Mediation vs Collider Bias

It is crucial to distinguish confounding from two related but different phenomena:

| Structure | Diagram | Effect of controlling for $Z$ |
|:---|:---:|:---|
| Confounding | $X \leftarrow Z \rightarrow Y$ | Removes bias; reveals true $X \to Y$ effect |
| Mediation | $X \rightarrow Z \rightarrow Y$ | Removes the indirect effect; shows only direct effect |
| Collider | $X \rightarrow Z \leftarrow Y$ | *Introduces* bias; creates spurious association |

Controlling for a mediator is not necessarily wrong, but it answers a different question (direct effect vs total effect). Controlling for a collider is always harmful -- it creates bias where none existed. These distinctions are formalized through [directed acyclic graphs](../causation/dags.md).

---

## Summary

A confounding variable is a common cause of both the exposure and the outcome that distorts the observed association between them. Confounding can inflate, reduce, or reverse the true effect. It can be addressed through study design (randomization, restriction, matching) or analysis (stratification, regression, propensity scores). The key to handling confounding correctly is understanding the underlying causal structure, which distinguishes confounders from mediators and colliders.
