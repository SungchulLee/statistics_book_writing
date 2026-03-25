# Lurking Variables and Common Causes

A **lurking variable** is a variable that is not included in the analysis but has a substantial effect on the relationship between the variables under study. Unlike a confounder that has been identified and measured, a lurking variable operates behind the scenes -- the researcher may not even be aware of its existence. The most common form of lurking variable is a **common cause** that simultaneously drives both the observed exposure and outcome, creating an association that can be misinterpreted as causal.

---

## What Is a Lurking Variable

A lurking variable is any variable that:

1. Is **not measured** or **not included** in the current analysis.
2. Is associated with both the explanatory variable $X$ and the response variable $Y$.
3. Could explain part or all of the observed association between $X$ and $Y$.

The distinction from a confounder is practical, not conceptual. A confounder is a lurking variable that has been identified and measured. A lurking variable is one that remains hidden. The statistical consequence is the same: the observed $X$-$Y$ relationship is distorted.

---

## The Common Cause Structure

The most important type of lurking variable is a **common cause** -- a variable $Z$ that causally affects both $X$ and $Y$. The causal structure is:

$$
X \leftarrow Z \rightarrow Y
$$

Because $Z$ causes variation in both $X$ and $Y$, the two variables will be correlated even if there is no direct causal link between them. The observed correlation between $X$ and $Y$ is an artifact of their shared dependence on $Z$.

??? example "Fire damage and firefighters"
    The number of firefighters deployed ($X$) is positively correlated with the amount of property damage ($Y$). The lurking variable is the severity of the fire ($Z$). Larger fires cause both more firefighters to be deployed and more damage. Sending more firefighters does not cause more damage -- if anything, it reduces it.

---

## Why Lurking Variables Are Dangerous

Lurking variables are particularly problematic because:

1. **They are invisible in the data.** Since the variable is not measured, there is no column in the dataset to examine or control for.

2. **They produce real correlations.** The association between $X$ and $Y$ is statistically genuine -- it exists in the data. The problem is not that the correlation is wrong but that its causal interpretation is wrong.

3. **They cannot be controlled statistically.** You cannot include a variable in a regression if it has not been measured. Partial correlation, stratification, and regression adjustment all require the variable to be observed.

4. **They can create or mask effects.** A lurking common cause can make two unrelated variables appear related (positive confounding) or make two genuinely related variables appear unrelated (negative confounding).

---

## Examples of Lurking Variables

### Storks and Birth Rates

Across European countries, the number of stork pairs ($X$) is positively correlated with birth rates ($Y$). The lurking variable is rurality ($Z$): rural areas have more storks and higher birth rates. Storks do not deliver babies.

### Spending and Academic Performance

At the school level, per-pupil spending ($X$) is sometimes negatively correlated with test scores ($Y$). A lurking variable is socioeconomic status ($Z$): wealthier districts may spend less per pupil (due to efficient administration) while having higher test scores (due to home environment advantages). The relationship between spending and performance, after controlling for socioeconomic factors, may be positive.

### Organic Food and Autism

Sales of organic food ($X$) are correlated with autism diagnosis rates ($Y$) over time. The lurking variable is time itself ($Z$): both have increased over the past two decades for entirely unrelated reasons (health trends vs. diagnostic criteria changes).

---

## How to Guard Against Lurking Variables

Since lurking variables are by definition unmeasured, they cannot be controlled directly. However, several strategies reduce their impact:

1. **Subject-matter knowledge.** Think carefully about what variables might influence both $X$ and $Y$. Draw a causal diagram and ask: are there arrows that I am missing?

2. **Randomized experiments.** Random assignment of $X$ breaks the association between $X$ and any lurking variable $Z$, whether measured or not. This is the primary advantage of experiments over observational studies. See [Experiments and Causation](../causation/experiments_causation.md).

3. **Sensitivity analysis.** Ask: how strong would an unmeasured confounder need to be to explain the observed association? If the association is robust to plausible confounding, the conclusion is more credible.

4. **Measure more variables.** The more potential confounders that are measured and controlled, the less room there is for unmeasured lurking variables to drive the results.

5. **Replication across settings.** If the same association appears across different populations, time periods, and contexts -- where the lurking variables would differ -- the association is more likely genuine.

---

## Lurking Variables vs Confounders vs Mediators

| Concept | Measured? | Causal structure | Effect on $X$-$Y$ association |
|:---|:---:|:---:|:---|
| Lurking variable | No | $X \leftarrow Z \rightarrow Y$ | Distorts the association (direction unknown) |
| Confounder | Yes | $X \leftarrow Z \rightarrow Y$ | Can be controlled for |
| Mediator | Yes | $X \rightarrow Z \rightarrow Y$ | Part of the causal pathway |

A lurking variable becomes a confounder once it is identified and measured. The challenge is recognizing that it exists in the first place.

---

## Connection to Other Sections

The concept of lurking variables motivates several important topics in this chapter:

- [Confounding variables](confounding_variables.md) are lurking variables that have been identified.
- [Spurious correlations](spurious_correlations.md) are the observable consequence of lurking common causes.
- [Directed acyclic graphs](../causation/dags.md) provide a formal framework for reasoning about lurking variables and their effects.
- [Randomized experiments](../causation/experiments_causation.md) are the gold standard for eliminating the influence of lurking variables.

---

## Summary

A lurking variable is an unmeasured variable that affects both the exposure and the outcome, creating a distorted association between them. The most common form is a common cause that simultaneously drives both observed variables. Because lurking variables are unmeasured, they cannot be controlled for statistically. Randomized experiments, careful subject-matter reasoning, sensitivity analysis, and replication across diverse settings are the primary defenses against lurking variable bias.
