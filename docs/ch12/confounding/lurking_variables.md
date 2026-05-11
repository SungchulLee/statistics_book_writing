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

## Exercises

**Exercise 1.**
Ice cream sales and drowning deaths are positively correlated. Identify the lurking variable and explain the causal structure.

??? success "Solution to Exercise 1"
    The lurking variable is **temperature (season/weather)**. Hot weather causes both increased ice cream consumption and increased swimming activity, which leads to more drownings. The causal structure is:

    Temperature $\to$ Ice cream sales, Temperature $\to$ Drownings.

    There is no causal link from ice cream to drownings. The observed positive correlation is entirely spurious, driven by the common cause. Conditioning on temperature (or season) would eliminate the association between ice cream sales and drowning deaths.

---

**Exercise 2.**
A study finds a strong positive correlation between the number of firefighters at a fire and the amount of damage caused. Should the city reduce the number of firefighters sent to fires?

??? success "Solution to Exercise 2"
    No. The lurking variable is **fire severity**. Larger, more intense fires cause both more damage and the dispatch of more firefighters. The causal structure is:

    Fire severity $\to$ Number of firefighters, Fire severity $\to$ Damage.

    The correlation between firefighters and damage is not causal -- it is confounded by fire severity. Reducing the number of firefighters would likely increase damage, not decrease it. The correct analysis would condition on fire severity (e.g., compare damage for fires of similar size with different numbers of firefighters dispatched).

---

**Exercise 3.**
Define Simpson's paradox and give a concrete example where the direction of an association reverses after conditioning on a lurking variable.

??? success "Solution to Exercise 3"
    **Simpson's paradox** occurs when the direction of an association between two variables reverses or disappears after conditioning on a third variable (a confounder).

    **Example:** A hospital reports that Treatment A has a higher overall survival rate than Treatment B. However, when patients are stratified by disease severity:

    - Among mild cases: Treatment B has higher survival.
    - Among severe cases: Treatment B has higher survival.

    The paradox arises because Treatment A is disproportionately given to mild cases (who have high survival regardless), while Treatment B is given to severe cases. Disease severity is the lurking variable that confounds the comparison. The correct conclusion (from the stratified analysis) is that Treatment B is superior.

---

**Exercise 4.**
A regression of salary on years of experience shows a positive coefficient. When "department" is added as a control variable, the coefficient for experience becomes negative. Explain how this is possible and which result is more trustworthy.

??? success "Solution to Exercise 4"
    This reversal can occur if **department** is a lurking variable that is positively correlated with both experience and salary:

    - Experienced employees tend to be in lower-paying departments (perhaps they entered the company when those departments were growing).
    - Within any given department, more experience is associated with lower salary (perhaps newer hires are paid market rates that have increased over time, i.e., salary compression).

    Without controlling for department, the positive cross-department variation masks the negative within-department pattern.

    Which result is more trustworthy depends on the causal question. If department is a confounder (causes both experience and salary), conditioning on it gives a better estimate of the within-department return to experience. However, if department is a mediator (experience causes people to move to certain departments), conditioning on it removes part of the causal effect, and the unconditional estimate may be more appropriate for the total effect.
