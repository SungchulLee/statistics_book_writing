# Randomized Experiments and Causation

Observational data can establish correlation but cannot, by itself, establish causation because of the ever-present threat of confounding. **Randomized experiments** solve this problem by using random assignment to break the link between the treatment and all potential confounders -- both measured and unmeasured. This section explains why randomization is the gold standard for causal inference and how randomized controlled trials (RCTs) are designed and analyzed.

---

## Why Randomization Enables Causal Claims

In an observational study, individuals who receive a treatment may differ systematically from those who do not. These preexisting differences (confounders) make it impossible to attribute differences in outcomes to the treatment alone.

Randomization solves this by assigning treatment **by chance**, ensuring that, on average, the treatment and control groups are comparable on every characteristic -- observed and unobserved. Formally, randomization guarantees the **ignorability condition**:

$$
Y(0), Y(1) \perp\!\!\!\perp X
$$

where $Y(0)$ and $Y(1)$ are the potential outcomes and $X$ is the treatment indicator. Under this condition, the simple difference in group means is an unbiased estimate of the **average treatment effect** (ATE):

$$
\text{ATE} = \mathbb{E}[Y \mid X = 1] - \mathbb{E}[Y \mid X = 0]
$$

No adjustment for confounders is needed because randomization has eliminated confounding by design.

---

## Structure of a Randomized Controlled Trial

A typical RCT consists of the following elements:

1. **Define the population.** Specify the target population and eligibility criteria.
2. **Recruit participants.** Obtain informed consent and enroll eligible individuals.
3. **Random assignment.** Use a random mechanism (e.g., coin flip, random number generator) to assign each participant to treatment or control.
4. **Administer the intervention.** The treatment group receives the intervention; the control group receives a placebo or standard care.
5. **Measure outcomes.** Record the response variable for all participants.
6. **Analyze.** Compare outcomes between groups, typically using a two-sample t-test, permutation test, or regression.

---

## Key Design Features

### Blinding

**Blinding** prevents knowledge of group assignment from influencing the results:

- **Single-blind**: participants do not know their assignment.
- **Double-blind**: neither participants nor the researchers who measure outcomes know the assignment.
- **Triple-blind**: participants, outcome assessors, and data analysts are all blinded.

Blinding reduces **placebo effects** (participants behave differently because they believe they are treated) and **observer bias** (researchers measure or record outcomes differently based on group knowledge).

### Placebo Control

A **placebo** is an inert treatment that mimics the intervention in every way except the active ingredient. Comparing to a placebo, rather than no treatment, isolates the effect of the active ingredient from the effect of receiving any treatment at all.

### Randomization Methods

- **Simple randomization**: each participant is independently assigned with a fixed probability (e.g., 0.5 for each group). Simple but can produce unbalanced groups in small trials.
- **Block randomization**: participants are randomized in blocks (e.g., blocks of 4) to ensure roughly equal group sizes.
- **Stratified randomization**: randomization is performed separately within strata defined by important covariates (e.g., age, sex), ensuring balance on those variables.

---

## The Logic of Causal Inference from RCTs

The reasoning proceeds as follows:

1. Before randomization, the treatment and control groups are (in expectation) identical on all characteristics.
2. The only systematic difference between the groups is the treatment itself.
3. Any difference in outcomes must therefore be caused by the treatment.

This logic depends on several assumptions:

- **No interference**: one participant's treatment does not affect another's outcome (the Stable Unit Treatment Value Assumption, or SUTVA).
- **Compliance**: participants actually receive the treatment they were assigned.
- **No attrition**: participants do not drop out differentially between groups.

Violations of these assumptions weaken the causal interpretation. Intention-to-treat (ITT) analysis preserves the benefits of randomization even when compliance is imperfect, by analyzing participants according to their assigned group regardless of actual treatment received.

---

## Example: Drug vs Placebo Trial

A pharmaceutical company tests a new blood pressure medication.

- **Population**: adults aged 40-65 with mild hypertension.
- **Design**: 200 participants randomized 1:1 to drug or placebo.
- **Outcome**: change in systolic blood pressure after 12 weeks.
- **Analysis**: two-sample t-test comparing mean blood pressure change.

| Group | $n$ | Mean change (mmHg) | SD |
|:---|:---:|:---:|:---:|
| Drug | 100 | $-12.3$ | 8.5 |
| Placebo | 100 | $-3.1$ | 7.9 |

The estimated ATE is $-12.3 - (-3.1) = -9.2$ mmHg. Because assignment was randomized, this difference can be attributed to the drug rather than to confounders like age, diet, or exercise habits.

---

## Limitations of Randomized Experiments

Despite their strengths, RCTs have important limitations:

1. **Ethical constraints.** Some treatments cannot be randomly assigned (e.g., smoking, poverty). It would be unethical to assign people to harmful exposures.

2. **External validity.** The population enrolled in an RCT may not represent the broader population of interest. Strict eligibility criteria can limit generalizability.

3. **Cost and feasibility.** Large, well-designed RCTs are expensive and time-consuming.

4. **Compliance issues.** Participants may not adhere to their assigned treatment, diluting the treatment effect.

5. **Hawthorne effect.** Being in a study may change participants' behavior regardless of the treatment.

When RCTs are not feasible, researchers turn to observational methods with careful adjustment for confounders, natural experiments, or [instrumental variables](instrumental_variables.md).

---

## Natural Experiments

A **natural experiment** occurs when some external event or policy creates variation in the treatment that is as-if random, even though no researcher performed the randomization. Examples include:

- **Draft lotteries**: random assignment of birth dates to military service.
- **Geographic boundaries**: students on opposite sides of a school district line receive different educational interventions.
- **Policy changes**: a new regulation applies to one group but not another based on an arbitrary threshold.

Natural experiments approximate the logic of RCTs by exploiting quasi-random variation. The causal inference is valid to the extent that the variation is truly exogenous (not driven by individual choices).

---

## Summary

Randomized experiments are the gold standard for establishing causal relationships because random assignment eliminates confounding, both measured and unmeasured. The key design elements -- randomization, blinding, and placebo control -- ensure that the only systematic difference between groups is the treatment itself. When RCTs are infeasible due to ethical, practical, or cost constraints, natural experiments and quasi-experimental designs can approximate the benefits of randomization. Understanding why randomization enables causal inference clarifies both its power and the limitations of observational studies.
