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

## Exercises

**Exercise 1.**
Explain why random assignment in an experiment eliminates confounding, even from unobserved variables.

??? success "Solution to Exercise 1"
    Random assignment ensures that the treatment group and control group are, on average, identical in all characteristics -- both observed and unobserved -- before the treatment is applied. Because assignment is determined by a random mechanism (coin flip, random number generator), it is independent of all pre-treatment variables.

    Any confounder $U$ (observed or not) satisfies $U \perp T$ where $T$ is the treatment indicator. This means $E[Y \mid T=1] - E[Y \mid T=0] = E[Y(1)] - E[Y(0)]$, the true average treatment effect, because there are no back-door paths from $T$ to $Y$ (the randomization "cuts" all such paths).

    With large enough sample sizes, the law of large numbers ensures the groups are balanced on every variable. With small samples, imbalances can occur by chance, which is why we still use hypothesis tests and confidence intervals.

---

**Exercise 2.**
A randomized experiment finds that a tutoring program increases test scores by 8 points ($p = 0.02$). A colleague argues this does not prove causation because "correlation does not imply causation." Is the colleague correct?

??? success "Solution to Exercise 2"
    The colleague is **incorrect** in this context. The statement "correlation does not imply causation" applies to observational studies where confounders may explain the association. In a properly randomized experiment, the causal interpretation is valid because:

    1. Random assignment eliminates confounding.
    2. The researcher controls the timing (treatment before outcome), establishing temporal precedence.
    3. The comparison group (control) provides the counterfactual.

    The 8-point increase is a valid estimate of the causal effect, subject to the usual caveats of statistical inference (sampling variability, as reflected by $p = 0.02$). The main threats to causal inference in experiments are practical issues (non-compliance, attrition, spillover effects), not confounding.

---

**Exercise 3.**
Define the concepts of internal validity and external validity for a randomized experiment. Give an example where an experiment has high internal validity but questionable external validity.

??? success "Solution to Exercise 3"
    **Internal validity:** The extent to which the experiment correctly measures the causal effect within the study. Requires proper randomization, no attrition bias, no spillover effects, and adherence to the treatment protocol.

    **External validity:** The extent to which the results generalize to other populations, settings, or time periods.

    **Example:** A randomized trial of a math app conducted in a single wealthy suburban school district shows a 12-point improvement in test scores (high internal validity -- proper randomization, low attrition). However, the results may not generalize to under-resourced urban schools where students have less internet access, different baseline skills, and different motivational profiles (questionable external validity).

---

**Exercise 4.**
In the potential outcomes framework, define the Average Treatment Effect (ATE) and explain the "fundamental problem of causal inference."

??? success "Solution to Exercise 4"
    For individual $i$, let $Y_i(1)$ be the potential outcome under treatment and $Y_i(0)$ be the potential outcome under control. The individual treatment effect is $\tau_i = Y_i(1) - Y_i(0)$.

    The **Average Treatment Effect** is:

    $$
    \text{ATE} = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]
    $$

    The **fundamental problem of causal inference** is that for each individual, we observe only one potential outcome: either $Y_i(1)$ (if treated) or $Y_i(0)$ (if not treated), never both. The individual causal effect $\tau_i$ is therefore unobservable.

    Randomization solves this at the group level: because $T \perp (Y(1), Y(0))$, we have $E[Y \mid T=1] = E[Y(1)]$ and $E[Y \mid T=0] = E[Y(0)]$, so the difference in group means estimates the ATE even though individual effects remain unknown.
