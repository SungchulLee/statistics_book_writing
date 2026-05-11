# Design Your Data Collection (Classical Approach)

## Overview

The **classical approach** to data analysis begins with a clear research question and then designs a data collection strategy tailored to answer it. The data does not yet exist when the study is planned—the researcher controls *how* it is gathered, *from whom*, and *under what conditions*. This deliberate design is the hallmark of traditional statistical practice.

## Core Principle

> **Design first, collect second, analyze third.**

The classical approach treats data collection as a **designed experiment** or a **carefully structured survey**. By controlling the data-generation process, the researcher can make strong claims about causality, quantify uncertainty precisely, and minimize bias.

## Three Classical Study Types

### 1. Observational Studies

The researcher observes and records data **without intervention**. Useful when manipulation is impractical or unethical, but limited to identifying associations rather than causal relationships.

### 2. Controlled Experiments

The researcher **manipulates** one or more variables and **randomly assigns** subjects to groups. The gold standard for establishing causality because randomization balances known and unknown confounders.

### 3. Sample Surveys

The researcher selects a **representative sample** from a population and collects data through structured questionnaires or interviews. Enables inference about the population when studying every member is infeasible.

## Strengths of the Classical Approach

- **Causal inference**: Randomized experiments can establish cause-and-effect relationships.
- **Known uncertainty**: Because the sampling mechanism is designed, standard errors, confidence intervals, and p-values have clear probabilistic interpretations.
- **Bias control**: Random sampling and random assignment directly address selection bias and confounding.
- **Reproducibility**: A well-documented design can be replicated by other researchers.

## When This Approach Works Best

- The research question is specific and well-defined.
- It is feasible to design and execute a study (time, budget, ethics).
- The population of interest is accessible for sampling or experimentation.
- Causal claims are needed (e.g., clinical trials, A/B tests, policy evaluations).

## Limitations

- **Cost and time**: Designing and running experiments or large-scale surveys is expensive and slow.
- **Ethical constraints**: Many important questions cannot be studied experimentally (e.g., the effect of poverty on health).
- **Scope**: The classical approach works best for structured, well-defined problems; it is less suited to open-ended exploration of massive, unstructured datasets.
- **Generalizability**: Laboratory experiments may not reflect real-world conditions.

## Key Takeaways

- The classical approach prioritizes **design** to ensure that the data can answer the research question with known precision.
- Its greatest strength is the ability to make **causal** and **inferential** claims with well-quantified uncertainty.
- It remains indispensable in medicine, social science, and policy evaluation, where the stakes of incorrect conclusions are high.

## Exercises

**Exercise 1.**
A researcher wants to study whether a new teaching method improves test scores. Describe how to design a randomized experiment for this question, specifying the treatment, control, randomization unit, and primary outcome.

??? success "Solution to Exercise 1"

    - **Treatment:** The new teaching method (e.g., flipped classroom) applied for one semester.
    - **Control:** The standard teaching method (business as usual) for the same course and semester.
    - **Randomization unit:** Students (or classroom sections, if individual randomization is impractical). Random assignment ensures that treatment and control groups are comparable in ability, motivation, and background.
    - **Primary outcome:** Final exam score (pre-specified before the experiment begins).
    - **Sample size:** Determined by a power analysis based on the minimum detectable effect size.
    - **Blinding:** The grader should be blinded to which group each student belongs.

---

**Exercise 2.**
Explain why stratified random sampling can be more efficient than simple random sampling. Give an example where stratification substantially reduces sampling variability.

??? success "Solution to Exercise 2"
    Stratified random sampling divides the population into homogeneous strata and samples independently within each stratum. It is more efficient because within-stratum variability is smaller than overall variability, so each stratum's mean is estimated more precisely.

    **Example:** Estimating average household income in a city with two neighborhoods: one wealthy (mean \$200K, small variance) and one modest (mean \$40K, small variance). A simple random sample might over- or under-represent one neighborhood by chance, producing high variability. Stratifying by neighborhood and sampling proportionally from each guarantees both are represented, reducing the variance of the overall mean estimate. The gain is largest when the strata means are very different but within-stratum variability is small.

---

**Exercise 3.**
A company plans to survey customer satisfaction. They have a budget for 500 surveys. Compare the trade-offs between a census of 500 randomly chosen customers, a convenience sample of 500 customers at one store, and a quota sample of 500 customers balanced by age group.

??? success "Solution to Exercise 3"
    | Method | Representativeness | Cost | Bias risk |
    |---|---|---|---|
    | **Random sample** (500 from full customer list) | High -- every customer has a known selection probability | Medium (requires full customer list, may need follow-up for non-response) | Low if response rate is high |
    | **Convenience sample** (500 at one store) | Low -- only captures customers at that location and time | Low (easy to administer) | High -- excludes online customers, other locations, and different shopping times |
    | **Quota sample** (500, balanced by age) | Medium -- ensures age representation but not other factors | Medium | Medium -- balances on age but may still be biased on income, geography, or other uncontrolled factors |

    The random sample is preferred for valid inference; the convenience sample is cheapest but most biased; the quota sample is a compromise that controls one source of imbalance.

---

**Exercise 4.**
Define **power** in the context of experiment design. If a study has 80% power to detect a 5-point difference, what does this mean in practical terms?

??? success "Solution to Exercise 4"
    **Power** is the probability of correctly rejecting the null hypothesis when the alternative is true, i.e., $\text{Power} = 1 - \beta$ where $\beta$ is the Type II error rate (probability of a false negative).

    If a study has 80% power to detect a 5-point difference, this means: *if the true treatment effect is exactly 5 points, the study has an 80% probability of producing a statistically significant result* (rejecting $H_0$ at the chosen significance level). Equivalently, there is a 20% chance of failing to detect a real 5-point effect.

    The 80% threshold is conventional. Power depends on three factors: the significance level $\alpha$, the effect size, and the sample size $n$. Increasing $n$ or $\alpha$ increases power; detecting smaller effects requires larger samples.

---

**Exercise 5.**
A researcher proposes a study with $n = 50$ per arm but has not performed a power analysis. The minimum effect size considered meaningful is 0.3 standard deviations. Compute the power at $\alpha = 0.05$ (two-sided $t$-test) and decide whether the study is adequately powered.

??? success "Solution to Exercise 5"
    For a two-sample $t$-test with equal $n$ per arm, effect size $d$ (in standard deviations), and significance $\alpha$, the power is approximately

    $$
    \mathrm{Power} = P\!\left(|Z| > z_{1 - \alpha/2} - d\sqrt{n/2}\right)
    $$

    With $d = 0.3$, $n = 50$, $\alpha = 0.05$, $z_{0.975} = 1.96$:

    $$
    d\sqrt{n/2} = 0.3 \times 5 = 1.5
    $$

    so the non-centrality parameter is 1.5. Power $\approx P(Z > 1.96 - 1.5) + P(Z < -1.96 - 1.5) = P(Z > 0.46) + P(Z < -3.46) \approx 0.323 + 0.0003 \approx 0.32$.

    Only 32% power — the study is severely underpowered. It would need roughly $n \approx 175$ per arm to reach 80% power at $d = 0.3$. Running an underpowered study risks both Type II errors (missing a real effect) and the **winner's curse** (the few significant findings will systematically overestimate effect sizes).

---

**Exercise 6.**
**Preregistration** asks researchers to commit to their hypothesis and analysis plan before collecting data. Explain how preregistration addresses the *garden of forking paths* problem, and one practical concern about strict preregistration.

??? success "Solution to Exercise 6"
    The **garden of forking paths** (Gelman & Loken, 2013) refers to researchers' many discretionary choices — variable definitions, exclusion criteria, transformation, subgroup analyses — that, made after seeing the data, can produce one "significant" result among the many implicit comparisons. Even with no deliberate p-hacking, this inflates Type I error rates.

    **Preregistration** locks in the choices before data is seen. The analysis plan specifies the primary hypothesis, the exact statistical test, the population, exclusions, and how multiple comparisons (if any) will be handled. Any departures must be reported as exploratory rather than confirmatory.

    **Practical concern:** strict preregistration discourages legitimate post-hoc discoveries. If the data show a clear and unexpected pattern not anticipated by the registered plan, the researcher must report this as exploratory — which often means the publication system treats it less favorably. The solution is a hybrid: preregister the confirmatory hypothesis with a strict significance threshold, then report exploratory analyses transparently as exploratory. Journals adopting **registered reports** (peer-review accepts the design *before* data collection) are removing the publication incentive against null results, further reducing publication bias.
