# Criteria for Causal Inference

Establishing that a correlation reflects a genuine causal relationship is one of the central challenges in science and statistics. Since correlation alone cannot distinguish causation from confounding or coincidence, researchers have developed frameworks of **criteria** that, taken together, provide evidence for or against a causal interpretation. Two influential frameworks are **Bradford Hill's criteria** from epidemiology and the **counterfactual (potential outcomes) framework** from statistics.

---

## Bradford Hill's Criteria

In 1965, the epidemiologist Sir Austin Bradford Hill proposed nine criteria for evaluating whether an observed association is causal. These criteria are guidelines for judgment, not a formal statistical test. No single criterion is necessary or sufficient, but the more criteria that are satisfied, the stronger the case for causation.

### 1. Strength of Association

Stronger associations are more likely to be causal. A relative risk of 10 is harder to explain by confounding than a relative risk of 1.2. However, weak associations can still be causal (e.g., passive smoking and lung cancer), and strong associations can still be confounded.

### 2. Consistency

The association is observed repeatedly across different populations, settings, and study designs. Replication reduces the likelihood that the finding is due to a study-specific bias or confounding structure.

### 3. Specificity

The exposure leads to a specific outcome (and the outcome is primarily caused by the exposure). This criterion is the weakest of the nine because many causal relationships are not specific -- smoking causes multiple diseases, not just lung cancer.

### 4. Temporality

The cause must precede the effect in time. This is the only criterion that is **necessary** for causation. If $X$ does not occur before $Y$, $X$ cannot cause $Y$.

!!! note "Temporality is necessary but not sufficient"
    Establishing that $X$ precedes $Y$ rules out reverse causation but does not rule out confounding by a lurking variable that precedes both $X$ and $Y$.

### 5. Biological Gradient (Dose-Response)

Greater exposure leads to a greater response. A dose-response relationship strengthens the causal argument because it is difficult (though not impossible) for confounding to produce a smooth gradient.

### 6. Plausibility

There is a biologically or mechanistically plausible explanation for how $X$ could cause $Y$. This criterion depends on current scientific knowledge and is therefore the most subjective.

### 7. Coherence

The causal interpretation does not conflict with the known natural history and biology of the disease. Coherence is related to plausibility but focuses on consistency with the broader body of knowledge.

### 8. Experiment

Experimental evidence (e.g., from randomized controlled trials or natural experiments) supports the causal claim. Experimental manipulation that changes $X$ and produces a change in $Y$ is strong evidence for causation.

### 9. Analogy

Similar causes produce similar effects. If one chemical is known to cause cancer, a structurally similar chemical might also cause cancer. This is the weakest criterion and is used mainly to support plausibility.

---

## The Counterfactual Framework

The **counterfactual** (or **potential outcomes**) framework, developed by Jerzy Neyman and Donald Rubin, provides a formal mathematical definition of causal effects. It is the foundation of modern causal inference in statistics and econometrics.

### Potential Outcomes

For each individual $i$, define two potential outcomes:

- $Y_i(1)$: the outcome that would occur if $i$ receives treatment ($X = 1$).
- $Y_i(0)$: the outcome that would occur if $i$ does not receive treatment ($X = 0$).

The **individual causal effect** is

$$
\tau_i = Y_i(1) - Y_i(0)
$$

The **fundamental problem of causal inference** is that we can never observe both $Y_i(1)$ and $Y_i(0)$ for the same individual. We observe $Y_i(1)$ if the individual is treated and $Y_i(0)$ if not, but never both.

### Average Treatment Effect

Since individual causal effects are unobservable, causal inference focuses on average effects across a population:

$$
\text{ATE} = \mathbb{E}[Y(1) - Y(0)] = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]
$$

Under random assignment, treated and untreated groups have the same distribution of potential outcomes, so

$$
\text{ATE} = \mathbb{E}[Y \mid X = 1] - \mathbb{E}[Y \mid X = 0]
$$

Without randomization, this equality fails because of **selection bias**: individuals who choose (or are assigned to) treatment may differ systematically from those who do not.

### Ignorability (Unconfoundedness)

The key assumption that enables causal inference from observational data is **ignorability** (also called unconfoundedness):

$$
Y(0), Y(1) \perp\!\!\!\perp X \mid Z
$$

This states that, conditional on observed covariates $Z$, treatment assignment is independent of potential outcomes. Under this assumption, adjusting for $Z$ (via regression, matching, or propensity scores) identifies the causal effect.

---

## Comparing the Two Frameworks

| Feature | Bradford Hill | Counterfactual |
|:---|:---|:---|
| Origin | Epidemiology (1965) | Statistics (Neyman 1923, Rubin 1974) |
| Nature | Guidelines for judgment | Formal mathematical framework |
| Requires experiment? | No (but experiment is one criterion) | No (but identifies when observation suffices) |
| Strength | Broad applicability, intuitive | Precise definitions, testable assumptions |
| Limitation | Subjective, no formal decision rule | Requires strong assumptions (ignorability) |

The two frameworks are complementary. Bradford Hill's criteria provide a qualitative checklist for evaluating evidence, while the counterfactual framework provides a rigorous foundation for defining and estimating causal effects.

---

## Practical Guidelines

When evaluating whether an observed correlation is causal:

1. **Start with temporality.** If the cause does not precede the effect, causation is ruled out.
2. **Look for a dose-response relationship.** A gradient strengthens the causal case.
3. **Consider confounders.** Can [confounding variables](../confounding/confounding_variables.md) or [lurking variables](../confounding/lurking_variables.md) explain the association?
4. **Seek experimental evidence.** Randomized experiments provide the strongest evidence for causation. See [Experiments and Causation](experiments_causation.md).
5. **Check replication.** Does the finding hold across different studies and populations?
6. **Evaluate plausibility.** Is there a credible mechanism?
7. **Apply the counterfactual test.** Can you articulate what would have happened in the absence of the exposure?

---

## Summary

Establishing causation from observational data requires evidence beyond correlation. Bradford Hill's nine criteria provide a qualitative framework for evaluating causal claims, with temporality being the only strictly necessary criterion. The counterfactual framework provides a formal definition of causal effects through potential outcomes and identifies the assumptions (particularly ignorability) needed to estimate causal effects from non-experimental data. Together, these frameworks guide researchers from observed associations to justified causal conclusions.

## Exercises

**Exercise 1.**
For each of the following claims, evaluate which of the five criteria for causation (temporal precedence, covariation, elimination of confounders, plausibility, experimental evidence) are met:

1. "Smoking causes lung cancer"
2. "Wearing a seatbelt prevents death in car accidents"
3. "Eating organic food causes better health"
4. "Social media use causes depression in teenagers"

??? success "Solution to Exercise 1"

    1. **Smoking causes lung cancer**: All five criteria are strongly met. Temporal precedence (smoking precedes cancer by years), covariation (dose-response relationship), elimination of confounders (extensive studies controlling for other risk factors), plausibility (carcinogens in smoke damage DNA), and experimental evidence (animal studies; human evidence is quasi-experimental due to ethical constraints, but prospective cohort studies provide strong support).

    2. **Wearing a seatbelt prevents death**: Temporal precedence (wearing precedes the accident), covariation (strong statistical association between seatbelt use and survival), plausibility (physics of force distribution), experimental evidence (crash test data). Elimination of confounders is partially met — cautious drivers may be more likely to wear seatbelts, but the mechanical argument is very strong.

    3. **Eating organic food causes better health**: Covariation may exist but is weak. Temporal precedence is met trivially. Elimination of confounders is the weakest criterion — people who buy organic food tend to be wealthier, more health-conscious, and engage in more exercise. Plausibility is debatable (lower pesticide exposure, but unclear clinical significance). Experimental evidence is very limited. The causal claim is not well supported.

    4. **Social media use causes depression in teenagers**: Covariation exists in many studies. Temporal precedence is difficult to establish (does social media use precede depression, or do depressed teens use more social media?). Elimination of confounders is challenging (loneliness, family dynamics, and pre-existing mental health conditions confound). Plausibility exists (social comparison, cyberbullying). Experimental evidence is limited and ethically constrained. The causal claim remains contested.

---

**Exercise 2.**
Design a study to test the causal relationship between sleep duration and academic performance. Specify:

1. The type of study (observational, RCT, longitudinal)
2. How you would control for confounders
3. What variables you would measure
4. Potential ethical constraints
5. How you would interpret the results

??? success "Solution to Exercise 2"

    1. **Study type**: A longitudinal observational study with repeated measures is most feasible. A true RCT (randomly assigning sleep durations) would be ideal but raises ethical concerns about sleep deprivation.

    2. **Controlling for confounders**: Measure and adjust for socioeconomic status, prior academic performance, mental health, caffeine intake, screen time, extracurricular activities, and course difficulty. Use regression or propensity score matching to control for these confounders.

    3. **Variables to measure**: Sleep duration and quality (via actigraphy or sleep diaries), GPA or standardized test scores, demographics, health behaviors, mental health scales, and time-varying confounders measured at each follow-up.

    4. **Ethical constraints**: Cannot ethically force students to sleep specific amounts. Must rely on natural variation or gentle interventions (sleep hygiene education). Need informed consent and IRB approval.

    5. **Interpretation**: If the association between sleep and academic performance persists after controlling for confounders and the temporal ordering is correct (sleep measured before performance outcomes), the evidence supports a causal interpretation but cannot definitively prove causation due to potential unmeasured confounders. Effect sizes and confidence intervals should be reported alongside p-values.
