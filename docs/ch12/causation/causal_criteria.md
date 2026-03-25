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
