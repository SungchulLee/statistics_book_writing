# Randomization and Blinding

Randomization and blinding are the two pillars of experimental design. Randomization guards against confounding by ensuring that observed and unobserved variables are balanced between treatment groups on average. Blinding guards against human expectation effects — the placebo effect on subjects, observer bias in measurement, and analyst bias in choices made during data processing. The combination of the two produces the **double-blind randomized controlled trial (RCT)**, which is the regulatory gold standard for causal evidence in medicine and a touchstone benchmark in psychology, education, and policy research.

## Definition

**Randomization** assigns subjects to treatment or control groups by a chance mechanism (random number generator, sealed envelopes, computer-generated allocation). Every subject has a known, nonzero probability of being assigned to each group, and the assignment is independent of their characteristics.

**Blinding** (also called **masking**) conceals the group assignment from parties whose knowledge could distort the trial. The standard levels:

| Level | Subject knows? | Treating clinician knows? | Outcome assessor / analyst knows? |
|---|---|---|---|
| Open-label | Yes | Yes | Yes |
| Single-blind | No | Yes | Yes |
| Double-blind | No | No | Yes |
| Triple-blind | No | No | No |

A trial may also be **outcome-assessor-blinded**, where clinicians know assignments but the people scoring outcomes do not — a useful compromise when full double-blinding is impractical (e.g., surgical interventions).

## Explanation

### What randomization achieves

Under random assignment, the treatment indicator $T$ is statistically independent of the potential outcomes $(Y(0), Y(1))$:

$$
T \perp (Y(0), Y(1))
$$

This independence is what makes the difference in observed group means an unbiased estimator of the average causal effect $\mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]$. Equally importantly, randomization gives a **probabilistic basis** for $p$-values: under the null hypothesis of no treatment effect, the distribution of the test statistic is determined by the randomization scheme alone, without reference to any model of the data.

### What randomization does not achieve

Randomization balances groups *in expectation*. Any particular randomization may produce imbalanced covariates — small studies are vulnerable to "unlucky" assignments where, say, all sicker patients land in one arm. Defenses:

- **Stratified (block) randomization**: randomize within strata defined by prognostic covariates, guaranteeing balance.
- **Larger sample size**: imbalance probability shrinks as $1/\sqrt{n}$.
- **Pre-specified covariate adjustment**: include important covariates in the analysis model regardless of randomization.

### What blinding achieves

- **Subject blinding** controls the placebo effect — genuine improvement caused by belief in treatment. Without it, the placebo effect is confounded with the pharmacological effect.
- **Clinician blinding** prevents differential care (e.g., more attentive follow-up for treatment arm) and differential measurement (e.g., recording slightly more favorable assessments for the treated).
- **Analyst blinding** prevents data-dependent choices in cleaning, exclusion criteria, and analysis specification — the formal counter to the "garden of forking paths."

### Historical landmark

The 1954 Salk polio vaccine trial randomly assigned about 400,000 children to vaccine or placebo in a double-blind design. The vaccine group experienced 28 cases per 100,000 versus 71 in controls, conclusively establishing efficacy. The parallel non-randomized **NFIP design** that used grade-level controls found a similar effect but with weaker internal validity, since second-graders (offered vaccine) differed systematically from first- and third-graders (the implicit control). The randomized arm became the basis for licensing; the non-randomized arm has been a teaching example ever since.

## Examples

```python
"""Randomized vs. confounded assignment under the same true effect."""

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n = 200
treatment_effect = 3.0

# === Properly randomized trial ===
control = rng.normal(50, 8, n)
treatment = rng.normal(50 + treatment_effect, 8, n)
t_stat, p_val = stats.ttest_ind(treatment, control)
print(f"[Randomized] estimated effect = {treatment.mean() - control.mean():+.2f} "
      f"(true = {treatment_effect}), p = {p_val:.4f}")

# === Confounded (non-random) assignment ===
severity = rng.uniform(0, 10, 2 * n)
prob_treat = 1 / (1 + np.exp(-(severity - 5)))  # sicker -> more likely treated
assigned = rng.binomial(1, prob_treat).astype(bool)
outcome = 50 - 2 * severity + treatment_effect * assigned + rng.normal(0, 5, 2 * n)
est = outcome[assigned].mean() - outcome[~assigned].mean()
print(f"[Confounded] estimated effect = {est:+.2f}  (severity confounds)")
```

## Exercises

**Exercise 1.**
A clinical trial tests whether a new pain reliever works better than an existing one.

**(a)** Explain the purpose of random assignment in this trial.
**(b)** Define single-blind, double-blind, and triple-blind designs. Which would you recommend and why?
**(c)** If the new drug is a pill and the existing drug is an injection, what problem arises for blinding? How might researchers address it?

??? success "Solution to Exercise 1"
    (a) Random assignment distributes both known (age, weight, baseline pain) and unknown confounders evenly across treatment groups in expectation. The resulting difference in mean outcomes is an unbiased estimator of the causal treatment effect, with valid $p$-values whose distribution under the null is determined by the randomization itself.

    (b) Single-blind: subject unaware. Double-blind: subject and treating clinician both unaware. Triple-blind: subject, clinician, and analyst all unaware. Recommend at least **double-blind** to control both the placebo effect (subject expectation) and observer bias (clinicians might unconsciously favor the new drug in pain assessments).

    (c) Different routes of administration break blinding. Subjects know whether they swallowed a pill or received an injection. Standard fix: **double-dummy** — every subject receives one of each, with one being a placebo. The new-drug arm gets a real pill and a saline injection; the existing-drug arm gets a placebo pill and the real injection.

---

**Exercise 2.**
A trial randomizes 200 patients, 100 to treatment and 100 to control. By chance, the treatment group has mean age 65 and the control group has mean age 55. Is the trial invalid? What is the correct response?

??? success "Solution to Exercise 2"
    The trial is not invalid — randomization was performed correctly, and the difference is the kind of imbalance that can arise by chance, especially in modest samples.

    Correct responses (in order of preference):

    - **Pre-specified covariate adjustment**: include age as a covariate in the analysis (ANCOVA or regression). This often improves precision and corrects for the chance imbalance.
    - **Report the result both unadjusted and adjusted**: if conclusions agree, the imbalance is unlikely to be driving the finding.
    - **Stratified randomization** (a future-design fix): in the next trial, stratify on age so imbalance cannot arise by chance.

    What you should *not* do: re-randomize after seeing the imbalance, or compare $p$-values for the baseline imbalance and use them to decide whether to adjust. Both invalidate the design.

---

**Exercise 3.**
Explain why **block (stratified) randomization** is preferred to **simple randomization** in a multi-center trial where the centers differ in patient case-mix.

??? success "Solution to Exercise 3"
    Simple randomization can produce imbalanced treatment counts within centers — by chance, one center might enroll mostly treatment patients and another mostly controls. If centers also differ in patient case-mix or follow-up quality, the imbalance confounds the center effect with the treatment effect.

    Stratified randomization randomizes *within* each center (typically in fixed blocks of size 4 or 6: TTCC, TCTC, CTCT, etc.). Each center then has a guaranteed near-1:1 ratio, removing center as a potential confounder. The cost is slightly more complex randomization procedure; the benefit is more efficient estimation and a more interpretable analysis.

---

**Exercise 4.**
A surgical trial cannot blind the surgeon. Discuss the threats to validity this introduces and the best partial mitigations.

??? success "Solution to Exercise 4"
    Surgeons may, consciously or not, vary their skill, attention, or post-operative care between the trial arms. Subjects may also infer their assignment from incidental cues, undermining subject blinding.

    Partial mitigations:

    - **Outcome-assessor blinding** is critical: have independent assessors who do not know assignment score the outcomes (pain, range of motion, blinded radiographs).
    - **Active-comparator design**: instead of "surgery vs. no surgery," compare two surgical procedures so both arms have a procedure with similar appearance. Sham surgery trials (with full anesthesia and incisions but no procedure) have been used for some interventions when ethically permissible.
    - **Pre-specified surgical protocols**: minimize discretion that could vary by arm.
    - **Audit trails**: video-record procedures or have independent observers in theater.

    Even with these in place, residual bias in surgical trials is generally larger than in pharmacological trials. Strong effects are required to compensate.

---

**Exercise 5.**
A drug trial blinds patients but not clinicians. Patients in the treatment arm experience common side effects (dry mouth, mild nausea) that placebo patients do not. Why does this threaten the integrity of the blind, and what is the consequence?

??? success "Solution to Exercise 5"
    Side effects "unblind" patients indirectly: a patient who notices dry mouth and nausea correctly guesses they are on the active drug, and the placebo group's lack of side effects similarly reveals their status. This is called **functional unblinding**.

    Consequences:

    - The placebo effect is no longer balanced: only one group expects benefit.
    - Subjects may report outcomes (especially subjective ones like fatigue or mood) differently based on inferred assignment.
    - Adherence may diverge: subjects who believe they are on placebo may drop out or seek other treatments.

    Mitigations: use an **active placebo** that mimics the side-effect profile without therapeutic effect (e.g., low-dose atropine for studies of anticholinergic drugs), and use objective outcome measures that are less sensitive to expectation effects.

---

**Exercise 6.**
Distinguish between **randomization** (which is a feature of the *design*) and **bootstrap resampling** (which is a feature of the *analysis*). Why is the former necessary for causal inference while the latter is not?

??? success "Solution to Exercise 6"
    **Randomization** is performed *before* outcomes are measured. It is the act of physically assigning subjects to treatment by a chance mechanism, which creates statistical independence between assignment and pre-existing characteristics. Causal inference rests on this independence.

    **Bootstrap resampling** is performed *after* outcomes are measured. It is an analysis technique that repeatedly resamples the observed data to estimate the sampling variability of a statistic. It does not change the data-generating process and provides no information about whether observed associations are causal.

    The bootstrap can quantify uncertainty in a regression coefficient estimated from observational data, but if the coefficient is biased by unmeasured confounding, the bootstrap CIs simply reflect uncertainty around the biased value. No amount of resampling fixes a non-random sample. Causation requires intervention (in design) or strong identifying assumptions (in analysis) — not just precise estimation.
