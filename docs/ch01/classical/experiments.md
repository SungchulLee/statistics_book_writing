# Controlled Experiments

A controlled experiment is the gold standard for establishing causal relationships, using random assignment and manipulation of variables to isolate treatment effects from confounders.

## Definition

A **controlled experiment** is a research design in which the investigator manipulates one or more independent variables (treatments), randomly assigns subjects to treatment and control groups, and measures the effect on a dependent variable. The **control group** receives a placebo or standard treatment; the **treatment group** receives the intervention. Random assignment ensures confounders are distributed evenly across groups.

## Explanation

The key advantage of controlled experiments over observational studies is the ability to establish **causation**, not merely association. Randomization ensures that any systematic differences between groups (age, health, socioeconomic status) balance out on average, so observed differences in outcomes can be attributed to the treatment.

The **placebo effect** -- genuine improvement from believing one is treated -- is controlled by giving the control group an inert treatment identical in appearance. Without a placebo control, treatment effects and psychological effects are confounded.

| | Controlled Experiment | Observational Study |
|---|---|---|
| Group assignment | Investigator (random) | Subject (self-selected) |
| Confounding | Minimized by randomization | Many potential confounders |
| Causal claims | Yes | Association only |

Limitations: ethical constraints (cannot assign harmful treatments), artificiality of lab settings, high cost and time, and limited generalizability from tightly controlled conditions.

## Examples

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n_per_group = 100
true_effect = 5.0

# Simulate a randomized experiment
control = np.random.normal(50, 10, n_per_group)
treatment = np.random.normal(50 + true_effect, 10, n_per_group)

t_stat, p_value = stats.ttest_ind(treatment, control)
diff = treatment.mean() - control.mean()
print(f"Control mean:   {control.mean():.2f}")
print(f"Treatment mean: {treatment.mean():.2f}")
print(f"Difference:     {diff:.2f} (true effect = {true_effect})")
print(f"t-statistic:    {t_stat:.3f}")
print(f"p-value:        {p_value:.4f}")
```

## Exercises

**Exercise 1.**
A pharmaceutical company tests a new drug by giving it to 100 volunteers who signed up. A separate group of 100 people who did not sign up serves as the control. Identify the key flaw in this experimental design.

??? success "Solution to Exercise 1"
    The key flaw is **self-selection bias** (lack of random assignment). Volunteers who sign up may differ systematically from those who do not -- they may be more health-conscious, have more severe symptoms, or have stronger beliefs that the drug works. Any observed difference between the groups could be due to these pre-existing differences rather than the drug itself. A proper randomized controlled trial would randomly assign all 200 participants to treatment or control, ensuring the groups are comparable on both observed and unobserved characteristics.

---

**Exercise 2.**
Explain the difference between a randomized controlled trial (RCT) and a natural experiment. Give one example of each.

??? success "Solution to Exercise 2"
    In a **randomized controlled trial**, the researcher actively assigns subjects to treatment and control groups using a random mechanism (e.g., coin flip, random number generator). Example: randomly assigning patients to receive a new drug versus a placebo.

    In a **natural experiment**, an external event or policy creates treatment and control groups "as if" by random assignment, but without the researcher's intervention. Example: a state lottery for school vouchers, where winners (treatment) and losers (control) are determined by chance rather than by the researcher.

    The key distinction is control: in an RCT the researcher controls assignment; in a natural experiment, nature or policy provides the variation. Natural experiments are valuable when RCTs are unethical or impractical, but their validity depends on the plausibility of the "as-if random" assumption.

---

**Exercise 3.**
A study finds that students who attend tutoring sessions score 15 points higher on exams than students who do not. The school concludes that tutoring causes a 15-point improvement. What confounders might explain this result without a causal effect of tutoring?

??? success "Solution to Exercise 3"
    Several confounders could explain the association without a causal effect:

    - **Motivation:** Students who attend tutoring may be more motivated to succeed and would have scored higher regardless.
    - **Prior ability:** Tutoring may attract students who already have stronger foundations.
    - **Study time:** Students who attend tutoring may also spend more time studying on their own.
    - **Parental involvement:** Students from families that arrange tutoring may have other academic support at home.
    - **Teacher recommendation:** Teachers may recommend tutoring to students who are already improving.

    Without random assignment to tutoring (or a convincing instrument/natural experiment), the 15-point difference reflects both the causal effect of tutoring and the selection effect.

---

**Exercise 4.**
Define the terms **internal validity** and **external validity** in the context of controlled experiments. Can an experiment have high internal validity but low external validity?

??? success "Solution to Exercise 4"
    **Internal validity** is the degree to which an experiment establishes a causal relationship between treatment and outcome within the study population. It depends on proper randomization, absence of confounders, and absence of systematic bias.

    **External validity** (generalizability) is the degree to which the results apply to populations, settings, or conditions beyond the study.

    Yes, an experiment can have high internal validity but low external validity. For example, a carefully randomized lab experiment on college students demonstrates a causal effect (high internal validity), but the effect may not generalize to the broader population because college students are not representative in age, education, or socioeconomic status (low external validity). This tension is fundamental in experimental design: tightly controlled settings improve internal validity but may limit generalizability.

---

**Exercise 5.**
A factorial experiment varies two binary factors (A: low / high; B: low / high) for a total of 4 treatment combinations, with 25 subjects per combination. State two quantities estimable in this design that would not be available with two separate one-factor experiments.

??? success "Solution to Exercise 5"
    A factorial design lets you estimate:

    - **Main effects** of A and B (averaging across levels of the other factor) — also obtainable from one-factor designs.
    - **Interaction effect** A × B: whether the effect of A *depends on* the level of B. For example, a drug + diet study might find that the drug works only on the high-protein diet — an interaction undetectable from separate single-factor experiments.
    - **Improved precision for main effects** at the same total $n$: each main-effect contrast uses all 100 subjects (50 vs. 50), not just 50 (25 vs. 25 in a single-factor design).

    Fisher's argument for factorial designs (the foundational *Design of Experiments*, 1935) is that interactions are common in nature and often the most scientifically interesting finding — yet they are invisible to "one factor at a time" experimentation.

---

**Exercise 6.**
**Intention-to-treat (ITT)** analysis is the convention in clinical-trial reporting: analyze each subject in the group they were randomly assigned to, regardless of whether they actually received the treatment. Explain why ITT is preferred over **per-protocol** analysis (which excludes non-compliers) despite seemingly diluting the estimated treatment effect.

??? success "Solution to Exercise 6"
    Randomization guarantees comparability between assigned groups, not between groups that actually received the treatment. Compliance is itself a behavior that may correlate with prognosis: sicker patients may stop taking a drug because of side effects, healthier patients may forget doses, motivated patients may comply more. **Per-protocol** analysis discards exactly the patients whose non-compliance carries information, breaking randomization and reintroducing confounding.

    **ITT** preserves the original random groups and estimates the effect of *being assigned to* the treatment — the **causal effect of policy** rather than the effect of the molecule. This is what regulators and policy-makers actually need: when a drug is approved, real-world patients will exhibit similar compliance patterns. ITT estimates are typically more conservative (closer to the null) but unbiased and externally valid.

    A common companion is the **complier-average causal effect (CACE)** estimated by instrumental-variable methods, using random assignment as an instrument for actual treatment receipt. CACE gives the effect among compliers without violating randomization.
