# Observational Studies

An **observational study** records data without the investigator intervening or assigning treatments. It is the dominant mode of research wherever experiments are impractical, unethical, or impossibly slow — epidemiology, economics, sociology, ecology, finance — and produces most of the empirical evidence in the social and health sciences. The price of this flexibility is a structural limitation: observational studies measure associations, not causal effects, unless additional identifying assumptions are imposed.

## Definition

An **observational study** is a research design in which the investigator observes subjects and measures variables without manipulating any conditions and without random assignment to groups. The four main archetypes are:

- **Cross-sectional**: snapshot of a population at a single point in time (e.g., a national health survey).
- **Cohort (longitudinal)**: a group followed over time, prospective (forward in time) or retrospective (looking backward from records).
- **Case–control**: subjects with a condition (cases) compared to those without (controls), with exposure history examined retrospectively.
- **Ecological**: data analyzed at the population or group level rather than individual level — vulnerable to the *ecological fallacy*.

## Explanation

### Why observational data cannot, by itself, establish causation

In an observational study, subjects self-select into groups. Differences in outcomes between groups confound the **treatment effect** with selection differences. The canonical example: smokers have worse health outcomes than non-smokers, but smokers also differ in income, diet, exercise, stress, and dozens of unmeasured variables. The observed association is the sum of the true causal effect (whatever it is) and the confounding selection effect.

Formally, denote the observed effect of treatment $T$ on outcome $Y$ by $\mathbb{E}[Y \mid T = 1] - \mathbb{E}[Y \mid T = 0]$. This equals the **causal** effect $\mathbb{E}[Y(1) - Y(0)]$ only when treatment assignment is independent of the potential outcomes — exactly what randomization guarantees and observational data does not.

### Common identification strategies

Although observational data alone cannot identify causal effects, several additional assumptions can:

| Strategy | Key assumption |
|---|---|
| Stratification / matching | Conditional on observed covariates, treatment is as-if random |
| Multivariable regression | Same, plus correct functional form |
| Propensity-score methods | Same, plus correct propensity-score model |
| Instrumental variables | An exogenous source of variation affects $T$ but not $Y$ directly |
| Difference-in-differences | Parallel trends in the absence of treatment |
| Regression discontinuity | Outcome and confounders are smooth at a sharp eligibility threshold |

Each replaces "randomization" with a different identifying assumption. None is verifiable from the data alone — choosing among them is part design, part craftsmanship.

### Strengths

- **Real-world relevance.** Effects observed in natural settings generalize more easily than results from controlled labs.
- **Feasibility.** Many exposures of interest (smoking, occupational hazards, education) cannot be randomly assigned.
- **Scale.** Modern administrative datasets routinely include millions of subjects, providing precision impossible in experiments.

### Weaknesses

- **Confounding** — the headline problem above.
- **Selection bias** in who enters the sample or whose data is recorded.
- **Reverse causation** — the "outcome" may actually drive the "exposure" (sick people exercise less, not the reverse).
- **Measurement error** in self-reported variables.

## Examples

```python
"""Demonstrate confounding by age in an observational dataset."""

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n = 500

# === Confounder: age ===
age = rng.uniform(20, 70, n)

# Older people exercise less AND have higher blood pressure
exercise = 10 - 0.1 * age + rng.normal(0, 1, n)
bp = 80 + 0.5 * age - 0.3 * exercise + rng.normal(0, 5, n)

# === Naive correlation ignoring age ===
r_naive, _ = stats.pearsonr(exercise, bp)
print(f"Naive correlation (exercise, BP): r = {r_naive:+.3f}")
print("  (appears exercise *raises* BP — wrong sign! age confounds)")

# === Partial correlation controlling for age ===
def residualize(y, x):
    b = np.polyfit(x, y, 1)
    return y - np.polyval(b, x)

r_partial, _ = stats.pearsonr(residualize(exercise, age),
                              residualize(bp, age))
print(f"Partial correlation (controlling age): r = {r_partial:+.3f}")
print("  (correct negative sign emerges)")
```

## Exercises

**Exercise 1.**
Classify each study as **observational** (cross-sectional, cohort, or case–control) or **experimental**.

**(a)** Researchers track 5,000 smokers and 5,000 non-smokers over 20 years to compare lung-cancer rates.
**(b)** A pharmaceutical company randomly assigns 300 patients to receive either a new drug or a placebo and measures symptom improvement after 8 weeks.
**(c)** A public-health team surveys 1,000 adults about their diet and exercise habits at a single point in time.
**(d)** Researchers identify 200 patients with heart disease and 200 without, then look back at their cholesterol history.
**(e)** A school is randomly assigned to receive a new curriculum; a neighboring school continues with the existing curriculum.

??? success "Solution to Exercise 1"
    (a) Observational — cohort (groups defined by exposure, followed forward in time).
    (b) Experimental — randomized controlled trial.
    (c) Observational — cross-sectional (data at one time point).
    (d) Observational — case–control (groups defined by outcome, exposure history examined retrospectively).
    (e) Experimental — randomized at the *school* (cluster) level, sometimes called a cluster-randomized trial.

---

**Exercise 2.**
A study finds that people who drink red wine moderately have lower rates of heart disease. List three plausible confounders that could produce this association even if red wine has no causal effect, and propose one identification strategy that could partially address them.

??? success "Solution to Exercise 2"
    Plausible confounders:

    - **Income / socioeconomic status**: regular red-wine drinkers tend to be wealthier; income drives access to healthcare and healthy food.
    - **Diet**: red-wine drinkers may also eat more vegetables and less processed food (a "Mediterranean" diet pattern).
    - **Exercise / overall health behaviors**: people who maintain moderate drinking may also exercise more.

    **Identification strategy:** an instrumental-variable approach using regional or temporal variation in alcohol taxation could provide quasi-experimental variation in consumption that is independent of personal income or diet (assuming taxes do not directly affect heart disease through other channels). Alternatively, a randomized controlled trial of moderate red-wine consumption (with appropriate ethical safeguards) would settle the question more directly.

---

**Exercise 3.**
Explain the difference between a **cohort study** and a **case–control study**. Why is case–control typically used for rare diseases?

??? success "Solution to Exercise 3"
    A **cohort study** identifies subjects by their *exposure* (smokers vs non-smokers) and follows them forward in time to observe outcomes. A **case–control study** identifies subjects by their *outcome* (cases with heart attack, controls without) and looks backward at their exposure history.

    Case–control is preferred for rare diseases because the alternative — a prospective cohort — would need an enormous sample to accrue enough cases. If a disease affects 1 in 10,000 people, a cohort of 1,000 people will yield about 0.1 expected cases, useless for analysis. A case–control study starts by *over-sampling* on the disease, recruiting (say) 200 cases and 200 controls and comparing exposure rates. The trade-off: cases and controls may be sampled from different populations, and recall of past exposure may be biased.

---

**Exercise 4.**
Consider an observational study reporting that students attending tutoring sessions score 15 points higher than students who do not. The school concludes tutoring causes a 15-point improvement. Identify three confounders, two of which you regard as **measurable** and one as **unmeasurable**, and explain why the latter is a fundamental limitation.

??? success "Solution to Exercise 4"
    **Measurable confounders:**

    - Prior academic ability (proxied by previous grades or standardized test scores).
    - Parental education / socioeconomic status (recorded on enrollment forms).

    **Unmeasurable confounder:**

    - Motivation / conscientiousness. Students who choose to attend optional tutoring tend to be more motivated, and motivation also drives study time outside tutoring and exam performance directly. There is no reliable way to measure motivation independently of the behaviors it causes.

    Why this is fundamental: measurable confounders can be addressed by stratification, matching, or regression. Unmeasurable confounders cannot. Even an analysis adjusting for every measured variable is biased to the extent that an unmeasured confounder explains both the exposure and the outcome. This is why randomization remains the gold standard — it balances *all* confounders, observed or not.

---

**Exercise 5.**
The **ecological fallacy** is the error of inferring individual-level associations from group-level data. Construct a small artificial example where the group-level correlation has the opposite sign of the individual-level correlation.

??? success "Solution to Exercise 5"
    Suppose two regions A and B have these citizens:

    | Region | Person | Income | Happiness |
    |---|---|---|---|
    | A | 1 | 10 | 1 |
    | A | 2 | 30 | 3 |
    | A | 3 | 50 | 5 |
    | B | 4 | 20 | 4 |
    | B | 5 | 40 | 6 |
    | B | 6 | 60 | 8 |

    **Individual level:** within each region, income and happiness are positively correlated (correlation = 1 in both regions).

    **Group level:** region A has mean income 30 and mean happiness 3; region B has mean income 40 and mean happiness 6. Group means show positive association.

    Now reorganize so that region B has uniformly *lower* citizens despite higher mean income — for example, B = (20→0, 40→2, 60→4), giving group means (40, 2) compared to A's (30, 3). Individual correlations within each group are still positive, but the group-level correlation between mean income and mean happiness is now *negative*. The same data tells two different stories at different aggregation levels; using one to make claims at the other level is the ecological fallacy. Simpson's paradox is the same phenomenon in a categorical setting.

---

**Exercise 6.**
A researcher claims their **multivariable regression** of $Y$ on $T$ controlling for ten covariates establishes the causal effect of $T$. State the additional identifying assumption this claim requires and explain one diagnostic that can probe (but not verify) it.

??? success "Solution to Exercise 6"
    The required assumption is **no unmeasured confounding** (also called *conditional independence* or *ignorability*): conditional on the ten measured covariates, treatment $T$ is independent of the potential outcomes $(Y(0), Y(1))$. Equivalently, all backdoor paths from $T$ to $Y$ are blocked by the covariates that have been controlled for.

    This assumption is **untestable** from the data alone — it is an assumption about variables you did *not* measure. Two probes that can flag (but not certify) the assumption:

    - **Sensitivity analysis** (e.g., Rosenbaum bounds, E-values): how strong would an unmeasured confounder have to be — in association with both $T$ and $Y$ — to overturn the conclusion? If the answer is "much stronger than any of your measured confounders," the result is robust; otherwise it is fragile.
    - **Negative-outcome controls**: pick an outcome that *should not* be affected by $T$ but would plausibly be affected by the same unmeasured confounder (e.g., for a smoking study, a negative outcome might be vehicle-accident mortality). If $T$ "predicts" the negative outcome after adjustment, unmeasured confounding remains.

    These diagnostics give evidence, not proof. The fundamental epistemic gap — that observational data cannot tell you what randomization would tell you — never fully closes.
