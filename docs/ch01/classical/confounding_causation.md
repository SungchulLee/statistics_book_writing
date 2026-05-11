# Confounding and Association vs Causation

Two variables can move together without one causing the other. **Confounding** is the formal name for the mechanism that produces such non-causal associations, and the slogan "correlation does not imply causation" is its everyday summary. Understanding confounding is essential for interpreting any statistical association in observational data — and recognizing it is the difference between a useful empirical finding and an actionable misdirection.

## Definition

A **confounding variable** $Z$ is a variable associated with both the exposure $X$ and the outcome $Y$, but not on the causal pathway from $X$ to $Y$. Graphically:

$$
\begin{aligned}
Z &\to X \\
Z &\to Y
\end{aligned}
$$

When $Z$ is present but unaccounted for, the observed association between $X$ and $Y$ mixes the (possibly zero) causal effect of $X$ on $Y$ with the spurious association induced through $Z$.

The directional language matters: a variable on the *causal pathway* ($X \to Z \to Y$) is a **mediator**, not a confounder. A variable caused by both $X$ and $Y$ is a **collider**, and conditioning on it can *create* spurious associations rather than remove them.

## Explanation

### Classic examples

- **Ice cream sales and drowning** both rise in summer. Confounder: temperature.
- **Coffee drinkers and lung cancer** appear correlated. Historical confounder: smoking (coffee drinkers tended to smoke).
- **Shoe size and reading ability** in schoolchildren are positively correlated. Confounder: age.
- **Hospital admission and mortality**: hospitalized patients die more often. Confounder: severity of illness.

In each case, the observed association is real but the proposed causal interpretation is wrong.

### Simpson's paradox

A particularly dramatic form of confounding: an association seen within every subgroup can *reverse* when subgroups are pooled. The 1973 Berkeley graduate-admissions study famously showed that women appeared to be admitted at a lower rate than men overall — but within each individual department, women were admitted at the same or higher rate. The confounder: women applied disproportionately to competitive (low-admit-rate) departments.

### From association to causation

The two routes to identifying causal effects:

1. **Intervention**: randomly assign $X$. Under randomization, $X$ becomes independent of all $Z$ (observed and unobserved), so the observed association reflects only the causal pathway. This is the gold standard.

2. **Identifying assumptions** (when randomization is impossible):
   - **Stratification / matching / regression adjustment** under the assumption that all confounders have been measured.
   - **Instrumental variables** when an exogenous source of variation affects $X$ but not $Y$ directly.
   - **Difference-in-differences** under a parallel-trends assumption.
   - **Regression discontinuity** when a sharp threshold determines $X$.

Each replaces randomization with an assumption that cannot be tested from the data alone.

### Adjusting for the wrong thing

A reflexive instinct is to "adjust for everything." This is wrong:

- **Mediators**: adjusting for a mediator blocks the very pathway you want to estimate. If the question is "does education increase earnings?", adjusting for occupation (a mediator) shrinks the estimated effect because you've already eliminated the indirect pathway.
- **Colliders**: adjusting for a variable caused by both $X$ and $Y$ opens a spurious path between them. Restricting an analysis to hospitalized patients (a collider of disease severity and treatment) can produce associations between treatment and outcome that do not exist in the population.

Causal diagrams (DAGs, Pearl 2000) formalize which adjustments are valid.

## Examples

```python
"""Simpson's paradox: confounding by department in admissions."""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500

dept = rng.choice(["A", "B"], size=n, p=[0.5, 0.5])
# Dept A: harder grading, attracts more studious applicants
study_hours = np.where(dept == "A",
                       rng.normal(8, 1, n),
                       rng.normal(4, 1, n))
grade = np.where(dept == "A",
                 50 + 3 * study_hours + rng.normal(0, 5, n),
                 70 + 3 * study_hours + rng.normal(0, 5, n))

df = pd.DataFrame({"dept": dept, "study_hours": study_hours, "grade": grade})

print(f"Overall correlation (hours, grade): {df['study_hours'].corr(df['grade']):+.3f}")
for d in ["A", "B"]:
    sub = df[df["dept"] == d]
    print(f"  Dept {d}: corr = {sub['study_hours'].corr(sub['grade']):+.3f}, "
          f"mean grade = {sub['grade'].mean():.1f}")
```

The within-department correlation is positive (more study → higher grades), as causation suggests. The overall correlation is dampened (or could be reversed in a more extreme setup) because the harder-grading department has both more studying *and* lower grades — the same departmental confounder that drives Simpson's paradox.

## Exercises

**Exercise 1.**
A newspaper reports that cities with more ice-cream trucks have higher crime rates, concluding that ice-cream trucks cause crime.

**(a)** Identify a plausible **confounding variable** explaining the association.
**(b)** Explain why the observational design cannot establish a causal relationship.
**(c)** Sketch a study that could better isolate the relationship, and explain why it is impractical.

??? success "Solution to Exercise 1"
    (a) **Temperature** (or **population density**). Hot weather increases both ice-cream demand (more trucks) and outdoor activity (more opportunities for crime). Population density does the same: denser cities have more of everything per square mile, including both trucks and crime.

    (b) The data are observational. Without controlling for the confounder, the observed correlation conflates a possible causal effect of ice-cream trucks with the lurking effect of temperature/density. Without controlling for *all* relevant confounders (which is impossible in general), no causal claim is justified.

    (c) A randomized experiment would randomly assign cities to different numbers of ice-cream trucks and measure crime rates. Impractical because (i) cities cannot be controlled like experimental units, (ii) ice-cream-truck counts are not centrally manipulable, (iii) ethical considerations of changing public-safety conditions for an experiment.

    A more realistic approach: a natural experiment using a sudden policy change in licensing in one city, with a similar city as a control (difference-in-differences). Even this is shaky — the parallel-trends assumption is hard to defend at the city level.

---

**Exercise 2.**
Distinguish **confounder**, **mediator**, and **collider**. For each, state the consequence of "adjusting for" it in a regression analysis of the effect of $X$ on $Y$.

??? success "Solution to Exercise 2"
    **Confounder** $Z$: causes both $X$ and $Y$. **Adjusting for it** removes the spurious component of the $X$-$Y$ association, *reducing* bias.

    **Mediator** $M$: caused by $X$ and itself causes $Y$ (so $X \to M \to Y$). **Adjusting for it** blocks the indirect pathway, *increasing* bias — the adjusted coefficient now estimates only the direct effect $X \to Y$, not the total effect.

    **Collider** $C$: caused by both $X$ and $Y$ (so $X \to C \leftarrow Y$). **Adjusting for it** opens a non-causal pathway between $X$ and $Y$ — creating a spurious association where none existed. This is **selection bias** in regression form.

    The rule: adjust for confounders; do *not* adjust for mediators (unless you want the direct effect explicitly) or colliders. Knowing which is which requires substantive knowledge of the data-generating process, often best expressed as a causal DAG.

---

**Exercise 3.**
**Simpson's paradox** with a concrete table. A drug is tested in two hospitals with the following outcomes:

| | Hospital A — recovered | A — total | Hospital B — recovered | B — total |
|---|---|---|---|---|
| Drug | 80 | 100 | 200 | 300 |
| No drug | 240 | 300 | 50 | 100 |

Compute the recovery rate for drug vs. no-drug overall and within each hospital. Comment on the apparent reversal.

??? success "Solution to Exercise 3"
    **Within Hospital A:** drug recovery rate $= 80/100 = 80\%$; no-drug rate $= 240/300 = 80\%$. Identical.

    **Within Hospital B:** drug recovery rate $= 200/300 = 66.7\%$; no-drug rate $= 50/100 = 50\%$. Drug helps.

    **Pooled:** drug $= (80+200)/(100+300) = 280/400 = 70\%$; no-drug $= (240+50)/(300+100) = 290/400 = 72.5\%$.

    Pooled, no-drug looks better. Within each hospital, drug is at least as good as no-drug — never worse. The reversal occurs because **drug use is correlated with hospital** (75% of drug patients are in Hospital B, vs. only 25% of no-drug patients), and **recovery rate also differs by hospital** (A has higher rates than B). Hospital is the confounder. The correct conclusion is the within-hospital one: the drug is at least as effective. Pooling without controlling for hospital is misleading.

---

**Exercise 4.**
The **back-door criterion** (Pearl 1995) gives a graphical rule for identifying which variables suffice as adjustment controls. State the criterion informally and give a small DAG example where adjusting for *the wrong set* opens a back-door rather than closing one.

??? success "Solution to Exercise 4"
    **Back-door criterion (informal):** to identify the causal effect of $X$ on $Y$, choose a set $S$ of variables such that (1) no variable in $S$ is a descendant of $X$, and (2) $S$ blocks every "back-door" path from $X$ to $Y$ — i.e., every path between $X$ and $Y$ that starts with an arrow into $X$.

    **DAG example:** suppose $X \to Y$, $Z \to X$, $Z \to W$, $Y \to W$. Here $W$ is a collider on the path $X \to Y \to W \leftarrow Z \to X$. Adjusting for $\{Z\}$ closes the back-door path $X \leftarrow Z \to W \leftarrow Y$ (which was already blocked at the collider $W$). Adjusting for $\{Z, W\}$ **opens** the back-door path through $W$ — conditioning on the collider creates an association between $Z$ and $Y$ within strata of $W$, which then leaks into the $X$-$Y$ analysis.

    The lesson: more controls are not always better. Selecting controls requires knowing the causal structure, and "adjust for everything" can hurt as easily as help.

---

**Exercise 5.**
A medical researcher claims a 30% reduction in heart-attack risk among coffee drinkers. The press releases summarize this as "coffee prevents heart attacks." List two confounders, then describe what kind of study would be needed to upgrade the claim to a causal conclusion.

??? success "Solution to Exercise 5"
    **Two confounders:**

    - **Lifestyle**: coffee drinkers tend to be employed, urban, and more health-aware overall (especially in modern Western samples — historical samples have the opposite pattern). These factors also reduce heart-attack risk.
    - **Selection (reverse causation)**: people with early signs of heart trouble have often been told by their doctor to reduce coffee intake. So sicker people stop drinking coffee. The remaining coffee drinkers are *healthier on average* — making coffee look protective when in fact coffee did nothing and the causal arrow points the other way.

    **Causal upgrade:** a randomized controlled trial. Assign healthy volunteers to drink 2–3 cups of coffee daily versus 0 cups for a decade. Measure heart-attack incidence. Ethical and practical issues: long follow-up, compliance, blinding (subjects know whether they drink coffee). Realistic compromise: instrumental-variable analysis using genetic variants that affect caffeine metabolism (Mendelian randomization) — exploits the fact that genetic variants are randomly assigned at conception and predict coffee consumption without being affected by lifestyle.

---

**Exercise 6.**
Distinguish the **average treatment effect (ATE)** from the **average treatment effect on the treated (ATT)**. When can observational data identify ATE, when only ATT, and why does this matter for policy?

??? success "Solution to Exercise 6"
    **ATE:** $\mathbb{E}[Y(1) - Y(0)]$ averaged over the entire population. What you'd see if you treated everyone vs. no one.

    **ATT:** $\mathbb{E}[Y(1) - Y(0) \mid T = 1]$ averaged over the treated subpopulation. What you'd see if you treated those who were actually treated, vs. not treating them.

    **From observational data:**

    - Under unconfoundedness $(Y(0), Y(1)) \perp T \mid X$ and overlap, **ATE** is identified — we can estimate counterfactual outcomes for both treated and untreated under their respective conditional distributions.
    - Under weaker conditions (e.g., $Y(0) \perp T \mid X$ alone), **ATT** is identified but not ATE — we can estimate counterfactual untreated outcomes for the treated, but not counterfactual treated outcomes for the untreated.

    **Policy implication:** if a policy will be applied to *the same population that volunteered for the program*, ATT is what matters. If the policy will be mandated for *everyone*, ATE is what matters — and the answer can be different. A drug that benefits the treated (who selected in because they expected benefit) may have little or even negative effect on the average person who would not have chosen it. This distinction haunts policy debates about voluntary versus mandatory programs and is rarely articulated explicitly in headline numbers.
