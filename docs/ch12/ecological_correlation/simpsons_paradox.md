# Simpson's Paradox

A trend that holds within every subgroup of a dataset can reverse or disappear when the subgroups are combined. This counterintuitive phenomenon is known as **Simpson's paradox**. It arises when a confounding variable influences both the grouping and the outcome, and it serves as a powerful reminder that aggregated data can be deeply misleading.

---

## Definition

**Simpson's paradox** occurs when an association between two variables reverses direction after conditioning on (stratifying by) a third variable. Formally, it is possible to have

$$
P(Y = 1 \mid X = 1, Z = z) < P(Y = 1 \mid X = 0, Z = z) \quad \text{for every } z
$$

and yet

$$
P(Y = 1 \mid X = 1) > P(Y = 1 \mid X = 0)
$$

when marginalizing over $Z$. The inequality reverses when the data are aggregated.

---

## The Berkeley Admissions Example

The most famous illustration of Simpson's paradox comes from graduate admissions at UC Berkeley in 1973.

**Aggregate data:**

| | Applicants | Admitted | Rate |
|:---|:---:|:---:|:---:|
| Men | 8442 | 3738 | 44% |
| Women | 4321 | 1494 | 35% |

At first glance, men appear to be admitted at a substantially higher rate, suggesting possible gender bias.

**Stratified by department:**

| Department | Men applied | Men admitted | Women applied | Women admitted |
|:---:|:---:|:---:|:---:|:---:|
| A | 825 | 62% | 108 | 82% |
| B | 560 | 63% | 25 | 68% |
| C | 325 | 37% | 593 | 34% |
| D | 417 | 33% | 375 | 35% |
| E | 191 | 28% | 393 | 24% |
| F | 373 | 6% | 341 | 7% |

Within most departments, women had equal or higher admission rates. The paradox arose because women disproportionately applied to more competitive departments (C, D, E, F) with lower overall admission rates, while men applied more to less competitive departments (A, B).

The confounding variable $Z$ (department choice) reversed the aggregate association.

---

## Why It Happens

Simpson's paradox occurs when:

1. A confounding variable $Z$ is associated with both the treatment/exposure $X$ and the outcome $Y$.
2. The groups defined by $X$ have different distributions over $Z$.
3. The effect of $Z$ on $Y$ is strong enough to overwhelm the $X$-$Y$ relationship when data are aggregated.

The paradox is not a contradiction in the mathematics. The marginal probability $P(Y \mid X)$ is a weighted average of the conditional probabilities $P(Y \mid X, Z = z)$, with weights proportional to $P(Z = z \mid X)$. When those weights differ between the $X = 0$ and $X = 1$ groups, the weighted averages can reverse the direction of the conditional relationship.

Formally:

$$
P(Y = 1 \mid X = x) = \sum_z P(Y = 1 \mid X = x, Z = z) \, P(Z = z \mid X = x)
$$

Different weighting schemes (different $P(Z = z \mid X = x)$) can produce different aggregate results.

---

## A Numerical Example

A hospital compares two treatments for kidney stones.

**Overall (aggregated):**

| Treatment | Successes | Total | Rate |
|:---|:---:|:---:|:---:|
| A | 273 | 350 | 78% |
| B | 289 | 350 | 83% |

Treatment B appears better overall.

**Stratified by stone size:**

| Stone size | Treatment A success rate | Treatment B success rate |
|:---:|:---:|:---:|
| Small | 93% (81/87) | 87% (234/270) |
| Large | 73% (192/263) | 69% (55/80) |

Treatment A is better for small stones *and* better for large stones, yet Treatment B appears better overall. The paradox arises because Treatment A was disproportionately given to patients with large stones (harder cases), dragging down its aggregate success rate.

---

## Connection to Confounding

Simpson's paradox is a manifestation of **confounding**. The confounding variable $Z$ creates a spurious association (or masks a real one) in the aggregated data. The solution is the same as for any confounding problem:

1. **Stratify** by the confounding variable and analyze each stratum separately.
2. **Adjust** using statistical methods (e.g., standardization, regression).
3. **Randomize** in experimental design to break the association between $X$ and $Z$.

See [Confounding Variables](../confounding/confounding_variables.md) for a detailed treatment.

---

## When to Trust the Aggregate vs the Stratified Data

A common question is: should we believe the aggregate result or the stratified result? The answer depends on the causal structure:

- If $Z$ is a **confounder** (a common cause of $X$ and $Y$), then the stratified analysis is correct and the aggregate is misleading.
- If $Z$ is a **mediator** (on the causal path from $X$ to $Y$), then stratifying on $Z$ removes part of the causal effect, and the aggregate may be more appropriate.
- If $Z$ is a **collider** (caused by both $X$ and $Y$), then stratifying on $Z$ can *create* a spurious association where none exists.

Determining the correct analysis requires knowledge of the causal relationships, not just statistical associations. See [Directed Acyclic Graphs](../causation/dags.md) for a framework to reason about these structures.

---

## Simpson's Paradox in Continuous Data

While often illustrated with proportions and contingency tables, Simpson's paradox also occurs with continuous variables and regression. A positive correlation between $X$ and $Y$ in the overall data can become negative within every subgroup defined by $Z$. This is essentially the same phenomenon described in the [ecological fallacy](ecological_fallacy.md), viewed from a different angle.

---

## Summary

Simpson's paradox occurs when an association reverses direction after stratifying by a confounding variable. It arises because aggregation changes the weighting of subgroups, allowing a confounding variable to distort the overall relationship. The Berkeley admissions case and the kidney stone treatment example illustrate that aggregate data can be deeply misleading. Resolving the paradox requires identifying the correct causal structure and analyzing the data at the appropriate level.
