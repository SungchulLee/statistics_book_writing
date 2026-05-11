# Interaction Effect Plot

## Overview

An interaction plot is the primary visual tool for diagnosing whether the effect of one factor depends on the level of another in a factorial design. When the lines in the plot are parallel, there is no interaction; when they are not parallel, an interaction is present. This page explains how to construct and interpret interaction plots using the ToothGrowth dataset, where the two factors are supplement type (OJ, VC) and dose level (0.5, 1.0, 2.0).

## What Is an Interaction?

In a two-way factorial model

$$
y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}
$$

the interaction term $(\alpha\beta)_{ij}$ captures the extent to which the combined effect of factors $A$ and $B$ differs from the sum of their individual effects. Formally, there is no interaction when

$$
(\alpha\beta)_{ij} = 0 \quad \text{for all } i, j
$$

which is equivalent to saying that the cell mean can be written as $\mu_{ij} = \mu + \alpha_i + \beta_j$ (a purely additive model).

## Constructing the Plot

An interaction plot displays:

- **Horizontal axis:** levels of one factor (e.g., dose).
- **Vertical axis:** cell means of the response (e.g., mean tooth length).
- **Separate lines:** one for each level of the second factor (e.g., supplement type).

```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot

url = ('https://raw.githubusercontent.com/vincentarelbundock/'
       'Rdatasets/master/csv/datasets/ToothGrowth.csv')
df = pd.read_csv(url, usecols=[1, 2, 3])

fig, ax = plt.subplots(figsize=(10, 4))
interaction_plot(df['dose'], df['supp'], df['len'],
                 ax=ax, markers=['o', 's'], linestyles=['--', '-.'])
ax.set_title("Interaction: dose x supp on tooth length")
ax.set_xlabel("dose")
ax.set_ylabel("len")
plt.tight_layout()
plt.show()
```

## Reading the Plot

| Pattern | Interpretation |
|---|---|
| Parallel lines | No interaction; the effect of dose is the same for both supplements |
| Non-parallel lines (converging/diverging) | Ordinal interaction; the effect direction is the same but magnitude differs |
| Crossing lines | Disordinal (crossover) interaction; the effect direction reverses |

For the ToothGrowth data, the OJ and VC lines converge at dose 2.0, indicating an **ordinal interaction**: both supplements increase tooth length with dose, but the advantage of OJ over VC diminishes at the highest dose.

## Quantifying the Interaction

The interaction contrast for two specific levels of each factor is

$$
\psi = (\mu_{11} - \mu_{12}) - (\mu_{21} - \mu_{22})
$$

If $\psi = 0$, the difference between levels of $A$ is the same at both levels of $B$ (parallel lines). The formal $F$-test for the interaction in two-way ANOVA tests $H_0: \text{all } (\alpha\beta)_{ij} = 0$ simultaneously.

## Interpretation

- **Parallel lines** mean the two factors act independently. Main effects can be interpreted on their own.
- **Non-parallel lines** mean the effect of one factor changes depending on the level of the other. Main effect averages may be misleading because they average over qualitatively different effects.
- **Crossing lines** indicate a particularly strong interaction where the ranking of factor levels reverses. In such cases, reporting only main effects can be actively misleading.
- The interaction plot is a visual guide; always confirm with the formal interaction $F$-test in the two-way ANOVA table.

## Exercises

**Exercise 1.**
Given the following cell means for a $2 \times 3$ factorial design, sketch the interaction plot and determine whether the interaction is ordinal or disordinal.

| | $B_1$ | $B_2$ | $B_3$ |
|---|---|---|---|
| $A_1$ | 5 | 10 | 15 |
| $A_2$ | 8 | 10 | 12 |

??? success "Solution to Exercise 1"
    Plot $B$ levels (1, 2, 3) on the horizontal axis and cell means on the vertical axis.

    - Line for $A_1$: 5, 10, 15 (slope = 5 per unit of $B$).
    - Line for $A_2$: 8, 10, 12 (slope = 2 per unit of $B$).

    The lines intersect between $B_1$ and $B_2$ (since $A_1$ starts below $A_2$ at $B_1$ but surpasses it at $B_3$). This is a **disordinal (crossover) interaction** because the ranking of $A_1$ vs. $A_2$ reverses across levels of $B$.

---

**Exercise 2.**
Prove that in a $2 \times 2$ factorial design, there is exactly one degree of freedom for the interaction. Express the interaction sum of squares in terms of the four cell means.

??? success "Solution to Exercise 2"
    With $a = 2$ levels of $A$ and $b = 2$ levels of $B$, the interaction has $(a-1)(b-1) = 1 \times 1 = 1$ degree of freedom.

    The interaction effect for the $(i,j)$ cell is $(\alpha\beta)_{ij} = \mu_{ij} - \mu_{i\cdot} - \mu_{\cdot j} + \mu_{\cdot\cdot}$. For a balanced design with $n$ observations per cell, the interaction sum of squares is

    $$
    SS_{AB} = n \sum_{i=1}^{2}\sum_{j=1}^{2} (\bar{y}_{ij\cdot} - \bar{y}_{i\cdot\cdot} - \bar{y}_{\cdot j\cdot} + \bar{y}_{\cdot\cdot\cdot})^2
    $$

    Since each of the four terms equals $\pm \delta$ where $\delta = (\bar{y}_{11} - \bar{y}_{12} - \bar{y}_{21} + \bar{y}_{22})/4$, we get

    $$
    SS_{AB} = \frac{n}{4}(\bar{y}_{11} - \bar{y}_{12} - \bar{y}_{21} + \bar{y}_{22})^2
    $$

    This is a single squared contrast, confirming the single degree of freedom. $\square$

---

**Exercise 3.**
Explain why it can be misleading to interpret main effects when a significant disordinal interaction is present. Use the ToothGrowth example to illustrate.

??? success "Solution to Exercise 3"
    The main effect of a factor is the difference between its marginal means, averaged over all levels of the other factor. When a disordinal interaction is present, the direction of one factor's effect reverses across levels of the other factor, so the average can be near zero even though the factor has a large effect at each level.

    In the ToothGrowth context, suppose (hypothetically) that OJ produced longer teeth than VC at low doses but shorter teeth at high doses. The marginal means for OJ and VC might be nearly equal, making the main effect of supplement non-significant. Yet the supplement clearly matters -- its effect simply depends on the dose. Reporting "supplement has no significant effect" would be misleading. The correct interpretation is conditional: OJ is better at low doses, VC is better at high doses, and the interaction is the key finding.

---

**Exercise 4.**
A researcher produces an interaction plot and observes non-parallel lines but the ANOVA interaction $F$-test gives $p = 0.23$. Explain how this can happen and what the researcher should conclude.

??? success "Solution to Exercise 4"
    Non-parallel lines in the interaction plot reflect sample cell means, which are subject to sampling variability. Even when there is no population-level interaction, random fluctuation will cause the sample lines to be slightly non-parallel. A non-significant $F$-test ($p = 0.23$) means the observed non-parallelism is consistent with what we would expect from chance alone.

    The researcher should conclude that there is insufficient evidence for an interaction at the chosen significance level. The non-parallelism in the plot does not represent a meaningful departure from additivity. Possible reasons include: (1) there truly is no interaction, (2) the sample size is too small to detect a real but weak interaction (low power), or (3) the interaction effect exists but is small relative to the within-group variability. The researcher might consider reporting effect sizes and confidence intervals for the interaction contrast rather than relying solely on the $p$-value.

---

**Exercise 5.**
Derive the condition under which the lines in a $2 \times 2$ interaction plot are exactly parallel. Start from the cell means $\mu_{11}, \mu_{12}, \mu_{21}, \mu_{22}$ and show the algebraic condition.

??? success "Solution to Exercise 5"
    The interaction plot has factor $B$ on the horizontal axis with levels $B_1$ and $B_2$, and separate lines for $A_1$ and $A_2$.

    - Line for $A_1$: passes through $(\mu_{11}, \mu_{12})$, with slope $\mu_{12} - \mu_{11}$.
    - Line for $A_2$: passes through $(\mu_{21}, \mu_{22})$, with slope $\mu_{22} - \mu_{21}$.

    The lines are parallel if and only if they have the same slope:

    $$
    \mu_{12} - \mu_{11} = \mu_{22} - \mu_{21}
    $$

    Rearranging:

    $$
    \mu_{11} - \mu_{12} - \mu_{21} + \mu_{22} = 0
    $$

    This is precisely the condition $(\alpha\beta)_{ij} = 0$ for all $i, j$ in the $2 \times 2$ case. Equivalently, the interaction contrast $\psi = \mu_{11} - \mu_{12} - \mu_{21} + \mu_{22}$ equals zero, which is the single degree-of-freedom interaction in a $2 \times 2$ design. $\square$
