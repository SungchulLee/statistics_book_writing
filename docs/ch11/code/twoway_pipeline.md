# Two-Way Analysis of Variance End-to-End Pipeline

## Overview

Two-way ANOVA extends the one-way design by examining the simultaneous effects of two factors and their interaction on a continuous response. This page walks through a complete pipeline on the ToothGrowth dataset: fitting a two-way ANOVA (Type II), running Tukey HSD post-hoc tests for each main effect and the interaction, and producing an interaction plot. The two factors are supplement type (OJ vs. VC) and dose level (0.5, 1.0, 2.0).

## The Two-Way ANOVA Model

For factors $A$ (with $a$ levels) and $B$ (with $b$ levels), the cell-means model is

$$
y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}
$$

where $\alpha_i$ is the main effect of factor $A$, $\beta_j$ is the main effect of factor $B$, $(\alpha\beta)_{ij}$ is the interaction effect, and $\varepsilon_{ijk} \sim N(0, \sigma^2)$.

The Type II ANOVA table tests three null hypotheses:

| Source | $H_0$ | $df$ |
|---|---|---|
| Factor $A$ | All $\alpha_i = 0$ | $a - 1$ |
| Factor $B$ | All $\beta_j = 0$ | $b - 1$ |
| $A \times B$ | All $(\alpha\beta)_{ij} = 0$ | $(a-1)(b-1)$ |
| Residual | | $N - ab$ |

## Step 1: Fit the Model

```python
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

url = ('https://raw.githubusercontent.com/vincentarelbundock/'
       'Rdatasets/master/csv/datasets/ToothGrowth.csv')
df = pd.read_csv(url, usecols=[1, 2, 3])

model = ols('len ~ C(supp) + C(dose) + C(supp):C(dose)', data=df).fit()
aov2 = anova_lm(model, typ=2)
print(aov2)
```

Type II sums of squares test each main effect after adjusting for the other main effect but ignoring the interaction. This is recommended when the design is balanced or nearly balanced.

## Step 2: Tukey HSD for Main Effects

Post-hoc tests identify which levels of a factor differ. Run Tukey HSD separately for each main effect.

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

print(pairwise_tukeyhsd(endog=df['len'], groups=df['dose'], alpha=0.05))
print(pairwise_tukeyhsd(endog=df['len'], groups=df['supp'], alpha=0.05))
```

## Step 3: Tukey HSD for the Interaction

To compare all $a \times b$ cell means, create a combined grouping variable and run Tukey HSD on the interaction cells.

```python
df['supp_dose'] = df['supp'].astype(str) + "_" + df['dose'].astype(str)
print(pairwise_tukeyhsd(endog=df['len'], groups=df['supp_dose'], alpha=0.05))
```

With $a \times b = 2 \times 3 = 6$ cells, there are $\binom{6}{2} = 15$ pairwise comparisons. The Tukey procedure controls the family-wise error rate across all 15 simultaneously.

## Step 4: Interaction Plot

An interaction plot displays cell means with one factor on the horizontal axis and separate lines for each level of the other factor. Non-parallel lines suggest an interaction.

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot

fig, ax = plt.subplots(figsize=(8, 4))
interaction_plot(df['dose'], df['supp'], df['len'], ax=ax,
                 markers=['o', 's'], linestyles=['--', '-.'])
ax.set_title("Interaction: dose x supp")
ax.set_xlabel("dose")
ax.set_ylabel("len")
plt.tight_layout()
plt.show()
```

## Interpretation

- **Main effect of dose:** If the ANOVA $p$-value for dose is small and Tukey HSD shows significant pairwise differences, higher doses lead to greater tooth growth.
- **Main effect of supplement:** A significant $p$-value for supp indicates that the two supplement types (OJ vs. VC) produce different mean tooth lengths.
- **Interaction:** A significant interaction means the effect of dose depends on supplement type (or vice versa). In the ToothGrowth data, OJ and VC produce similar results at dose 2.0 but differ at lower doses, which appears as converging lines in the interaction plot.
- **Type II vs. Type III:** Type II is appropriate when there is no a priori reason to test main effects in the presence of interactions. If the interaction is significant and the design is unbalanced, Type III (which tests each effect controlling for all other effects including the interaction) may be preferred.

## Exercises

**Exercise 1.**
In a $2 \times 3$ factorial design with $n = 10$ observations per cell, state the degrees of freedom for each source in the ANOVA table and the total degrees of freedom.

??? success "Solution to Exercise 1"
    With $a = 2$ levels of factor $A$, $b = 3$ levels of factor $B$, and $n = 10$ per cell, $N = 2 \times 3 \times 10 = 60$.

    | Source | $df$ |
    |---|---|
    | Factor $A$ | $a - 1 = 1$ |
    | Factor $B$ | $b - 1 = 2$ |
    | $A \times B$ | $(a-1)(b-1) = 2$ |
    | Residual | $N - ab = 60 - 6 = 54$ |
    | Total | $N - 1 = 59$ |

---

**Exercise 2.**
Explain the difference between Type I, Type II, and Type III sums of squares. Under what conditions do they give identical results?

??? success "Solution to Exercise 2"

    - **Type I (sequential):** Each effect is tested after adjusting only for the effects entered before it. The results depend on the order of terms in the model.
    - **Type II:** Each main effect is tested after adjusting for the other main effect but not for the interaction. The interaction is tested after adjusting for both main effects.
    - **Type III:** Each effect is tested after adjusting for all other effects, including the interaction.

    All three types give identical results when the design is **balanced** (equal cell sizes) and the model is fully specified. In balanced designs, the sums of squares are orthogonal, so the order of entry does not matter and the adjustment for other terms has no effect. For unbalanced designs, the three types can differ substantially.

---

**Exercise 3.**
In the interaction plot, the lines for OJ and VC converge at dose 2.0. What does this imply about the interaction term? Write out the contrast that tests whether the OJ-VC difference is the same at dose 0.5 and dose 2.0.

??? success "Solution to Exercise 3"
    Convergence means the supplement effect diminishes at higher doses, which is a form of interaction. The contrast testing whether the OJ-VC difference is the same at doses 0.5 and 2.0 is

    $$
    \psi = (\mu_{\text{OJ},0.5} - \mu_{\text{VC},0.5}) - (\mu_{\text{OJ},2.0} - \mu_{\text{VC},2.0})
    $$

    Under $H_0: \psi = 0$, the supplement effect is the same at both doses. A significant result indicates that the magnitude of the OJ-VC difference changes across dose levels, which is exactly what the interaction term captures in the ANOVA model.

---

**Exercise 4.**
Show that the total sum of squares in a two-way ANOVA decomposes as

$$
SST = SS_A + SS_B + SS_{AB} + SSE
$$

for a balanced design. State the independence assumptions required for this decomposition.

??? success "Solution to Exercise 4"
    Start from the identity

    $$
    y_{ijk} - \bar{y}_{\cdot\cdot\cdot} = (\bar{y}_{i\cdot\cdot} - \bar{y}_{\cdot\cdot\cdot}) + (\bar{y}_{\cdot j\cdot} - \bar{y}_{\cdot\cdot\cdot}) + (\bar{y}_{ij\cdot} - \bar{y}_{i\cdot\cdot} - \bar{y}_{\cdot j\cdot} + \bar{y}_{\cdot\cdot\cdot}) + (y_{ijk} - \bar{y}_{ij\cdot})
    $$

    Squaring and summing over all $i, j, k$, all cross-product terms vanish due to orthogonality (which holds when the design is balanced), giving

    $$
    \sum_{i,j,k} (y_{ijk} - \bar{y}_{\cdot\cdot\cdot})^2 = bn\sum_i (\bar{y}_{i\cdot\cdot} - \bar{y}_{\cdot\cdot\cdot})^2 + an\sum_j (\bar{y}_{\cdot j\cdot} - \bar{y}_{\cdot\cdot\cdot})^2 + n\sum_{i,j}(\bar{y}_{ij\cdot} - \bar{y}_{i\cdot\cdot} - \bar{y}_{\cdot j\cdot} + \bar{y}_{\cdot\cdot\cdot})^2 + \sum_{i,j,k}(y_{ijk} - \bar{y}_{ij\cdot})^2
    $$

    That is, $SST = SS_A + SS_B + SS_{AB} + SSE$.

    The decomposition requires: (1) balanced design (equal $n$ per cell), and (2) the errors $\varepsilon_{ijk}$ are independent with common variance $\sigma^2$. Independence ensures that $SSE / \sigma^2 \sim \chi^2_{N-ab}$ and that $SSE$ is independent of $SS_A$, $SS_B$, and $SS_{AB}$, which is needed for the $F$-tests to have exact $F$-distributions. $\square$

---

**Exercise 5.**
When the interaction is significant but one main effect is not, some textbooks warn against interpreting the non-significant main effect. Explain why, using a concrete numerical example.

??? success "Solution to Exercise 5"
    A significant interaction means the effect of one factor depends on the level of the other. In this situation, the main effect -- which averages over the levels of the other factor -- may not represent any group's actual experience.

    **Example:** Consider a $2 \times 2$ design with cell means:

    | | $B_1$ | $B_2$ |
    |---|---|---|
    | $A_1$ | 10 | 20 |
    | $A_2$ | 20 | 10 |

    The marginal means of $A$ are $\bar{y}_{1\cdot} = 15$ and $\bar{y}_{2\cdot} = 15$, so the main effect of $A$ is zero. However, $A$ clearly has a large effect: it increases the response by 10 units in the $B_2$ condition and decreases it by 10 units in the $B_1$ condition. These opposite effects cancel out in the marginal mean, making the main effect test meaningless. The interaction is the informative quantity here, and the correct interpretation is that the direction of $A$'s effect reverses depending on the level of $B$.
