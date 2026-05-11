# Two-Way ANOVA: Main Effects and Blocking


## 1. Two-Way ANOVA (Analysis of Variance)

### A. Two-Way ANOVA

Two-Way ANOVA (Analysis of Variance) is an extension of One-Way ANOVA that allows researchers to examine the effects of two independent categorical variables (factors) on a continuous dependent variable. Unlike One-Way ANOVA, which assesses the mean differences across a single factor, Two-Way ANOVA can evaluate both the individual effect of each factor (main effects) and the interaction effect between them. This makes it especially useful for studies where two factors might influence the outcome, either independently or jointly.

The primary objective of Two-Way ANOVA is to assess:

1. **Main Effects**: The independent effect of each factor on the dependent variable. For instance, if studying the effects of "Diet" and "Exercise" on weight loss, Two-Way ANOVA would evaluate the effect of "Diet" alone and "Exercise" alone.
2. **Interaction Effect**: The combined effect of both factors on the dependent variable, which is not explained by either factor alone. An interaction effect occurs if the effect of one factor depends on the level of the other factor.

Two-Way ANOVA partitions the total variance in the data into three main components:

- **Variance due to Factor A**: This measures the variation in the dependent variable due to differences in the levels of the first factor.
- **Variance due to Factor B**: This captures the variation due to differences in the levels of the second factor.
- **Variance due to the Interaction between Factors A and B**: This represents the combined effect of both factors beyond their individual contributions.

In Two-Way ANOVA, an **F-statistic** is calculated for each source of variance: Factor A, Factor B, and the interaction between them. Each F-statistic assesses whether the corresponding variance component is significantly greater than the variance within the groups, as would be expected if the factor or interaction has a real effect.

**Assumptions of Two-Way ANOVA**

Like One-Way ANOVA, Two-Way ANOVA has key assumptions:

1. **Normality**: The data within each group should follow a normal distribution.
2. **Homogeneity of Variances**: The variances across all groups (combinations of factor levels) should be approximately equal.
3. **Independence**: Observations within and between groups should be independent.

Violations of these assumptions may lead to unreliable results. When assumptions are not met, alternative methods, such as non-parametric tests or adjustments like the use of a generalized linear model, may be appropriate.

**Advantages and Applications**

Two-Way ANOVA is widely used in fields like agriculture, psychology, and engineering, where researchers investigate multiple factors simultaneously. The key advantage of Two-Way ANOVA is its ability to identify interactions between factors, providing a more comprehensive understanding of how multiple variables influence the dependent variable.

**Post-Hoc Analysis**

If the Two-Way ANOVA indicates significant main effects or interaction effects, post-hoc tests may be conducted to explore specific differences between groups. For example, Tukey's HSD can help determine which levels within each factor differ, or if specific combinations of factor levels show unique effects.

### B. One-Way ANOVA vs. Two-Way ANOVA

Both One-Way ANOVA and Two-Way ANOVA are statistical techniques used to compare the means of different groups, but they differ in the number of factors they evaluate and the complexity of the analysis.

#### One-Way ANOVA

One-Way ANOVA is used to determine whether there are significant differences between the means of **three or more independent groups** based on a single categorical factor (independent variable).

- **Factors Analyzed**: Only one factor is considered (e.g., Fertilizer Type).
- **Purpose**: Tests whether any group mean is significantly different from the others.
- **F-Statistic**: Calculated as the ratio of the variance **between groups** to the variance **within groups**.
- **Interpretation**: A significant result indicates that at least one group mean is different, but it does not specify which groups differ. **Post-hoc tests** are often needed.
- **Assumptions**: Assumes normality, homogeneity of variances, and independence of observations.

#### Two-Way ANOVA

Two-Way ANOVA is an extension of One-Way ANOVA that allows for the analysis of two independent factors and their combined effect on the dependent variable.

- **Factors Analyzed**: Two factors are analyzed simultaneously.
- **Purpose**: Tests three effects: the main effect of the first factor, the main effect of the second factor, and the interaction effect between the two factors.
- **F-Statistics**: Separate F-statistics are calculated for each main effect and the interaction effect.
- **Interpretation**: Significant main effects indicate that at least one level of a factor differs from others, while a significant interaction effect suggests that the factors do not operate independently.
- **Assumptions**: Like One-Way ANOVA, it assumes normality, homogeneity of variances, and independence of observations.

#### Key Differences

| Feature               | One-Way ANOVA                                 | Two-Way ANOVA                                      |
|-----------------------|-----------------------------------------------|----------------------------------------------------|
| **Number of Factors** | One factor                                    | Two factors                                        |
| **Main Purpose**      | Determine if there are significant differences between group means based on one factor | Assess both main effects of each factor and their interaction effect |
| **Interaction Effect**| Not analyzed                                  | Analyzed, showing whether factors affect the outcome jointly |
| **F-Statistics**      | One F-statistic for the single factor         | Separate F-statistics for each main effect and interaction |
| **Example**           | Testing the effect of different fertilizers on plant height | Testing the effects of fertilizer and watering frequency on plant height |

#### When to Use Each Type

- **Use One-Way ANOVA** when you are interested in testing the effect of a single categorical factor on a continuous outcome variable and have three or more groups to compare.
- **Use Two-Way ANOVA** when you want to study the effects of two categorical factors on a continuous outcome and understand whether there's an interaction between these factors.

### C. Comparison of Null of Two-Way ANOVA to Linear Regression

$$\begin{array}{ccc}
&&\text{Linear Regression}&&\text{Two Way ANOVA}\\
\text{Model}&&y=\alpha+\beta_1 x_1+\beta_2 x_2 + \gamma x_1x_2
&&
y=\alpha+\sum_{i=1}^{a}\beta^{(1)}_i 1_{C^{(1)}_i}
+\sum_{j=1}^{b}\beta^{(2)}_j 1_{C^{(2)}_j}
+\sum_{i=1}^{a}\sum_{j=1}^{b}\beta_{ij} 1_{C^{(1)}_iC^{(2)}_j}\\
\text{H}_0&&\beta_1=0
&&\quad\text{all }\beta^{(1)}_{i}=0\\
\text{H}_0&&\beta_2=0
&&\quad\text{all }\beta^{(2)}_{j}=0\\
\text{H}_0&&\gamma=0&&\quad\text{all }\beta_{ij}=0\\
\end{array}$$

Both **Linear Regression** and **Two-Way ANOVA** evaluate the effects of two independent variables (or factors) on a dependent variable, but they differ in how the independent variables are treated. Linear regression deals with continuous predictors, whereas Two-Way ANOVA uses categorical variables, allowing for the analysis of main effects and interactions between factors.

In linear regression, when there are two continuous independent variables, $x_1$ and $x_2$, the model is expressed as:

$$
y = \alpha + \beta_1 x_1 + \beta_2 x_2 + \gamma x_1 x_2
$$

where $\alpha$ is the intercept, $\beta_1$ and $\beta_2$ represent the effects of $x_1$ and $x_2$ on $y$, and $\gamma$ represents the interaction effect between $x_1$ and $x_2$.

The hypotheses for this model are:

- $H_0: \beta_1 = 0$: There is no effect of $x_1$ on $y$
- $H_0: \beta_2 = 0$: There is no effect of $x_2$ on $y$
- $H_0: \gamma = 0$: There is no interaction between $x_1$ and $x_2$ on $y$

In contrast, **Two-Way ANOVA** deals with categorical independent variables (factors), and the model is represented as:

$$
y = \alpha + \sum_{i=1}^{a} \beta^{(1)}_i 1_{C^{(1)}_i} + \sum_{j=1}^{b} \beta^{(2)}_j 1_{C^{(2)}_j} + \sum_{i=1}^{a} \sum_{j=1}^{b} \beta_{ij} 1_{C^{(1)}_i C^{(2)}_j} + \text{noise}
$$

where $C^{(1)}_i$ are the $a$ levels of Factor A, $C^{(2)}_j$ are the $b$ levels of Factor B, $1_{C^{(1)}_i}$ and $1_{C^{(2)}_j}$ are indicator functions, $\beta^{(1)}_i$ represents the effect of Factor A (Main Effect A), $\beta^{(2)}_j$ represents the effect of Factor B (Main Effect B), and $\beta_{ij}$ represents the interaction effect.

The hypotheses in **Two-Way ANOVA** are:

- $H_0: \beta^{(1)}_i = 0$ for all $i$: No main effect of Factor A on $y$
- $H_0: \beta^{(2)}_j = 0$ for all $j$: No main effect of Factor B on $y$
- $H_0: \beta_{ij} = 0$ for all $i$ and $j$: No interaction between Factors A and B on $y$

## 2. Two Different Types of Two-Way ANOVA

### Two-Way ANOVA without Replication

In a **Two-Way ANOVA without Replication**, each combination of the levels of the two factors has only **one observation**.

**Characteristics:**

- **Single Observation per Cell**: Since each cell has only one data point, there is no way to estimate the variability within that cell.
- **Analysis of Main Effects and Interaction**: The test can still assess whether there are significant differences due to each factor's levels and if there's a significant interaction.
- **Limited Error Analysis**: Without replication, all variability that isn't explained by the main effects or interaction is lumped into a general term.

**Decomposition:**

$$
SST=SSA+SSB+SSAB
$$

### Two-Way ANOVA with Replication

In a **Two-Way ANOVA with Replication**, there are **multiple observations** for each combination of the levels of the two factors.

**Characteristics:**

- **Multiple Observations per Cell**: Having replication allows for the measurement of within-cell variability, or "error" variance.
- **Testing for Interaction Effects**: With replication, it's possible to detect and analyze interaction effects with greater accuracy.
- **Separation of Variance Components**: Two-Way ANOVA with replication partitions total variability into: Factor A, Factor B, Interaction, and Error (within-cell variability).
- **More Reliable Error Estimate**: By separating out within-cell variability, the error term is better estimated, improving reliability.

**Decomposition:**

$$
SST=SSA+SSB+SSAB+SSE
$$

### Summary Comparison

| Feature                     | Two-Way ANOVA without Replication       | Two-Way ANOVA with Replication      |
|-----------------------------|----------------------------------------|-------------------------------------|
| **Observations per Cell**   | One                                     | Multiple                            |
| **Error Term**              | Not separable (no within-group error)   | Separates within-group error        |
| **Main Effects**            | Can estimate                           | Can estimate                        |
| **Interaction Effects**     | Can estimate                           | Can estimate                        |
| **Reliability**             | Lower (no within-group variance)       | Higher (within-group variance)      |
| **Example**                 | One participant per condition          | Multiple participants per condition |

## 3. Assumptions of Two-Way ANOVA

### When to Use Two-Way ANOVA

Two-Way ANOVA is a powerful statistical method used to examine the effects of two independent categorical variables on a single continuous dependent variable. It also allows researchers to investigate whether there is an *interaction effect* between the two factors.

- **When There are Two Independent Variables (Factors)**: Two-Way ANOVA is most appropriate when the study involves two independent variables, and each of these variables can take on different levels. A crucial advantage over multiple one-way ANOVAs is that it can simultaneously assess the impact of both independent variables and their interaction.

- **When Exploring Interaction Effects**: An interaction occurs when the effect of one factor on the dependent variable changes depending on the level of the other factor.

- **When the Dependent Variable is Continuous**: Two-Way ANOVA requires the dependent variable to be measured on a continuous scale.

- **When Interested in Both Main and Interaction Effects**: Two-Way ANOVA provides insight into both the **main effects** of each independent variable and their **interaction effects**.

### Assumptions

**Independence of Observations**

- Each observation is independent of one another.
- This is typically ensured through the study's design (e.g., randomized controlled experiments).

**Normality**

- The dependent variable is assumed to be approximately normally distributed within each group.
- **How to Check**: Visual inspection using histograms or Q-Q plots; Shapiro-Wilk Test for formal assessment.

**Homogeneity of Variance (Homoscedasticity)**

- The variances of the dependent variable should be approximately equal across all groups.
- **How to Check**: Levene's Test or visual inspection with residual plots or box plots.
- **What to Do if Violated**: Log transformation of the dependent variable or using a more robust method such as Welch's ANOVA.

## 4. Limitations

- **Assumption Sensitivity**: As with One-Way ANOVA, the results of Two-Way ANOVA can be sensitive to violations of assumptions, especially homogeneity of variance.
- **Complexity with Interaction**: Interpreting the interaction effect can be challenging, especially if the interaction is significant but the main effects are not. Graphical methods (like interaction plots) are often used to interpret these effects.

## Exercises

**Exercise 1.**
Two factors: Training (online vs in-person) × Experience (junior vs senior), $n = 4$ per cell. Cell means: 12.5, 18.5, 15.0, 22.5. (a)-(d) Compute two-way ANOVA.

??? success "Solution to Exercise 1"
    (a) Row means: Online 15.5, In-person 18.75. Col means: Junior 13.75, Senior 20.5. Grand: 17.125.

    (b) $\mathrm{SSA}$ (Training) = 42.25, $\mathrm{SSB}$ (Experience) = 182.25, $\mathrm{SSAB}$ (Interaction) = 2.25, $\mathrm{SSW}$ = 20.0.

    (c) ANOVA table:

    | Source | SS | df | MS | F |
    |---|---|---|---|---|
    | Training | 42.25 | 1 | 42.25 | 25.35 |
    | Experience | 182.25 | 1 | 182.25 | 109.35 |
    | Interaction | 2.25 | 1 | 2.25 | 1.35 |
    | Within | 20.0 | 12 | 1.667 | |
    | Total | 246.75 | 15 | | |

    Critical $F_{1, 12, 0.05} = 4.75$. Both main effects significant; interaction not significant.

    (d) Effect of training method on productivity does NOT depend on experience level. In-person uniformly better, by about the same amount for juniors and seniors. **Additive** model is adequate.

---

**Exercise 2.**
**Interaction interpretation.** What does a significant interaction mean for marginal effects?

??? success "Solution to Exercise 2"
    Significant interaction: effect of factor A depends on level of factor B (and vice versa).

    **Marginal effects become misleading** when interaction is present:

    - "Drug A increases blood pressure by 10 on average" — but might increase by 20 in elderly and decrease by 5 in young.
    - The "marginal" effect averages over groups, hiding heterogeneity.

    **Best practice with significant interaction:**

    - Report **simple effects** (effect of A at each level of B).
    - Interpret marginal effects with caution.
    - Visualize interaction plot (each level of B as separate line).

    Without interaction (Exercise 1): marginal effects are accurate descriptions. Each factor effect is the same across other factor's levels.

---

**Exercise 3.**
**Crossed vs nested designs.**

??? success "Solution to Exercise 3"
    **Crossed:** every level of A appears with every level of B (Exercise 1: training × experience). Standard two-way ANOVA.

    **Nested:** levels of one factor are unique to each level of another (departments nested within schools — different departments per school). Different analysis: hierarchical/mixed-effects models.

    Crossed designs allow estimating interaction; nested designs don't (the "interaction" is confounded with factor B).

    Common confusion: random vs fixed factors. Random factors (e.g., subjects) introduce additional variance components. Use mixed models.

---

**Exercise 4.**
**Higher-order interactions.** Three-factor ANOVA has main effects, 2-way interactions, and 3-way interaction. What does the 3-way mean?

??? success "Solution to Exercise 4"
    3-way interaction (A × B × C): the 2-way A × B interaction depends on the level of C.

    Example: drug effect depends on disease type, but the disease-dependence varies by age group.

    **Higher-order interactions are hard to interpret.** Often suggests reorganizing the analysis:

    - Subset to specific level of C and reanalyze.
    - Look at the 2-way interaction within each level of C.

    Practical advice: with $k$ factors, the number of interactions grows as $2^k - k - 1$. Beyond 2-way, often work in regression framework with explicit terms rather than ANOVA decomposition.

---

**Exercise 5.**
**Sums-of-squares Type I/II/III** for unbalanced designs.

??? success "Solution to Exercise 5"
    For balanced designs, the orthogonality makes SS decomposition unambiguous. For unbalanced:

    **Type I (sequential):** SS for each factor adjusted for previously entered factors. Order-dependent.

    **Type II:** SS for each factor adjusted for other main effects, but not interactions. Symmetric across main effects.

    **Type III:** SS for each factor adjusted for everything else (including interactions). Most commonly reported. SAS default.

    R's `aov()` uses Type I; `car::Anova(type="III")` for Type III. For balanced designs, all types agree.

---

**Exercise 6.**
**Two-way vs separate one-way analyses.** Why is two-way preferred when both factors are studied?

??? success "Solution to Exercise 6"
    Two-way ANOVA:

    - **Estimates interaction** (cannot be done with separate one-ways).
    - **More efficient:** combines noise across cells for a better $\mathrm{MSW}$ estimate.
    - **Power:** smaller residual variance → easier to detect main effects.
    - **Control for confounding:** main effect of A is estimated holding B fixed.

    Separate one-ways:

    - Lose interaction information.
    - Inflated noise estimates (each separate analysis ignores other factor).
    - Potential confounding if factor levels are correlated.

    **Always use multi-way ANOVA (or regression) when multiple factors are manipulated.** This is the foundation of factorial experimental design (Fisher's seminal contribution).
