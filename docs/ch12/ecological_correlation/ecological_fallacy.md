# Ecological Fallacy

When correlations are computed on aggregated data (averages over groups, regions, or time periods), the resulting **ecological correlations** can be dramatically different from the correlations that exist at the individual level. Drawing conclusions about individuals based on group-level correlations is known as the **ecological fallacy**. It is one of the most common and consequential errors in applied statistics.

---

## Definition

The **ecological fallacy** occurs when a statistical relationship observed between variables at an aggregate level (e.g., countries, states, schools) is incorrectly assumed to hold at the individual level.

More precisely, if we observe a correlation between group-level averages $\bar{X}_g$ and $\bar{Y}_g$ across $g = 1, \ldots, G$ groups, it does not follow that the same correlation exists between $X_i$ and $Y_i$ at the individual level. The ecological correlation can be much stronger, weaker, or even opposite in sign to the individual-level correlation.

---

## Robinson's Paradox

The classic demonstration of the ecological fallacy comes from William S. Robinson's 1950 paper. Robinson examined the relationship between the percentage of foreign-born residents and literacy rates across U.S. states.

- **Ecological correlation** (state-level): $r \approx 0.53$ -- states with higher percentages of foreign-born residents tended to have *higher* literacy rates.
- **Individual correlation**: $r \approx -0.11$ -- foreign-born individuals actually tended to have *lower* literacy rates than native-born individuals.

The paradox arises because immigrants tended to settle in states with high overall literacy (e.g., New York, California), creating a positive ecological correlation. But within those states, immigrants individually tended to have lower literacy rates than the native-born population.

!!! warning "Ecological correlations can reverse sign"
    Robinson's example shows that the ecological correlation ($+0.53$) was not merely larger than the individual correlation ($-0.11$); it had the opposite sign. This is not an edge case but a systematic phenomenon that can occur whenever group composition varies.

---

## Why It Happens

The ecological fallacy arises from the mathematical relationship between group-level and individual-level correlations. When data are aggregated into groups, two sources of variation contribute to the ecological correlation:

1. **Between-group variation**: differences in group means.
2. **Within-group variation**: differences among individuals within the same group.

Aggregation eliminates within-group variation, leaving only between-group variation. Since the between-group and within-group relationships can differ in magnitude and sign, the ecological correlation can diverge substantially from the individual-level correlation.

Formally, the total correlation can be decomposed as a weighted combination of between-group and within-group components. The ecological correlation reflects only the between-group component, which can dominate or contradict the within-group pattern.

---

## A Numerical Illustration

Consider two groups, each with five individuals:

**Group A:**

| Individual | $X$ | $Y$ |
|:---:|:---:|:---:|
| 1 | 10 | 8 |
| 2 | 12 | 6 |
| 3 | 14 | 4 |
| 4 | 16 | 2 |
| 5 | 18 | 0 |

**Group B:**

| Individual | $X$ | $Y$ |
|:---:|:---:|:---:|
| 6 | 20 | 18 |
| 7 | 22 | 16 |
| 8 | 24 | 14 |
| 9 | 26 | 12 |
| 10 | 28 | 10 |

Within each group, $X$ and $Y$ have a **perfect negative** correlation ($r = -1$). However, the group means are $(\bar{X}_A, \bar{Y}_A) = (14, 4)$ and $(\bar{X}_B, \bar{Y}_B) = (24, 14)$. Across the two group means, the ecological correlation is **perfectly positive** ($r = +1$): higher average $X$ goes with higher average $Y$.

If we only looked at the ecological data (two points: the group averages), we would conclude a strong positive relationship. The individual data tell the opposite story.

---

## Common Settings for the Ecological Fallacy

The ecological fallacy appears frequently in:

- **Public health**: correlating country-level fat consumption with cancer rates does not imply that individuals who eat more fat get more cancer.
- **Education**: correlating school-level spending with test scores does not mean that individual students benefit proportionally from spending.
- **Political science**: correlating district-level demographics with voting patterns does not reveal individual voting behavior.
- **Economics**: correlating national GDP with life satisfaction does not tell us whether richer individuals within a country are happier.

---

## How to Avoid the Ecological Fallacy

1. **Use individual-level data** whenever possible. If the research question concerns individuals, analyze individual observations.

2. **Be explicit about the level of analysis.** Clearly state whether correlations are computed at the individual, group, or ecological level.

3. **Use multilevel models.** Hierarchical/multilevel regression models can simultaneously account for both within-group and between-group variation, avoiding the conflation of levels.

4. **Report both levels.** When only aggregate data are available, acknowledge the limitation and avoid drawing individual-level conclusions.

---

## Connection to Simpson's Paradox

The ecological fallacy is closely related to [Simpson's paradox](simpsons_paradox.md), where a trend that appears in aggregated data reverses when the data are disaggregated by a confounding variable. Both phenomena illustrate that aggregation can distort or reverse the direction of an association. The ecological fallacy focuses specifically on the danger of inferring individual behavior from group averages, while Simpson's paradox is a broader phenomenon involving conditional versus marginal relationships.

---

## Summary

The ecological fallacy occurs when correlations observed at the group level are wrongly attributed to individuals. Robinson's paradox demonstrates that ecological correlations can even reverse sign compared to individual-level correlations. This happens because aggregation eliminates within-group variation, which may carry a different signal than between-group variation. To avoid the ecological fallacy, analyze data at the appropriate level and use multilevel models when group structure is present.

## Exercises

**Exercise 1.**
Define the ecological fallacy and explain why conclusions about groups cannot be directly applied to individuals.

??? success "Solution to Exercise 1"
    The **ecological fallacy** is the error of inferring individual-level relationships from group-level (aggregate) data. It occurs because correlations or associations observed at the group level do not necessarily hold at the individual level.

    Group-level data reflect between-group variation only, while individual behavior involves both between-group and within-group variation. The within-group relationship may differ in magnitude or even direction from the between-group relationship.

    **Example:** Countries with higher average chocolate consumption have more Nobel laureates per capita (ecological correlation). Concluding that individuals who eat more chocolate are more likely to win Nobel prizes is the ecological fallacy. The country-level association may be driven by wealth (rich countries both consume more chocolate and fund more research).

---

**Exercise 2.**
Construct a numerical example where the ecological correlation is positive but the individual-level correlation is negative.

??? success "Solution to Exercise 2"
    Consider two groups:

    - **Group A** (wealthy city): individuals' $X$ values average 80, $Y$ values average 90. Within the group, higher $X$ is associated with lower $Y$ (individual $r = -0.5$).
    - **Group B** (poor city): individuals' $X$ values average 20, $Y$ values average 30. Within the group, higher $X$ is also associated with lower $Y$ (individual $r = -0.5$).

    At the group level: Group A has higher $\bar{X}$ (80 vs 20) and higher $\bar{Y}$ (90 vs 30). The ecological correlation between group means is $r = +1$.

    This is a Simpson's paradox structure: within each group, the relationship is negative, but the between-group trend is positive because the groups differ systematically in both variables. Analyzing only the group-level data would produce the wrong sign.

---

**Exercise 3.**
A newspaper reports: "States with higher average income have lower crime rates, proving that higher income reduces crime." Identify the ecological fallacy and suggest a better analysis.

??? success "Solution to Exercise 3"
    The ecological fallacy is inferring that higher income *for individuals* reduces their likelihood of committing crimes, based on *state-level* aggregate data. The state-level correlation conflates multiple mechanisms:

    - States with higher average income may have better policing, education, and social services (institutional factors).
    - The relationship between individual income and individual criminality may be different from the state-level pattern.
    - High-income states may have high income inequality, which could increase crime even though the average is high.

    **Better analysis:** Use individual-level data linking personal income to criminal behavior, controlling for confounders like education, age, neighborhood characteristics, and employment status. Panel data (tracking individuals over time) or natural experiments (e.g., lottery winners) would provide stronger causal evidence.

---

**Exercise 4.**
Explain the concept of the "modifiable areal unit problem" (MAUP) and how it relates to the ecological fallacy.

??? success "Solution to Exercise 4"
    The **Modifiable Areal Unit Problem** (MAUP) is the observation that statistical results based on aggregated spatial data depend on how the geographic units are defined. It has two components:

    1. **Scale effect:** Changing the level of aggregation (e.g., census tracts vs. counties vs. states) changes the correlation. Larger units show stronger ecological correlations because more individual-level variation is averaged away.

    2. **Zonation effect:** At the same scale, different ways of drawing boundaries produce different results. Gerrymandering is an extreme example.

    MAUP relates to the ecological fallacy because both involve the loss of information from aggregation. The ecological fallacy is the interpretive error of treating aggregate results as individual-level truths. MAUP shows that even the aggregate results themselves are not stable -- they depend on arbitrary choices about the units of analysis. Together, they demonstrate that aggregate data are fundamentally limited for individual-level inference.
