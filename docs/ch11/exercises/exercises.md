# Exercises

## Exercise 1: One-Way ANOVA Computation

A psychologist wants to determine whether test performance differs depending on the time of day. Three groups of students (morning, afternoon, and evening) take a standardized test, with the following scores:

| Morning | Afternoon | Evening |
|---------|-----------|---------|
| 85 | 76 | 93 |
| 92 | 81 | 95 |
| 89 | 82 | 97 |
| 90 | 80 | 92 |
| 87 | 78 | 94 |

**(a)** State the null and alternative hypotheses.

**(b)** Compute the grand mean $\bar{Y}_{\cdot\cdot}$ and the three group means $\bar{Y}_{1\cdot}$, $\bar{Y}_{2\cdot}$, $\bar{Y}_{3\cdot}$.

**(c)** Calculate the between-group sum of squares $\text{SS}_B$, the within-group sum of squares $\text{SS}_W$, and the total sum of squares $\text{SS}_T$. Verify that $\text{SS}_T = \text{SS}_B + \text{SS}_W$.

**(d)** Construct the ANOVA table with degrees of freedom, mean squares, and the F-statistic.

**(e)** Using a significance level of $\alpha = 0.05$, determine whether there is a statistically significant difference in test scores among the three time groups. State your conclusion in the context of the problem.

??? tip "Solution"

    **(a)** $H_0: \mu_1 = \mu_2 = \mu_3$ (the population mean test scores are the same for all three time groups). $H_a$: at least one group mean differs.

    **(b)** Group means:

    - Morning: $\bar{Y}_{1\cdot} = (85 + 92 + 89 + 90 + 87)/5 = 88.6$
    - Afternoon: $\bar{Y}_{2\cdot} = (76 + 81 + 82 + 80 + 78)/5 = 79.4$
    - Evening: $\bar{Y}_{3\cdot} = (93 + 95 + 97 + 92 + 94)/5 = 94.2$
    - Grand mean: $\bar{Y}_{\cdot\cdot} = (88.6 + 79.4 + 94.2)/3 = 87.4$

    **(c)** Between-group sum of squares:

    $$
    \text{SS}_B = 5[(88.6 - 87.4)^2 + (79.4 - 87.4)^2 + (94.2 - 87.4)^2] = 5[1.44 + 64.0 + 46.24] = 558.4
    $$

    Within-group sum of squares: compute $(Y_{ij} - \bar{Y}_{i\cdot})^2$ for each observation within its group.

    - Morning: $(85-88.6)^2 + (92-88.6)^2 + (89-88.6)^2 + (90-88.6)^2 + (87-88.6)^2 = 12.96 + 11.56 + 0.16 + 1.96 + 2.56 = 29.2$
    - Afternoon: $(76-79.4)^2 + (81-79.4)^2 + (82-79.4)^2 + (80-79.4)^2 + (78-79.4)^2 = 11.56 + 2.56 + 6.76 + 0.36 + 1.96 = 23.2$
    - Evening: $(93-94.2)^2 + (95-94.2)^2 + (97-94.2)^2 + (92-94.2)^2 + (94-94.2)^2 = 1.44 + 0.64 + 7.84 + 4.84 + 0.04 = 14.8$

    $$
    \text{SS}_W = 29.2 + 23.2 + 14.8 = 67.2
    $$

    Verification: $\text{SS}_T = 558.4 + 67.2 = 625.6$.

    **(d)** ANOVA table:

    | Source | SS | df | MS | F |
    |--------|------|-----|-------|-------|
    | Between | 558.4 | 2 | 279.2 | 49.85 |
    | Within | 67.2 | 12 | 5.6 | |
    | Total | 625.6 | 14 | | |

    The F-statistic is $F = 279.2 / 5.6 = 49.86$.

    **(e)** The critical value $F_{0.05, 2, 12} \approx 3.89$. Since $49.86 \gg 3.89$, we reject $H_0$ at $\alpha = 0.05$. There is strong evidence that the population mean test scores differ across the three time-of-day groups.

---

## Exercise 2: One-Way ANOVA with Post-Hoc Analysis

An exercise coach assigns 15 clients to three workout routines (5 per group): high-intensity interval training (HIIT), traditional strength training, and yoga. After ten weeks, weight loss (in pounds) is recorded:

| HIIT | Strength | Yoga |
|------|----------|------|
| 10 | 7 | 4 |
| 8 | 6 | 4 |
| 9 | 6 | 3 |
| 10 | 8 | 5 |
| 7 | 5 | 6 |

**(a)** Perform a one-way ANOVA at $\alpha = 0.05$. Construct the full ANOVA table and state your conclusion.

**(b)** If the ANOVA result is significant, apply Tukey's HSD test to identify which pairs of routines differ significantly. Use $q_{0.05, 3, 12}$ from the Studentized range distribution.

**(c)** Interpret the Tukey results in the context of the problem. Which workout routine(s) produce the most weight loss?

??? tip "Solution"

    **(a)** Group means: $\bar{Y}_{\text{HIIT}} = 8.8$, $\bar{Y}_{\text{Strength}} = 6.4$, $\bar{Y}_{\text{Yoga}} = 4.4$. Grand mean: $\bar{Y}_{\cdot\cdot} = 6.533$.

    $$
    \text{SS}_B = 5[(8.8 - 6.533)^2 + (6.4 - 6.533)^2 + (4.4 - 6.533)^2] = 5[5.138 + 0.018 + 4.551] = 48.53
    $$

    $$
    \text{SS}_W = (1.44+0.64+0.04+1.44+3.24) + (0.36+0.16+0.16+2.56+1.96) + (0.16+0.16+1.96+0.36+2.56) = 6.80 + 5.20 + 5.20 = 17.20
    $$

    | Source | SS | df | MS | F |
    |--------|-------|-----|-------|-------|
    | Between | 48.53 | 2 | 24.27 | 16.93 |
    | Within | 17.20 | 12 | 1.43 | |
    | Total | 65.73 | 14 | | |

    Since $F = 16.93 > F_{0.05, 2, 12} \approx 3.89$, we reject $H_0$. There is significant evidence that mean weight loss differs across workout routines.

    **(b)** Tukey's HSD: The critical difference is $\text{HSD} = q_{0.05, 3, 12} \times \sqrt{MS_W / n} = 3.77 \times \sqrt{1.43/5} = 3.77 \times 0.535 = 2.02$.

    Pairwise differences:

    - $|\bar{Y}_{\text{HIIT}} - \bar{Y}_{\text{Strength}}| = 2.4 > 2.02$ (significant)
    - $|\bar{Y}_{\text{HIIT}} - \bar{Y}_{\text{Yoga}}| = 4.4 > 2.02$ (significant)
    - $|\bar{Y}_{\text{Strength}} - \bar{Y}_{\text{Yoga}}| = 2.0 < 2.02$ (not significant)

    **(c)** HIIT produces significantly more weight loss than both strength training and yoga. The difference between strength training and yoga is not statistically significant at $\alpha = 0.05$. The coach has evidence that HIIT is the most effective routine among those tested, but cannot distinguish between strength training and yoga based on these data.

---

## Exercise 3: Assumption Checking

A researcher collects the following residuals from a one-way ANOVA with three groups ($n = 8$ per group):

| Group A residuals | Group B residuals | Group C residuals |
|-------------------|-------------------|-------------------|
| -2.1, 1.3, 0.8, -0.5 | -5.2, 3.8, 4.1, -3.9 | -0.3, 0.1, -0.2, 0.5 |
| 0.9, -1.2, 0.4, 0.4 | 2.7, -1.8, -2.1, 2.4 | 0.2, -0.4, 0.3, -0.2 |

**(a)** Compute the sample variance of the residuals within each group. Do the variances appear roughly equal?

**(b)** Based on the residual variances, would you expect Levene's test to reject the null hypothesis of equal variances? Explain.

**(c)** If homoscedasticity is violated, which alternative to the standard F-test would you recommend, and why?

**(d)** Looking at the residuals in Group C, they are noticeably smaller in magnitude than Groups A and B. If Group C also has a different (smaller) sample size, explain how this could affect the standard ANOVA F-test.

??? tip "Solution"

    **(a)** Sample variances of residuals:

    - Group A: $s_A^2 = \frac{1}{7}\sum(e_{Aj} - \bar{e}_A)^2$. Since residuals sum to approximately zero within each group, $\bar{e}_A \approx 0$. Thus $s_A^2 \approx \frac{1}{7}(4.41 + 1.69 + 0.64 + 0.25 + 0.81 + 1.44 + 0.16 + 0.16) = \frac{9.56}{7} \approx 1.37$.
    - Group B: $s_B^2 \approx \frac{1}{7}(27.04 + 14.44 + 16.81 + 15.21 + 7.29 + 3.24 + 4.41 + 5.76) = \frac{94.20}{7} \approx 13.46$.
    - Group C: $s_C^2 \approx \frac{1}{7}(0.09 + 0.01 + 0.04 + 0.25 + 0.04 + 0.16 + 0.09 + 0.04) = \frac{0.72}{7} \approx 0.10$.

    The variances are vastly different ($0.10$ vs. $1.37$ vs. $13.46$), so the equal-variance assumption does not hold.

    **(b)** Yes, Levene's test would almost certainly reject $H_0$ of equal variances, given that the largest group variance is more than 100 times the smallest.

    **(c)** Welch's ANOVA is recommended because it does not assume equal variances. It uses separate variance estimates for each group and adjusts the degrees of freedom accordingly, maintaining the correct Type I error rate under heteroscedasticity.

    **(d)** When a group with a smaller variance also has a smaller sample size, the pooled variance overestimates the variability for that group and underestimates it for the groups with larger variance. This causes the standard F-test to become liberal (rejecting $H_0$ more often than the nominal $\alpha$), inflating the Type I error rate.

---

## Exercise 4: Two-Way ANOVA

A company tests two factors that may affect employee productivity (units produced per hour): **Training Method** (A: online, B: in-person) and **Experience Level** (1: junior, 2: senior). Four employees are observed in each combination:

|  | Junior | Senior |
|--|--------|--------|
| Online | 12, 14, 11, 13 | 18, 20, 17, 19 |
| In-person | 15, 16, 14, 15 | 22, 24, 21, 23 |

**(a)** Compute the cell means, row (Training) means, column (Experience) means, and the grand mean.

**(b)** Calculate $\text{SS}_A$ (Training), $\text{SS}_B$ (Experience), $\text{SS}_{AB}$ (Interaction), and $\text{SS}_W$ (Within-cell error).

**(c)** Construct the two-way ANOVA table and test each effect at $\alpha = 0.05$.

**(d)** Is there a significant interaction between training method and experience level? What does this mean in practical terms?

??? tip "Solution"

    **(a)** Cell means:

    - Online, Junior: $\bar{Y}_{A1} = 12.5$
    - Online, Senior: $\bar{Y}_{A2} = 18.5$
    - In-person, Junior: $\bar{Y}_{B1} = 15.0$
    - In-person, Senior: $\bar{Y}_{B2} = 22.5$

    Row means: $\bar{Y}_{A\cdot} = (12.5 + 18.5)/2 = 15.5$, $\bar{Y}_{B\cdot} = (15.0 + 22.5)/2 = 18.75$.

    Column means: $\bar{Y}_{\cdot 1} = (12.5 + 15.0)/2 = 13.75$, $\bar{Y}_{\cdot 2} = (18.5 + 22.5)/2 = 20.5$.

    Grand mean: $\bar{Y}_{\cdot\cdot} = (12.5 + 18.5 + 15.0 + 22.5)/4 = 17.125$.

    **(b)**

    $$
    \text{SS}_A = bn \sum_{i}(\bar{Y}_{i\cdot} - \bar{Y}_{\cdot\cdot})^2 = 2 \times 4 \times [(15.5 - 17.125)^2 + (18.75 - 17.125)^2] = 8 \times [2.641 + 2.641] = 42.25
    $$

    $$
    \text{SS}_B = an \sum_{j}(\bar{Y}_{\cdot j} - \bar{Y}_{\cdot\cdot})^2 = 2 \times 4 \times [(13.75 - 17.125)^2 + (20.5 - 17.125)^2] = 8 \times [11.39 + 11.39] = 182.25
    $$

    $$
    \text{SS}_{AB} = n \sum_{ij}(\bar{Y}_{ij} - \bar{Y}_{i\cdot} - \bar{Y}_{\cdot j} + \bar{Y}_{\cdot\cdot})^2
    $$

    Computing each cell's interaction term: for example, Online-Junior: $12.5 - 15.5 - 13.75 + 17.125 = 0.375$. All four cells yield $(\pm 0.375)^2 = 0.141$, so $\text{SS}_{AB} = 4 \times 4 \times 0.141 = 2.25$.

    Within-cell SS: sum of squared deviations from cell means across all 16 observations. Each cell has variance approximately $1.67$, so $\text{SS}_W = 12 \times 1.67 = 20.0$.

    **(c)** ANOVA table (with $a=2, b=2, n=4$):

    | Source | SS | df | MS | F |
    |--------|--------|-----|-------|-------|
    | Training (A) | 42.25 | 1 | 42.25 | 25.35 |
    | Experience (B) | 182.25 | 1 | 182.25 | 109.35 |
    | Interaction (AB) | 2.25 | 1 | 2.25 | 1.35 |
    | Within | 20.0 | 12 | 1.667 | |
    | Total | 246.75 | 15 | | |

    Critical value: $F_{0.05, 1, 12} = 4.75$. Both main effects are significant ($F_A = 25.35 > 4.75$, $F_B = 109.35 > 4.75$). The interaction is not significant ($F_{AB} = 1.35 < 4.75$).

    **(d)** The interaction is not significant, meaning the effect of training method on productivity does not depend on experience level. In practical terms, in-person training leads to higher productivity than online training by roughly the same amount for both junior and senior employees. Similarly, senior employees outperform junior employees by a similar margin regardless of training method.

---

## Exercise 5: Post-Hoc Test Selection

A marketing analyst compares click-through rates across four advertisement designs. The one-way ANOVA yields $F = 5.12$ with $p = 0.008$, indicating at least one design differs from the others.

**(a)** The analyst wants to compare every design with every other design. Which post-hoc test is most appropriate? Justify your choice.

**(b)** The analyst's manager only cares about how each new design (B, C, D) compares to the current design (A). Which post-hoc test is more appropriate in this case, and why is it preferred over the method in part (a)?

**(c)** A colleague points out that the group variances are $s_A^2 = 2.1$, $s_B^2 = 8.7$, $s_C^2 = 3.0$, $s_D^2 = 9.2$, and the sample sizes are $n_A = 30$, $n_B = 12$, $n_C = 25$, $n_D = 10$. Does this change your recommendation? Which test should be used now?

??? tip "Solution"

    **(a)** Tukey's HSD is most appropriate for all pairwise comparisons. It controls the family-wise error rate (FWER) at $\alpha$ while being specifically designed for pairwise comparisons, making it more powerful than Bonferroni or Scheffe for this purpose.

    **(b)** Dunnett's test is more appropriate when comparing each treatment to a single control. It controls the FWER while making only $k - 1 = 3$ comparisons instead of $\binom{4}{2} = 6$, giving it greater statistical power than Tukey's HSD for this specific comparison structure.

    **(c)** Yes, this changes the recommendation. The group variances differ substantially (the largest is more than 4 times the smallest), and the sample sizes are unequal. Under these conditions, Games-Howell is the appropriate post-hoc test because it does not assume equal variances or equal sample sizes. It uses separate variance estimates and Welch-Satterthwaite degrees of freedom for each pairwise comparison.

---

## Exercise 6: Welch's ANOVA

Three investment strategies are compared based on monthly returns (%). The data show:

| Strategy | $n$ | $\bar{Y}$ | $s^2$ |
|----------|-----|-----------|-------|
| Momentum | 36 | 1.8 | 12.5 |
| Value | 24 | 1.2 | 3.1 |
| Index | 48 | 1.0 | 5.8 |

**(a)** Explain why the standard one-way ANOVA F-test may be inappropriate for these data.

**(b)** Welch's ANOVA uses the test statistic

$$
F_W = \frac{\sum_{i=1}^{k} w_i (\bar{Y}_i - \tilde{Y})^2 / (k-1)}{1 + \frac{2(k-2)}{k^2-1} \sum_{i=1}^{k} \frac{(1 - w_i/\sum w_j)^2}{n_i - 1}}
$$

where $w_i = n_i / s_i^2$ and $\tilde{Y} = \sum w_i \bar{Y}_i / \sum w_i$. Compute the weights $w_1, w_2, w_3$ and the weighted grand mean $\tilde{Y}$.

**(c)** Without completing the full calculation, explain conceptually why Welch's approach gives more weight to groups with smaller variances.

??? tip "Solution"

    **(a)** The group variances differ substantially: $s_1^2 = 12.5$, $s_2^2 = 3.1$, $s_3^2 = 5.8$. The largest variance is about four times the smallest. Combined with the unequal sample sizes ($n = 36, 24, 48$), the standard F-test's assumption of homoscedasticity is violated. The pooled variance estimate would not accurately represent any single group's variability.

    **(b)** Weights: $w_1 = 36/12.5 = 2.88$, $w_2 = 24/3.1 = 7.74$, $w_3 = 48/5.8 = 8.28$.

    Sum of weights: $\sum w_i = 2.88 + 7.74 + 8.28 = 18.90$.

    Weighted grand mean:

    $$
    \tilde{Y} = \frac{2.88 \times 1.8 + 7.74 \times 1.2 + 8.28 \times 1.0}{18.90} = \frac{5.184 + 9.288 + 8.280}{18.90} = \frac{22.752}{18.90} = 1.204
    $$

    **(c)** The weight $w_i = n_i / s_i^2$ is inversely proportional to the group's variance. Groups with smaller variances provide more precise estimates of their population means, so they receive greater weight in the analysis. This is analogous to weighted least squares, where observations with lower variance contribute more to the estimate. The Momentum strategy, despite having the largest sample size, receives the lowest weight because its high variance ($s^2 = 12.5$) makes its sample mean a less reliable estimate of its population mean.

---

## Exercise 7: Bonferroni vs. Scheffe

A researcher performs a one-way ANOVA with $k = 5$ groups and $n = 10$ observations per group, yielding $\text{MS}_W = 4.0$. The researcher wants to test the following three planned contrasts at $\alpha = 0.05$:

- $C_1$: $\mu_1 - \mu_2 = 0$ (comparing groups 1 and 2)
- $C_2$: $\mu_3 - \frac{1}{2}(\mu_4 + \mu_5) = 0$ (comparing group 3 to the average of groups 4 and 5)
- $C_3$: $\mu_4 - \mu_5 = 0$ (comparing groups 4 and 5)

**(a)** What is the Bonferroni-adjusted significance level for each contrast?

**(b)** For $C_1$, suppose $\bar{Y}_1 = 12.0$ and $\bar{Y}_2 = 9.5$. Compute the test statistic and determine whether the contrast is significant using the Bonferroni correction.

**(c)** Would Scheffe's method be more or less powerful than Bonferroni for these three specific contrasts? Explain.

??? tip "Solution"

    **(a)** With $m = 3$ planned contrasts and $\alpha = 0.05$, the Bonferroni-adjusted level is $\alpha^* = 0.05 / 3 = 0.0167$ per contrast.

    **(b)** For contrast $C_1: \mu_1 - \mu_2$, the test statistic is

    $$
    t = \frac{\bar{Y}_1 - \bar{Y}_2}{\sqrt{\text{MS}_W \left(\frac{1}{n_1} + \frac{1}{n_2}\right)}} = \frac{12.0 - 9.5}{\sqrt{4.0 \times (1/10 + 1/10)}} = \frac{2.5}{\sqrt{0.8}} = \frac{2.5}{0.894} = 2.80
    $$

    The critical value from $t_{45}$ at $\alpha^*/2 = 0.0083$ is approximately $t_{0.0083, 45} \approx 2.50$. Since $2.80 > 2.50$, the contrast is significant: groups 1 and 2 differ at the Bonferroni-corrected level.

    **(c)** Scheffe's method would be less powerful for these three specific contrasts. Scheffe controls the FWER for all possible contrasts (not just the three being tested), so its critical value is determined by $\sqrt{(k-1) F_{0.05, k-1, N-k}} = \sqrt{4 \times F_{0.05, 4, 45}} \approx \sqrt{4 \times 2.58} = 3.21$. This is larger than the Bonferroni critical value of approximately 2.50. Bonferroni is more powerful when the number of planned contrasts is small relative to the total number of possible contrasts.
