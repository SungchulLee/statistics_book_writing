# When to Use Paired vs Two-Sample Tests

## Overview

Choosing between a paired-sample test and a two-sample test is a fundamental decision in hypothesis testing. The choice depends on the study design and how the data were collected, not on the data values themselves.

## Paired-Sample Tests

Use a **paired-sample test** when each observation in one group is naturally matched or linked to a specific observation in the other group. This pairing creates a dependency structure that must be accounted for in the analysis.

### Common Paired Designs

- **Before-and-after measurements**: The same subjects measured at two time points (e.g., blood pressure before and after treatment).
- **Matched subjects**: Participants paired on key characteristics (e.g., age, gender) with one receiving treatment and the other a placebo.
- **Repeated measures**: The same subjects tested under two different conditions (e.g., running speed with two different shoe brands).
- **Self-pairing**: Each subject serves as their own control (e.g., comparing left eye vs right eye measurements).

### Advantages of Paired Designs

- **Controls for individual variability**: By comparing each subject to themselves, between-subject variability is removed.
- **Greater statistical power**: Reducing variability makes it easier to detect true differences.
- **Smaller sample sizes needed**: Because of the increased power, fewer subjects are required.

### Key Indicator

If you can meaningfully compute a difference $d_i = X_i - Y_i$ for each pair, a paired test is appropriate.

---

## Two-Sample Tests

Use a **two-sample test** when the observations in the two groups are independent — there is no natural pairing between a specific observation in group 1 and a specific observation in group 2.

### Common Two-Sample Designs

- **Two independent groups**: Comparing means of men vs women, treatment group vs control group (different individuals).
- **Different populations**: Comparing average income in two countries using separate random samples.
- **Randomized experiments**: Subjects randomly assigned to one of two groups.

### Key Indicator

If the samples are drawn independently and there is no meaningful way to pair specific observations across groups, a two-sample test is appropriate.

---

## Decision Guide

| Question | Paired | Two-Sample |
|---|---|---|
| Same subjects measured twice? | ✓ | |
| Subjects matched on characteristics? | ✓ | |
| Independent groups with no pairing? | | ✓ |
| Can you compute a meaningful difference per pair? | ✓ | |
| Different sample sizes possible? | Rare | Common |

## Example Comparisons

**Paired**: A fitness coach measures body fat percentage of 10 participants before and after an 8-week workout program.

- Test: Paired t-test on $d_i = \text{Before}_i - \text{After}_i$
- Reason: Same participants measured at two time points

**Two-Sample**: Researchers compare average salaries of employees from Department A ($n=12$) vs Department B ($n=15$).

- Test: Two-sample t-test (or Welch's t-test)
- Reason: Different employees in each department, no natural pairing

## Common Mistake

A common mistake is to use a two-sample test when a paired test is appropriate. This ignores the correlation between paired observations, leading to a larger standard error and reduced statistical power. Always examine the study design carefully before selecting the test.

## Exercises

**Exercise 1.**
For each scenario, determine whether a paired or two-sample test is appropriate: (a) comparing reaction times of participants under caffeine vs. placebo in a crossover design, (b) comparing test scores of students in two different schools.

??? success "Solution to Exercise 1"
    **(a) Paired test.** In a crossover design, each participant is tested under both conditions (caffeine and placebo). The same person provides both measurements, creating natural pairs. A paired $t$-test on the within-subject differences is appropriate.

    **(b) Two-sample test.** Students in School A and School B are different individuals with no natural pairing. A two-sample $t$-test (or Welch's $t$-test if variances are unequal) is appropriate.

---

**Exercise 2.**
A study measures anxiety scores of 20 patients before and after therapy. The mean difference is $\bar{d} = -5.2$ with $s_d = 8.1$. Conduct a paired $t$-test of $H_0: \mu_d = 0$ at $\alpha = 0.05$.

??? success "Solution to Exercise 2"
    The test statistic is:

    $$
    t = \frac{\bar{d} - 0}{s_d / \sqrt{n}} = \frac{-5.2}{8.1/\sqrt{20}} = \frac{-5.2}{1.812} \approx -2.87
    $$

    With $df = 19$, the critical values for a two-sided test are $\pm t_{19, 0.025} = \pm 2.093$. Since $|t| = 2.87 > 2.093$, we **reject** $H_0$. The therapy produced a statistically significant reduction in anxiety scores.

---

**Exercise 3.**
Explain why the paired test is generally more powerful than the two-sample test when there is positive within-pair correlation.

??? success "Solution to Exercise 3"
    The paired test works with the differences $d_i = x_{1i} - x_{2i}$, whose variance is:

    $$
    \text{Var}(D) = \sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2
    $$

    When $\rho > 0$, the term $-2\rho\sigma_1\sigma_2$ reduces $\text{Var}(D)$ below the independent-sample variance $\sigma_1^2 + \sigma_2^2$. A smaller variance of differences produces a smaller standard error, a larger test statistic for the same true effect, and therefore higher power. The higher the within-pair correlation, the greater the power advantage of the paired test.

---

**Exercise 4.**
A researcher has matched pairs of twins assigned to two different treatments. However, 3 of the 15 pairs have one twin drop out, leaving unmatched data. Discuss the options for analyzing this data.

??? success "Solution to Exercise 4"
    The researcher has three options:

    1. **Analyze only complete pairs** ($n = 12$): Use a paired $t$-test on the 12 complete pairs. This is simple but discards data from the 3 incomplete pairs, reducing power.

    2. **Use a mixed approach**: Analyze the 12 complete pairs with a paired test and the 6 remaining individuals (3 from each group, if both groups lose a twin) with a two-sample test, then combine the results. This is complex and rarely done in practice.

    3. **Use a linear mixed model**: Fit a model that accounts for the twin-pair structure as a random effect. This approach can handle both complete and incomplete pairs, using all available data. This is the recommended modern approach as it maximizes power while correctly accounting for the paired structure.
