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
