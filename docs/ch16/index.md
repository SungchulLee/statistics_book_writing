# Chapter 19: Non-Parametric Tests

## Overview

Non-parametric tests are statistical methods that do not assume a specific parametric form (such as normality) for the underlying population distribution. They are sometimes called **distribution-free tests** because their validity does not depend on the data following a particular distribution.

These methods are especially valuable when:

- The normality assumption required by parametric tests (t-tests, ANOVA) is violated.
- The data are ordinal (ranks) rather than interval or ratio scale.
- The sample size is too small to invoke the Central Limit Theorem.
- The data contain outliers that would distort parametric results.

## Key Idea — Ranks Replace Raw Values

Most non-parametric tests work by converting raw observations to **ranks** and then operating on those ranks. Because ranks are bounded and equi-spaced, extreme values (outliers) cannot exert the disproportionate influence they have on means and variances. The trade-off is a modest loss of statistical power when the parametric assumptions actually hold.

## Chapter Roadmap

| Section | Test | Parametric Counterpart | Use Case |
|:--------|:-----|:----------------------|:---------|
| 19.1 | Runs Test | — | Test randomness of a binary sequence |
| 19.1 | Sign Test | One-sample t-test | Test median with paired or single-sample data |
| 19.1 | Wilcoxon Signed-Rank Test | One-sample t-test | Test median using both sign and magnitude |
| 19.2 | Paired Sign Test | Paired t-test | Paired differences, direction only |
| 19.2 | Paired Wilcoxon Signed-Rank Test | Paired t-test | Paired differences, direction and magnitude |
| 19.3 | Wilcoxon Rank-Sum Test | Two-sample t-test | Compare two independent groups |
| 19.3 | Mann–Whitney U Test | Two-sample t-test | Compare two independent groups (with ties) |
| 19.3 | Kruskal–Wallis H Test | One-way ANOVA | Compare three or more independent groups |
| 19.3 | Mood's Median Test | One-way ANOVA | Compare medians of multiple groups |

## Choosing the Right Non-Parametric Test

```
Is the data a single binary sequence?
├── Yes → Runs Test (randomness)
└── No
    ├── One sample or paired?
    │   ├── Only signs matter → Sign Test
    │   └── Signs + magnitudes → Wilcoxon Signed-Rank Test
    └── Two or more independent samples?
        ├── Two groups
        │   ├── No ties → Wilcoxon Rank-Sum Test
        │   └── Ties present → Mann–Whitney U Test
        └── Three+ groups
            ├── Compare distributions → Kruskal–Wallis H Test
            └── Compare medians only → Mood's Median Test
```

## Prerequisites

- Chapter 9: Hypothesis Testing (null/alternative hypotheses, p-values, Type I/II errors)
- Chapter 5: Sampling Distributions (normal approximation)
- Chapter 15: Normality Tests (when to abandon parametric methods)
