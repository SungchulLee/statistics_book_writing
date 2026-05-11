# Mood's Median Test

**Mood's median test** is one of the simplest non-parametric tests for comparing two or more independent groups. It tests whether the groups share the same median by classifying each observation as above or below the grand median and then applying a chi-square test of independence to the resulting contingency table.

The test is conceptually straightforward and extremely robust to outliers, but it is generally less powerful than the [Kruskal-Wallis test](kruskal_wallis.md) because it discards all information about how far each observation falls from the median.

## Hypotheses

$$
H_0 \colon \text{All } k \text{ groups have the same median}
$$

$$
H_a \colon \text{At least one group has a different median}
$$

## Procedure

**Step 1.** Compute the **grand median** $\tilde{x}$ of the combined sample of size $N = \sum_{i=1}^{k} n_i$.

**Step 2.** For each group, count the number of observations above and below (or equal to) the grand median. This produces a $2 \times k$ contingency table:

|  | Group 1 | Group 2 | $\cdots$ | Group $k$ | Total |
|:-|:-------:|:-------:|:--------:|:---------:|:-----:|
| Above $\tilde{x}$ | $a_1$ | $a_2$ | $\cdots$ | $a_k$ | $A$ |
| Below $\tilde{x}$ | $b_1$ | $b_2$ | $\cdots$ | $b_k$ | $B$ |
| **Total** | $n_1$ | $n_2$ | $\cdots$ | $n_k$ | $N$ |

Observations exactly equal to $\tilde{x}$ are typically placed in the "below or equal" category, though conventions vary.

**Step 3.** Apply the **Pearson chi-square test** to the contingency table:

$$
\chi^2 = \sum_{i=1}^{k} \left[\frac{(a_i - E_{a_i})^2}{E_{a_i}} + \frac{(b_i - E_{b_i})^2}{E_{b_i}}\right]
$$

where the expected counts under $H_0$ are

$$
E_{a_i} = \frac{n_i \times A}{N}, \qquad E_{b_i} = \frac{n_i \times B}{N}
$$

**Step 4.** Under $H_0$, the test statistic follows approximately

$$
\chi^2 \sim \chi^2_{k-1}
$$

## Worked Example

Three diets are compared for weight loss (kg) over 8 weeks.

**Diet A** ($n_1 = 6$): 3.2, 4.5, 2.8, 5.1, 3.9, 4.2

**Diet B** ($n_2 = 6$): 1.5, 2.0, 3.0, 2.5, 1.8, 2.2

**Diet C** ($n_3 = 6$): 4.0, 5.5, 3.5, 6.0, 4.8, 5.0

**Step 1.** Combined sample (sorted): 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 3.9, 4.0, 4.2, 4.5, 4.8, 5.0, 5.1, 5.5, 6.0.

Grand median: $\tilde{x} = (3.2 + 3.5)/2 = 3.35$.

**Step 2.** Count above and below $\tilde{x}$:

|  | Diet A | Diet B | Diet C | Total |
|:-|:------:|:------:|:------:|:-----:|
| Above 3.35 | 4 | 0 | 5 | 9 |
| Below 3.35 | 2 | 6 | 1 | 9 |
| **Total** | 6 | 6 | 6 | 18 |

**Step 3.** Expected counts: each $E = 6 \times 9/18 = 3$.

$$
\chi^2 = \frac{(4-3)^2}{3} + \frac{(0-3)^2}{3} + \frac{(5-3)^2}{3} + \frac{(2-3)^2}{3} + \frac{(6-3)^2}{3} + \frac{(1-3)^2}{3}
$$

$$
= \frac{1 + 9 + 4 + 1 + 9 + 4}{3} = \frac{28}{3} \approx 9.33
$$

**Step 4.** Compare to $\chi^2_2$: $P(\chi^2_2 > 9.33) \approx 0.009$.

At $\alpha = 0.05$, we reject $H_0$. There is significant evidence that the diets differ in median weight loss. Diet B appears least effective (0 out of 6 above the grand median), while Diet C appears most effective (5 out of 6).

## Advantages and Limitations

| Advantage | Limitation |
|:----------|:-----------|
| Very simple to compute | Low power (discards magnitude information) |
| Extremely robust to outliers | Only detects differences in medians, not spread |
| Works with ordinal data | Chi-square approximation may be poor for small expected counts |
| Extends naturally to $k > 2$ groups | Generally outperformed by [Kruskal-Wallis](kruskal_wallis.md) |

!!! warning "Low power"
    Mood's median test has substantially lower power than the Kruskal-Wallis test. It should be used primarily when outlier resistance is paramount or when data are naturally dichotomized around a meaningful threshold. For most routine comparisons, the Kruskal-Wallis test is preferred.

## When to Use

- **Extreme outliers are present** and even rank-based tests might be influenced (though this is rare in practice).
- **Quick exploratory analysis** is needed and a formal rank-based test will follow.
- **The research question concerns the median specifically**, and the contingency table structure provides an intuitive summary.

## Summary

Mood's median test compares group medians by constructing a $2 \times k$ contingency table of counts above and below the grand median, then applying a chi-square test of independence. Its extreme simplicity and robustness to outliers are offset by low statistical power. For most applications, the [Kruskal-Wallis test](kruskal_wallis.md) provides a more powerful alternative, and Mood's median test is best reserved for situations where outlier resistance is the primary concern.


## Exercises

**Exercise 1.**
Describe the main concept of Mood's Median Test and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Mood's Median Test is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

---

**Exercise 2.**
State the key assumptions required by the method discussed here. How can each assumption be checked?

??? success "Solution to Exercise 2"
    The main assumptions typically include: (1) independence of observations -- verified by understanding the data collection process and checking for serial correlation; (2) distributional requirements (e.g., normality) -- checked with Q-Q plots and formal tests like Shapiro-Wilk; (3) equal variances (if applicable) -- assessed with boxplots and Levene's test. When assumptions are violated, consider robust alternatives, transformations, or nonparametric methods.

---

**Exercise 3.**
Work through a small numerical example illustrating the application of the technique from this section.

??? success "Solution to Exercise 3"
    A structured approach to applying this technique involves: (1) clearly stating the hypotheses or estimation goal; (2) verifying that the data meet the required assumptions; (3) computing the relevant test statistic, estimate, or model fit; (4) obtaining the p-value, confidence interval, or posterior distribution; (5) interpreting the result in the context of the original question. Following these steps systematically ensures a rigorous and reproducible analysis.

---

**Exercise 4.**
Compare the approach from this section with an alternative method. When would you choose each?

??? success "Solution to Exercise 4"
    The method discussed here is appropriate when its assumptions hold and the sample size is sufficient for the asymptotic approximations to be accurate. Alternative approaches include: (1) nonparametric methods -- preferred when distributional assumptions are suspect; (2) bootstrap methods -- useful when analytical reference distributions are unavailable; (3) Bayesian methods -- valuable when incorporating prior information or when direct probability statements about parameters are desired. Running multiple approaches and comparing results provides a useful robustness check.
