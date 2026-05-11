# Paired Sign Test

When paired observations $(X_i, Y_i)$ are collected -- such as before-and-after measurements on the same subjects -- the natural approach is to reduce the data to differences $D_i = X_i - Y_i$ and then test whether the median difference is zero. The **paired sign test** applies the [sign test](../one_sample_nonparametric/sign_test.md) to these differences, using only the direction (positive or negative) of each difference and ignoring its magnitude.

Because the paired sign test makes essentially no distributional assumptions, it is the safest non-parametric choice for paired data. The trade-off is lower power compared to the [paired Wilcoxon signed-rank test](paired_wilcoxon.md), which additionally exploits magnitude information.

## Hypotheses

Let $D_i = X_i - Y_i$ for $i = 1, 2, \ldots, n$, and let $p = P(D_i > 0)$. Under $H_0$ the median difference is zero, which implies $p = 0.5$.

| Test type | $H_0$ | $H_a$ |
|:----------|:------|:------|
| Two-sided | $p = 0.5$ | $p \ne 0.5$ |
| Left-tailed | $p = 0.5$ | $p < 0.5$ (treatment worsens outcome) |
| Right-tailed | $p = 0.5$ | $p > 0.5$ (treatment improves outcome) |

## Procedure

**Step 1.** Compute the paired differences $D_i = X_i - Y_i$.

**Step 2.** Discard all pairs where $D_i = 0$. Let $n$ denote the number of remaining (non-zero) differences.

**Step 3.** Count the number of positive differences $n_+$ and negative differences $n_-$.

**Step 4.** Under $H_0$, the number of positive signs follows

$$
n_+ \sim \text{Binomial}(n, 0.5)
$$

**Step 5.** Compute the $p$-value:

- **Two-sided:** $p\text{-value} = 2\min\!\bigl(P(S \le n_+),\; P(S \ge n_+)\bigr)$ where $S \sim \text{Binomial}(n, 0.5)$.
- **Right-tailed:** $p\text{-value} = P(S \ge n_+)$.
- **Left-tailed:** $p\text{-value} = P(S \le n_+)$.

For large $n$, the normal approximation gives

$$
Z = \frac{n_+ - n/2}{\sqrt{n/4}}
$$

## Worked Example

A fitness program is evaluated by measuring resting heart rate (bpm) before and after a 12-week program for 10 participants.

| Participant | Before ($X_i$) | After ($Y_i$) | $D_i = X_i - Y_i$ | Sign |
|:-----------:|:------:|:------:|:----:|:----:|
| 1 | 72 | 68 | 4 | $+$ |
| 2 | 80 | 75 | 5 | $+$ |
| 3 | 68 | 70 | $-2$ | $-$ |
| 4 | 76 | 74 | 2 | $+$ |
| 5 | 85 | 78 | 7 | $+$ |
| 6 | 74 | 74 | 0 | (tie) |
| 7 | 90 | 82 | 8 | $+$ |
| 8 | 78 | 76 | 2 | $+$ |
| 9 | 82 | 79 | 3 | $+$ |
| 10 | 70 | 72 | $-2$ | $-$ |

**Step 1.** Differences computed above. One tie ($D_6 = 0$) is excluded.

**Step 2.** Remaining: $n = 9$, $n_+ = 7$, $n_- = 2$.

**Step 3.** Under $H_0$: $n_+ \sim \text{Binomial}(9, 0.5)$.

**Step 4.** Right-tailed $p$-value ($H_a$: the program reduces heart rate, so $D_i > 0$ is expected):

$$
P(S \ge 7) = \sum_{k=7}^{9} \binom{9}{k} (0.5)^9 = \binom{9}{7}(0.5)^9 + \binom{9}{8}(0.5)^9 + \binom{9}{9}(0.5)^9
$$

$$
= \frac{36 + 9 + 1}{512} = \frac{46}{512} \approx 0.090
$$

At $\alpha = 0.05$, we fail to reject $H_0$. Despite 7 out of 9 non-tied pairs showing improvement, the sample is too small for the sign test to reach significance. This illustrates the low power of the sign test with small samples.

!!! tip "Power consideration"
    The [paired Wilcoxon signed-rank test](paired_wilcoxon.md) applied to the same data would use the magnitudes of the differences (the large positive differences carry more weight), potentially yielding a smaller $p$-value. When the symmetry assumption is plausible, the paired Wilcoxon test is preferred.

## When to Use the Paired Sign Test

The paired sign test is appropriate when:

- **Only the direction of change is known.** For example, patients report "better" or "worse" without quantifying the degree of change.
- **The differences are ordinal.** If the measurement scale does not support meaningful arithmetic (e.g., Likert scales), magnitudes cannot be ranked reliably.
- **Symmetry is suspect.** The Wilcoxon signed-rank test assumes symmetry of the difference distribution. If the differences are highly skewed, the sign test avoids this assumption.

## Comparison with Alternatives

| Feature | Paired $t$-test | Paired Wilcoxon | Paired Sign Test |
|:--------|:----------------|:----------------|:-----------------|
| Assumption on differences | Normal | Symmetric, continuous | Continuous |
| Information used | Raw values | Signs + ranks | Signs only |
| Power (normal data) | Highest | High (ARE $\approx 0.955$) | Low (ARE $\approx 0.637$) |
| Robustness to outliers | Low | High | Very high |
| Applicable to ordinal data | No | Sometimes | Yes |

## Summary

The paired sign test reduces paired observations to a sequence of positive and negative signs and tests whether the median difference is zero using the binomial distribution. It requires no distributional assumptions beyond independence and continuity, making it the most robust paired-sample test. Its low power relative to the paired Wilcoxon test or paired $t$-test reflects the information discarded by ignoring magnitudes. When magnitude information is available and the symmetry assumption is reasonable, the [paired Wilcoxon signed-rank test](paired_wilcoxon.md) is generally preferred.


## Exercises

**Exercise 1.**
Describe the main concept of Paired Sign Test and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Paired Sign Test is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
