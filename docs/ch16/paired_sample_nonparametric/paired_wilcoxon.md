# Wilcoxon Signed-Rank Test for Paired Data

The [paired sign test](paired_sign.md) tests whether the median paired difference is zero but ignores how large each difference is. When the differences are measured on a meaningful numerical scale and their distribution is approximately symmetric, the **Wilcoxon signed-rank test for paired data** provides a more powerful alternative by incorporating both the sign and the rank of each absolute difference. This is the same [Wilcoxon signed-rank procedure](../one_sample_nonparametric/wilcoxon_signed_rank.md) applied to the differences $D_i = X_i - Y_i$.

## Assumptions

1. The pairs $(X_i, Y_i)$ are independent.
2. Each difference $D_i = X_i - Y_i$ comes from a continuous distribution.
3. The distribution of $D_i$ is **symmetric** about its median under $H_0$.

The symmetry requirement is the key additional assumption compared to the paired sign test. If the differences are noticeably skewed, the paired sign test or [paired permutation test](paired_permutation.md) may be more appropriate.

## Hypotheses

$$
H_0 \colon \text{The median of } D_i = X_i - Y_i \text{ is zero}
$$

$$
H_a \colon \text{The median of } D_i \ne 0 \quad \text{(two-sided)}
$$

One-sided alternatives ($H_a \colon \text{median} > 0$ or $< 0$) follow analogously.

## Procedure

**Step 1.** Compute paired differences $D_i = X_i - Y_i$.

**Step 2.** Exclude pairs where $D_i = 0$. Let $n$ be the number of remaining pairs.

**Step 3.** Rank the absolute differences $|D_1|, |D_2|, \ldots, |D_n|$ from smallest to largest, assigning midranks to ties.

**Step 4.** Compute the signed rank sums:

$$
W^+ = \sum_{\{i : D_i > 0\}} R_i, \qquad W^- = \sum_{\{i : D_i < 0\}} R_i
$$

Note that $W^+ + W^- = n(n+1)/2$.

**Step 5.** The test statistic is $T = \min(W^+, W^-)$ for a two-sided test. Equivalently, use $W^+$ and compare against its null distribution.

## Null Distribution

Under $H_0$ with symmetric differences, each sign assignment is equally likely. The null distribution of $W^+$ has

$$
\mu_{W^+} = \frac{n(n+1)}{4}, \qquad \sigma_{W^+}^2 = \frac{n(n+1)(2n+1)}{24}
$$

The normal approximation for large $n$:

$$
Z = \frac{W^+ - n(n+1)/4}{\sqrt{n(n+1)(2n+1)/24}}
$$

With a tie correction for $g$ groups of tied absolute differences of sizes $t_1, \ldots, t_g$:

$$
\sigma_{W^+}^2 = \frac{n(n+1)(2n+1)}{24} - \frac{1}{48}\sum_{j=1}^{g}(t_j^3 - t_j)
$$

## Worked Example

A company tests whether a training program improves employee productivity scores. Twelve employees are measured before and after the program.

| Employee | Before ($X_i$) | After ($Y_i$) | $D_i$ | $|D_i|$ | Rank | Signed Rank |
|:--------:|:------:|:------:|:-----:|:--------:|:----:|:-----------:|
| 1 | 45 | 52 | 7 | 7 | 5.5 | $+5.5$ |
| 2 | 38 | 41 | 3 | 3 | 2 | $+2$ |
| 3 | 50 | 48 | $-2$ | 2 | 1 | $-1$ |
| 4 | 42 | 49 | 7 | 7 | 5.5 | $+5.5$ |
| 5 | 55 | 60 | 5 | 5 | 3.5 | $+3.5$ |
| 6 | 48 | 53 | 5 | 5 | 3.5 | $+3.5$ |
| 7 | 41 | 50 | 9 | 9 | 8 | $+8$ |
| 8 | 53 | 45 | $-8$ | 8 | 7 | $-7$ |
| 9 | 47 | 58 | 11 | 11 | 9.5 | $+9.5$ |
| 10 | 44 | 55 | 11 | 11 | 9.5 | $+9.5$ |
| 11 | 50 | 62 | 12 | 12 | 11 | $+11$ |
| 12 | 46 | 59 | 13 | 13 | 12 | $+12$ |

**Compute rank sums:**

$$
W^+ = 5.5 + 2 + 5.5 + 3.5 + 3.5 + 8 + 9.5 + 9.5 + 11 + 12 = 70
$$

$$
W^- = 1 + 7 = 8
$$

**Check:** $W^+ + W^- = 70 + 8 = 78 = 12(13)/2$. $\checkmark$

**Test statistic:** $T = \min(70, 8) = 8$.

**Normal approximation** ($n = 12$):

$$
\mu_{W^+} = \frac{12 \times 13}{4} = 39
$$

$$
\sigma_{W^+} = \sqrt{\frac{12 \times 13 \times 25}{24}} = \sqrt{162.5} \approx 12.748
$$

Tie correction for two groups of size 2 (ranks 5.5 and ranks 3.5 and ranks 9.5): there are three tied groups each of size 2, so $\sum(t_j^3 - t_j) = 3(8 - 2) = 18$.

$$
\sigma_{W^+} = \sqrt{162.5 - \frac{18}{48}} = \sqrt{162.5 - 0.375} \approx 12.733
$$

$$
Z = \frac{70 - 39}{12.733} \approx 2.434
$$

$$
p = 2\,\mathcal{N}(-2.434) \approx 0.015
$$

At $\alpha = 0.05$, we reject $H_0$. There is significant evidence that the training program improved productivity scores.

## Why More Powerful than the Paired Sign Test

The paired sign test would count $n_+ = 10$ and $n_- = 2$ from the same data. It ignores the fact that the negative differences ($-2$ and $-8$) are generally smaller in magnitude than the positive ones. The Wilcoxon signed-rank test captures this by assigning the negative differences low ranks (1 and 7) while the positive differences accumulate higher ranks. This magnitude-aware weighting concentrates the evidence more effectively.

!!! note "ARE comparison"
    Under normal differences, the ARE of the paired Wilcoxon test relative to the paired $t$-test is $3/\pi \approx 0.955$. The ARE of the paired sign test is only $2/\pi \approx 0.637$. The Wilcoxon test thus requires about 5% more observations than the $t$-test to match its power, while the sign test requires about 57% more.

## Summary

The Wilcoxon signed-rank test for paired data ranks the absolute paired differences and weights each sign by its rank, producing a more powerful test than the paired sign test whenever the symmetry assumption holds. The procedure is identical to the one-sample Wilcoxon signed-rank test applied to the differences $D_i = X_i - Y_i$. For small samples, exact critical values are available; for larger samples, the normal approximation with tie correction provides reliable $p$-values.


## Exercises

**Exercise 1.**
Describe the main concept of Wilcoxon Signed-Rank Test for Paired Data and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Wilcoxon Signed-Rank Test for Paired Data is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
