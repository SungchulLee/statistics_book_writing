# Paired Permutation Test

The [paired sign test](paired_sign.md) and [paired Wilcoxon test](paired_wilcoxon.md) both make specific choices about what information to use from the paired differences: signs only, or signs plus ranks. The **paired permutation test** takes a different approach -- it uses the full numerical values of the differences and derives the null distribution by considering all possible ways the signs could have been assigned. This yields an exact, assumption-free test that exploits all available information.

## Intuition

Under $H_0$ (no treatment effect), each difference $D_i = X_i - Y_i$ is just as likely to be positive as negative, because swapping the labels "before" and "after" within a pair would flip the sign of $D_i$ without changing its magnitude. If we randomly reassign the signs of all $n$ differences, the resulting test statistic should look similar to the observed one. The permutation test formalizes this reasoning by comparing the observed statistic to the distribution of the statistic over all $2^n$ possible sign assignments.

## Hypotheses

$$
H_0 \colon \text{The distribution of } D_i \text{ is symmetric about zero}
$$

$$
H_a \colon \text{The distribution of } D_i \text{ is not symmetric about zero (two-sided)}
$$

One-sided alternatives follow by restricting attention to positive or negative shifts.

## Test Statistic

Let $D_1, D_2, \ldots, D_n$ be the observed paired differences. Any function of the differences can serve as the test statistic. Common choices include:

- **Mean of differences:** $\bar{D} = \frac{1}{n}\sum_{i=1}^{n} D_i$
- **Sum of differences:** $T = \sum_{i=1}^{n} D_i$

Both lead to equivalent tests since they are proportional. We use $T$ for simplicity.

## Permutation Distribution

Under $H_0$, each difference $D_i$ is equally likely to be $+|D_i|$ or $-|D_i|$. There are $2^n$ equally likely sign assignments $\mathbf{s} = (s_1, s_2, \ldots, s_n)$ where each $s_i \in \{-1, +1\}$. For each assignment, compute

$$
T(\mathbf{s}) = \sum_{i=1}^{n} s_i \, |D_i|
$$

The collection $\{T(\mathbf{s}) : \mathbf{s} \in \{-1, +1\}^n\}$ forms the **permutation distribution** of $T$ under $H_0$.

## Exact p-Value

The exact two-sided $p$-value is the proportion of sign assignments that produce a test statistic at least as extreme as the observed value:

$$
p = \frac{\#\{|\,T(\mathbf{s})| \ge |T_{\text{obs}}|\}}{2^n}
$$

For one-sided tests:

$$
p = \frac{\#\{T(\mathbf{s}) \ge T_{\text{obs}}\}}{2^n} \qquad \text{(right-tailed)}
$$

## Worked Example

A researcher tests whether a new study technique improves exam scores. Five students take two exams, one with the old method and one with the new.

| Student | Old ($Y_i$) | New ($X_i$) | $D_i = X_i - Y_i$ |
|:-------:|:------:|:------:|:-----:|
| 1 | 72 | 78 | 6 |
| 2 | 85 | 82 | $-3$ |
| 3 | 68 | 75 | 7 |
| 4 | 90 | 93 | 3 |
| 5 | 76 | 80 | 4 |

**Observed test statistic:** $T_{\text{obs}} = 6 + (-3) + 7 + 3 + 4 = 17$.

**Enumerate all $2^5 = 32$ sign assignments.** For each assignment $\mathbf{s} = (s_1, \ldots, s_5)$, compute $T(\mathbf{s}) = 6s_1 + 3s_2 + 7s_3 + 3s_4 + 4s_5$.

The absolute differences are $(6, 3, 7, 3, 4)$ and the maximum possible $T$ is $6 + 3 + 7 + 3 + 4 = 23$.

After enumerating all 32 assignments, we count those with $|T(\mathbf{s})| \ge 17$:

| $T(\mathbf{s})$ values $\ge 17$ | Sign assignment |
|:-:|:-:|
| 23 | $(+,+,+,+,+)$ |
| 17 | $(+,-,+,+,+)$ |
| 17 | $(+,+,+,-,+)$ |

And $T(\mathbf{s}) \le -17$:

| $T(\mathbf{s})$ values $\le -17$ | Sign assignment |
|:-:|:-:|
| $-23$ | $(-,-,-,-,-)$ |
| $-17$ | $(-,+,-,-,-)$ |
| $-17$ | $(-,-,-,+,-)$ |

Total: 6 out of 32 assignments yield $|T| \ge 17$.

$$
p = \frac{6}{32} = 0.1875
$$

At $\alpha = 0.05$, we fail to reject $H_0$. With only 5 pairs, the test lacks power to detect a moderate effect.

## Monte Carlo Approximation

When $n$ is large, the $2^n$ permutations become computationally infeasible to enumerate. A **Monte Carlo approximation** randomly samples $B$ sign assignments (typically $B = 10{,}000$ or more) and estimates the $p$-value as

$$
\hat{p} = \frac{\#\{|T(\mathbf{s}_b)| \ge |T_{\text{obs}}| : b = 1, \ldots, B\} + 1}{B + 1}
$$

The $+1$ in numerator and denominator ensures the $p$-value is never exactly zero and accounts for the observed data being one of the permutations.

!!! note "Accuracy of Monte Carlo p-values"
    With $B = 10{,}000$ random permutations, the Monte Carlo standard error of the estimated $p$-value is at most $\sqrt{0.25 / 10{,}000} = 0.005$. For most practical purposes, this precision is sufficient.

## Comparison with Other Paired Tests

| Feature | Paired $t$-test | Paired Wilcoxon | Paired Sign | Paired Permutation |
|:--------|:----------------|:----------------|:------------|:-------------------|
| Normality required | Yes | No | No | No |
| Symmetry required | Yes | Yes | No | Under $H_0$ only |
| Uses magnitudes | Yes | Via ranks | No | Yes (full values) |
| Exact $p$-values | No (approx.) | Small $n$ only | Yes | Yes |
| Power (normal data) | Highest | Very high | Low | Very high |

The paired permutation test uses the full numerical differences, giving it power comparable to the paired $t$-test without requiring normality. Its main limitation is computational: exact enumeration requires $2^n$ evaluations, though Monte Carlo sampling extends the method to arbitrarily large samples.

## Summary

The paired permutation test constructs the null distribution by enumerating (or randomly sampling) all $2^n$ possible sign reassignments of the observed paired differences. This yields exact, distribution-free $p$-values that exploit the full numerical information in the data. For small $n$, exact enumeration is feasible; for large $n$, Monte Carlo sampling provides accurate approximations. The test is especially valuable when neither the normality assumption of the paired $t$-test nor the symmetry assumption of the Wilcoxon signed-rank test is justified.


## Exercises

**Exercise 1.**
Describe the main concept of Paired Permutation Test and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Paired Permutation Test is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
