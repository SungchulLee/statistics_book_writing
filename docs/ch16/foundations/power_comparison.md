# Power Comparison with Parametric Tests

A natural question arises whenever a non-parametric test is chosen over its parametric counterpart: how much statistical power do we lose? If the parametric assumptions happen to hold, the rank-based test discards potentially useful information (the exact magnitudes), so we expect some efficiency loss. The key insight is that this loss is often surprisingly small, and when the assumptions fail, the non-parametric test can actually be *more* powerful.

This section formalizes the comparison using **asymptotic relative efficiency** (ARE), also known as **Pitman efficiency**, and summarizes the main results for the tests covered in this chapter.

## Power of a Test

Recall from [Chapter 9](../../ch09/index.md) that the **power** of a test is the probability of correctly rejecting the null hypothesis when a specific alternative is true:

$$
\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_a \text{ is true})
$$

Power depends on the sample size $n$, the significance level $\alpha$, the effect size $\delta$, and the choice of test statistic. Two tests applied to the same problem may differ in power, and comparing their power functions is the standard way to evaluate their relative performance.

## Asymptotic Relative Efficiency

The **asymptotic relative efficiency** (ARE) of test $B$ relative to test $A$ is defined as the limiting ratio of sample sizes needed to achieve the same power against a sequence of local alternatives converging to the null:

$$
\text{ARE}(B, A) = \lim_{n \to \infty} \frac{n_A}{n_B}
$$

where $n_A$ is the sample size required by test $A$ and $n_B$ is the sample size required by test $B$ to achieve the same power at the same significance level, as the effect size shrinks toward zero at rate $n^{-1/2}$.

An ARE of 0.95 means that test $B$ needs $n_B / 0.95 \approx 1.05 \, n_A$ observations to match the power of test $A$. An ARE greater than 1 means test $B$ is *more* efficient than test $A$.

!!! note "Pitman efficiency"
    The ARE defined above is often called **Pitman efficiency** because it evaluates tests against *contiguous alternatives* -- alternatives that approach the null as $n \to \infty$. This is the most widely used framework for comparing non-parametric and parametric tests.

## Key ARE Results

The following table summarizes the ARE of the most common non-parametric tests relative to their parametric counterparts, assuming the parametric model (normality) holds.

| Non-parametric test | Parametric counterpart | ARE (under normality) |
|:--------------------|:----------------------|:---------------------:|
| Wilcoxon signed-rank | One-sample $t$-test | $3/\pi \approx 0.955$ |
| Sign test | One-sample $t$-test | $2/\pi \approx 0.637$ |
| Wilcoxon rank-sum (Mann-Whitney) | Two-sample $t$-test | $3/\pi \approx 0.955$ |
| Kruskal-Wallis | One-way ANOVA $F$-test | $3/\pi \approx 0.955$ |
| Friedman | Repeated-measures ANOVA | $3/\pi \approx 0.955$ |
| Spearman's $r_s$ | Pearson's $r$ | $3/\pi \approx 0.955$ |

### Interpretation

The remarkable result is that under normality, the Wilcoxon-type rank tests lose only about 4.5% efficiency compared to the optimal parametric tests. The sign test, which uses only the direction of deviations and ignores magnitudes entirely, pays a heavier price at about 63.7% efficiency.

!!! tip "The 3/pi rule"
    The value $3/\pi \approx 0.955$ appears repeatedly because it is the ARE of any rank-based test that uses the Wilcoxon (linear rank) scores relative to the corresponding normal-theory test. This universality arises from the properties of the rank transformation applied to normal data.

## ARE Under Non-Normal Distributions

The ARE values above assume normality -- the best-case scenario for parametric tests. Under heavier-tailed distributions, the comparison reverses dramatically.

**Heavy-tailed distributions.** For data from a double-exponential (Laplace) distribution:

$$
\text{ARE}(\text{Wilcoxon signed-rank}, \; t\text{-test}) = \frac{3}{2} = 1.5
$$

This means the Wilcoxon test requires only two-thirds as many observations as the $t$-test to achieve the same power. The advantage grows further for even heavier tails.

**Contaminated normals.** A mixture $(1 - \varepsilon) \, \mathcal{N}(0,1) + \varepsilon \, \mathcal{N}(0, \sigma^2)$ with even modest contamination ($\varepsilon = 0.05$, $\sigma = 3$) can push the ARE of rank tests well above 1.

**Lower bound.** A fundamental result states that for any continuous distribution:

$$
\text{ARE}(\text{Wilcoxon rank-sum}, \; t\text{-test}) \ge 0.864
$$

This means the Wilcoxon test never loses more than about 14% efficiency relative to the $t$-test, regardless of the true distribution. There is no corresponding upper bound -- the ARE can be arbitrarily large for sufficiently heavy-tailed distributions.

!!! warning "No upper bound on ARE"
    While the worst-case efficiency loss of rank tests relative to $t$-tests is bounded and small ($\approx 14\%$), the potential gain under non-normality is unbounded. This asymmetry strongly favors non-parametric methods when the distributional assumption is uncertain.

## Finite-Sample Power

The ARE describes limiting behavior as $n \to \infty$. For finite samples, the actual power comparison depends on the specific sample size, effect size, and distribution. Monte Carlo simulations provide the most reliable finite-sample comparisons.

General patterns from simulation studies include:

- For $n \ge 20$ and moderate effect sizes, finite-sample power differences closely track the ARE predictions.
- For very small samples ($n < 10$), exact non-parametric tests (based on permutation distributions) can have *higher* power than parametric tests when the normality assumption is violated, even though the ARE comparison assumes large samples.
- The sign test, despite its low ARE of $2/\pi$ under normality, remains competitive when the underlying distribution is highly asymmetric or when only the direction of change is reliably measured.

## Practical Guidelines

The following decision principles emerge from the ARE analysis:

1. **Under confirmed normality**, parametric tests are preferred, but the power advantage over Wilcoxon-type tests is only about 4.5% -- often negligible in practice.
2. **Under uncertain normality**, non-parametric tests offer insurance: at most a 14% efficiency loss if normality holds, with potentially large gains if it does not.
3. **Under known non-normality**, non-parametric tests are often *more* powerful than their parametric counterparts, especially for heavy-tailed or contaminated distributions.
4. **The sign test** should be reserved for situations where only the direction of change is meaningful, or where the symmetry assumption of the Wilcoxon signed-rank test is suspect.

## Summary

The asymptotic relative efficiency provides a principled framework for comparing non-parametric and parametric tests. The Wilcoxon family of rank-based tests achieves an ARE of $3/\pi \approx 0.955$ under normality, meaning they lose less than 5% efficiency in the best case for parametric methods. Under non-normal distributions the ARE can exceed 1, making rank tests more powerful. The worst-case ARE for the Wilcoxon rank-sum test is bounded below by 0.864, ensuring that the efficiency cost of choosing a non-parametric test is always modest. These results provide strong justification for using rank-based methods when distributional assumptions are in doubt.

## Exercises

**Exercise 1.**
For each scenario below, state whether you would use a parametric or non-parametric test, name the specific test, and justify your choice.

**(a)** You want to compare the mean blood pressure of two groups (drug vs. placebo). Both groups have $n = 50$ observations, and Q-Q plots suggest approximate normality.

**(b)** You have 8 observations of customer satisfaction ratings (on a 1--5 Likert scale) from two store locations and want to test if the locations differ.

**(c)** You have paired before/after measurements for 12 subjects, but the differences are heavily right-skewed with one extreme outlier.

**(d)** You want to test whether three teaching methods produce different exam score distributions. Group sizes are 8, 10, and 7, and Shapiro-Wilk tests reject normality in two of the three groups.

??? success "Solution to Exercise 1"

    **(a)** **Parametric: two-sample $t$-test** (or Welch's $t$-test). With $n = 50$ per group and approximate normality confirmed by Q-Q plots, the conditions for a parametric test are well satisfied. The $t$-test will have higher power than a non-parametric alternative under these conditions.

    **(b)** **Non-parametric: Mann-Whitney U test** (Wilcoxon rank-sum test). Likert-scale data are ordinal, not continuous, so means and standard deviations are not meaningful. The small sample size ($n = 8$) and discrete nature of the data make non-parametric methods more appropriate.

    **(c)** **Non-parametric: Wilcoxon signed-rank test** (or even the sign test if symmetry of differences is in doubt). The heavy skewness and extreme outlier violate the normality assumption of the paired $t$-test. With only 12 observations, the CLT does not provide reliable normal approximations for highly skewed data. The Wilcoxon signed-rank test, based on ranks, is resistant to the outlier.

    **(d)** **Non-parametric: Kruskal-Wallis test**. Since normality is rejected in two of three groups, one-way ANOVA assumptions are violated. The sample sizes are relatively small (7--10), offering insufficient data for the CLT to compensate. The Kruskal-Wallis test does not require normality and is the appropriate multi-group comparison. If the Kruskal-Wallis test is significant, follow up with Dunn's test for pairwise comparisons.
