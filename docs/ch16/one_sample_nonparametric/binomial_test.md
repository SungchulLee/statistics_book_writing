# Binomial Test

The **binomial test** is an exact non-parametric test for whether the probability of "success" in a sequence of independent Bernoulli trials equals a hypothesized value $p_0$. It is the most general member of the sign-test family: the [sign test](sign_test.md) is the special case with $p_0 = 0.5$, while the binomial test allows testing any proportion.

Because the test is based directly on the binomial distribution, it produces exact $p$-values without relying on normal approximations, making it especially appropriate for small samples.

## Hypotheses

Let $X_1, X_2, \ldots, X_n$ be independent Bernoulli trials with success probability $p$. The hypothesized value is $p_0$.

| Test type | $H_0$ | $H_a$ |
|:----------|:------|:------|
| Two-sided | $p = p_0$ | $p \ne p_0$ |
| Left-tailed | $p = p_0$ | $p < p_0$ |
| Right-tailed | $p = p_0$ | $p > p_0$ |

## Test Statistic

The test statistic is the observed number of successes:

$$
S = \sum_{i=1}^{n} X_i
$$

Under $H_0$,

$$
S \sim \text{Binomial}(n, p_0)
$$

No further transformation or standardization is needed because the exact distribution is fully known.

## Exact p-Values

**Right-tailed test** ($H_a \colon p > p_0$):

$$
p\text{-value} = P(S \ge s_{\text{obs}}) = \sum_{k=s_{\text{obs}}}^{n} \binom{n}{k} p_0^k (1 - p_0)^{n-k}
$$

**Left-tailed test** ($H_a \colon p < p_0$):

$$
p\text{-value} = P(S \le s_{\text{obs}}) = \sum_{k=0}^{s_{\text{obs}}} \binom{n}{k} p_0^k (1 - p_0)^{n-k}
$$

**Two-sided test** ($H_a \colon p \ne p_0$):

$$
p\text{-value} = 2 \min\!\bigl(P(S \le s_{\text{obs}}), \; P(S \ge s_{\text{obs}})\bigr)
$$

capped at 1. An alternative definition sums the probabilities of all outcomes at least as unlikely as the observed one.

!!! tip "No need for a normal approximation"
    Unlike many non-parametric tests that rely on large-sample normal approximations, the binomial test computes $p$-values exactly for any sample size. This makes it the preferred choice for small samples when testing a single proportion.

## Worked Example

A pharmaceutical company claims that its drug is effective in 70% of patients ($p_0 = 0.70$). In a clinical trial with $n = 20$ patients, only $s_{\text{obs}} = 10$ respond. Test whether the true response rate is less than 70%.

**Hypotheses:** $H_0 \colon p = 0.70$ vs $H_a \colon p < 0.70$.

**$p$-value (left-tailed):**

$$
p = P(S \le 10) = \sum_{k=0}^{10} \binom{20}{k} (0.70)^k (0.30)^{20-k}
$$

Computing this sum (or using software):

$$
p \approx 0.0480
$$

At $\alpha = 0.05$, we reject $H_0$. There is evidence that the true response rate is below 70%.

??? example "Exact computation detail"
    The individual terms are $P(S = k) = \binom{20}{k}(0.7)^k(0.3)^{20-k}$ for $k = 0, 1, \ldots, 10$. The cumulative sum can be obtained in Python via `scipy.stats.binom.cdf(10, 20, 0.70)`, which returns approximately 0.048.

## Relationship to the Sign Test

The sign test for the median is a binomial test with $p_0 = 0.5$. When testing $H_0 \colon \text{median} = m_0$, each observation is coded as a success ($X_i > m_0$) or failure ($X_i < m_0$), with ties excluded. Under $H_0$, the number of successes follows $\text{Binomial}(n, 0.5)$.

The binomial test generalizes this by allowing any $p_0$, which is useful when:

- Testing whether a treatment has a specific success rate (not necessarily 50%).
- Testing whether the proportion of values exceeding a threshold matches a known baseline.

## Normal Approximation for Large Samples

For large $n$, the exact binomial computation can be replaced by the normal approximation. The standardized statistic is

$$
Z = \frac{\hat{p} - p_0}{\sqrt{p_0(1 - p_0)/n}}
$$

where $\hat{p} = S/n$ is the sample proportion. This approximation is reliable when both $np_0 \ge 5$ and $n(1 - p_0) \ge 5$.

## Confidence Interval for p

The binomial test naturally yields a confidence interval for the true proportion $p$. The **Clopper-Pearson** exact confidence interval inverts the two one-sided binomial tests:

$$
\text{CI}_{1-\alpha} = \bigl(p_L, \; p_U\bigr)
$$

where $p_L$ and $p_U$ are the values of $p$ for which the observed result lies at the boundary of significance. This interval has guaranteed coverage of at least $1 - \alpha$, though it can be conservative.

## Summary

The binomial test provides exact inference for a population proportion without distributional assumptions beyond independence. It generalizes the sign test from $p_0 = 0.5$ to arbitrary hypothesized proportions and is particularly valuable for small samples where normal approximations are unreliable. For large samples, the standard normal approximation to the binomial provides a computationally simpler alternative with equivalent conclusions.


## Exercises

**Exercise 1.**
Describe the main concept of Binomial Test and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Binomial Test is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
