# Fligner-Killeen Test

The Fligner-Killeen test (1976) is the most robust of the variance homogeneity tests discussed in this chapter. It replaces the raw absolute deviations used in Levene's test with **ranks** of those deviations, and then converts the ranks into **normal scores**. By working entirely with ranks and normal scores, the test becomes nearly distribution-free, maintaining its nominal Type I error rate even under heavy-tailed or strongly skewed distributions.

## Procedure

The test proceeds in five steps:

**Step 1.** Compute the absolute deviations from the group medians:

$$
Z_{ij} = |X_{ij} - \tilde{X}_i|
$$

where $\tilde{X}_i$ is the median of group $i$. This is the same transformation used in the Brown-Forsythe test.

**Step 2.** Rank all $N$ absolute deviations $Z_{ij}$ from smallest to largest. Let $R_{ij}$ denote the rank of $Z_{ij}$ among all $N$ values. Ties are resolved using the average rank.

**Step 3.** Convert the ranks into normal scores using the inverse normal (quantile) transformation:

$$
a_{ij} = \mathcal{N}^{-1}\!\left(\frac{1 + R_{ij}/(N+1)}{2}\right)
$$

where $\mathcal{N}^{-1}$ is the inverse of the standard normal CDF. The argument $(1 + R_{ij}/(N+1))/2$ maps the rank to a value in $(0.5, 1)$, and the inverse normal maps this to a positive score. Larger deviations receive larger normal scores.

**Step 4.** Compute the group means of the normal scores:

$$
\bar{a}_i = \frac{1}{n_i}\sum_{j=1}^{n_i} a_{ij}, \qquad \bar{a} = \frac{1}{N}\sum_{i=1}^{k}\sum_{j=1}^{n_i} a_{ij}
$$

**Step 5.** Compute the test statistic:

$$
\chi^2_{\text{FK}} = \frac{\sum_{i=1}^{k} n_i (\bar{a}_i - \bar{a})^2}{\frac{1}{N-1}\sum_{i=1}^{k}\sum_{j=1}^{n_i}(a_{ij} - \bar{a})^2}
$$

Under $H_0\colon \sigma_1^2 = \cdots = \sigma_k^2$, the statistic $\chi^2_{\text{FK}}$ follows approximately a chi-square distribution with $k - 1$ degrees of freedom.

## Hypotheses and Decision Rule

$$
H_0\colon \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2
$$

$$
H_1\colon \sigma_i^2 \neq \sigma_j^2 \text{ for at least one pair } i \neq j
$$

Reject $H_0$ at significance level $\alpha$ if $\chi^2_{\text{FK}} > \chi^2_{1-\alpha,\, k-1}$.

## Why Normal Scores Provide Robustness

The use of ranks eliminates the influence of the actual magnitude of the deviations. An extremely large deviation receives a high rank but not an extremely large score, because the normal score function compresses the upper tail. This double layer of protection (median-based deviations followed by rank-based scores) makes the test robust to:

- **Outliers:** A single extreme observation changes only one rank, with minimal effect on the overall test.
- **Heavy tails:** Distributions like the $t_3$ or Cauchy produce occasional large deviations, but the normal score transformation bounds their influence.
- **Skewness:** The median center and the rank transformation together neutralize the asymmetry.

!!! note "Comparison with Levene and Brown-Forsythe"
    Levene's test works directly with absolute deviations from the mean. Brown-Forsythe improves robustness by using the median. Fligner-Killeen adds a third layer by converting deviations to ranks and then to normal scores. Each step trades a small amount of power under normality for greater robustness under non-normality.

## Example

Consider two groups:

| Group 1 | Group 2 |
|---|---|
| 10, 12, 11, 13, 10 | 8, 25, 15, 30, 12 |

**Step 1.** Medians: $\tilde{X}_1 = 11$, $\tilde{X}_2 = 15$.

**Step 2.** Absolute deviations from medians:

| Group 1 | Group 2 |
|---|---|
| 1, 1, 0, 2, 1 | 7, 10, 0, 15, 3 |

**Step 3.** Rank the 10 deviations: $\{0, 0, 1, 1, 1, 2, 3, 7, 10, 15\}$ with ranks $\{1.5, 1.5, 4, 4, 4, 6, 7, 8, 9, 10\}$ (average ranks for ties).

**Step 4.** Convert ranks to normal scores using $a = \mathcal{N}^{-1}((1 + R/11)/2)$.

**Step 5.** Compute the Fligner-Killeen statistic from the group means of the normal scores. Group 2 has larger deviations and therefore larger normal scores on average, producing a large $\chi^2_{\text{FK}}$ value.

## Python Implementation

```python
import numpy as np
from scipy import stats

# Group data
group1 = [10, 12, 11, 13, 10]
group2 = [8, 25, 15, 30, 12]

# Fligner-Killeen test
stat, p_value = stats.fligner(group1, group2)
print(f"Fligner-Killeen statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Reject H0: variances are significantly different.")
else:
    print("Fail to reject H0: no significant difference in variances.")
```

## Strengths and Limitations

**Strengths:**

- Most robust test for variance homogeneity among the classical methods
- Controls Type I error rate well across a wide range of distributions
- Particularly effective for heavy-tailed and contaminated data
- Uses a chi-square reference distribution, which is simple and well-tabulated

**Limitations:**

- Lower power than Levene's or Brown-Forsythe when data are normal or nearly normal
- More computationally involved than other tests (though negligible with modern software)
- Less commonly implemented in basic statistical software compared to Levene's test

The Fligner-Killeen test is the preferred choice when the analyst suspects heavy contamination or has no confidence in the normality of the data. For routine analyses where the distribution is mildly non-normal, the Brown-Forsythe test offers a better power-robustness tradeoff.


## Exercises

**Exercise 1.**
Describe the main concept of Fligner-Killeen Test and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Fligner-Killeen Test is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
