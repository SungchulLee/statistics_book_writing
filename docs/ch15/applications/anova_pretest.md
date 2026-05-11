# Pre-Test for Analysis of Variance Homoscedasticity

One-way ANOVA assumes that all $k$ groups share a common variance: $\sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2$. When this assumption is violated, the standard ANOVA F-test can produce misleading $p$-values. A natural strategy is to test the equal-variance assumption before running the ANOVA, using one of the variance tests from this chapter. This section discusses the pre-testing workflow, the choice of pre-test, and the controversy surrounding the two-stage approach.

## The Two-Stage Workflow

The traditional approach proceeds in two stages:

**Stage 1 (Pre-test).** Apply a variance homogeneity test at some significance level $\alpha_{\text{pre}}$:

- If the pre-test fails to reject $H_0\colon \sigma_1^2 = \cdots = \sigma_k^2$, proceed with the standard ANOVA F-test.
- If the pre-test rejects $H_0$, use an alternative procedure that does not require equal variances, such as Welch's ANOVA.

**Stage 2 (Main test).** Run the ANOVA or its alternative:

- **Standard ANOVA** uses the pooled variance $\text{MSE} = S_p^2$ in the denominator of the F-statistic and follows an $F_{k-1, N-k}$ distribution.
- **Welch's ANOVA** does not pool the variances. Instead, it uses a weighted average of the group means and an adjusted degrees-of-freedom approximation (Welch-Satterthwaite).

## Choosing the Pre-Test

The pre-test should be robust enough to produce reliable results on the same data that will enter the ANOVA. Since ANOVA data are often not perfectly normal, a normality-dependent pre-test (like Bartlett's) can reject the equal-variance hypothesis due to non-normality rather than genuine variance differences.

**Recommended pre-tests:**

| Pre-test | When to use |
|---|---|
| Brown-Forsythe | Default choice for most situations |
| Levene (mean) | When data are approximately symmetric |
| Fligner-Killeen | When data are heavily non-normal |

!!! warning "Do Not Use Bartlett's Test as a Pre-Test"
    Bartlett's test is too sensitive to non-normality to serve as a reliable ANOVA pre-test. If the data are non-normal, Bartlett's test may reject $H_0$ even when the variances are equal, leading the analyst to use Welch's ANOVA unnecessarily. The Brown-Forsythe test is the standard recommendation.

## The Pre-Testing Controversy

The two-stage approach has been criticized on several grounds:

**1. Inflated overall Type I error.** The combined procedure (pre-test followed by conditional ANOVA) does not maintain the nominal significance level $\alpha$ for the main test. The overall Type I error rate depends on the pre-test's power, the pre-test's significance level, and the degree of variance heterogeneity. Simulation studies show that the two-stage procedure can have an actual Type I error rate that differs from $\alpha$ by several percentage points.

**2. Low power of the pre-test.** Variance tests have limited power to detect moderate variance differences, especially with small samples. A non-significant pre-test does not mean the variances are equal; it may simply mean the sample sizes are too small to detect the difference.

**3. Conditional bias.** The decision to use standard ANOVA or Welch's ANOVA is data-dependent. This conditioning introduces a subtle bias into the main test's operating characteristics.

## The Modern Recommendation

Many statisticians now recommend bypassing the pre-test entirely and using Welch's ANOVA by default:

- **Welch's ANOVA with equal variances.** When the variances are in fact equal, Welch's ANOVA has only slightly lower power than standard ANOVA (typically 1--2 percentage points).
- **Welch's ANOVA with unequal variances.** When the variances are unequal, Welch's ANOVA maintains the correct Type I error rate while standard ANOVA does not.

The small power loss under equal variances is a modest price for the protection against variance heterogeneity.

!!! tip "When to Pre-Test vs. When to Default to Welch"

    - **Default to Welch's ANOVA** in routine analyses where robustness is more important than extracting every last bit of power.
    - **Use the pre-test** when the sample sizes are large (so the pre-test has good power), normality is confirmed, and the analyst wants to maximize power by pooling variances when justified.

## Example Workflow

A researcher compares the mean scores of $k = 4$ treatment groups, each with $n_i = 15$ observations.

**Step 1.** Check normality within each group (Shapiro-Wilk test or Q-Q plots). Suppose all four groups pass the normality check.

**Step 2.** Run the Brown-Forsythe test for equal variances at $\alpha_{\text{pre}} = 0.05$.

**Step 3.** If the Brown-Forsythe test fails to reject ($p > 0.05$): run standard one-way ANOVA. If it rejects ($p \le 0.05$): run Welch's ANOVA.

## Python Implementation

```python
import numpy as np
from scipy import stats

# Simulated group data
rng = np.random.default_rng(42)
g1 = rng.normal(50, 5, size=15)
g2 = rng.normal(55, 5, size=15)
g3 = rng.normal(52, 5, size=15)
g4 = rng.normal(48, 5, size=15)

# Step 1: Brown-Forsythe pre-test
bf_stat, bf_p = stats.levene(g1, g2, g3, g4, center='median')
print(f"Brown-Forsythe statistic: {bf_stat:.4f}")
print(f"Brown-Forsythe p-value:   {bf_p:.4f}")

# Step 2: Choose ANOVA type based on pre-test
alpha_pre = 0.05
if bf_p > alpha_pre:
    # Equal variances assumed
    f_stat, p_val = stats.f_oneway(g1, g2, g3, g4)
    print(f"\nStandard ANOVA F = {f_stat:.4f}, p = {p_val:.4f}")
else:
    # Unequal variances
    # Welch's ANOVA (available in scipy as Alexander-Govern or via pingouin)
    print("\nVariances are unequal. Use Welch's ANOVA.")

# Alternative: always use Welch's ANOVA (no pre-test needed)
# from pingouin import welch_anova
# welch_anova(data=df, dv='score', between='group')
```

## Summary

The pre-test for ANOVA homoscedasticity remains a common practice, but its limitations should be understood. The Brown-Forsythe test is the appropriate pre-test when one is used. However, the modern consensus favors defaulting to Welch's ANOVA, which performs well regardless of whether the variances are equal. The pre-test is most valuable when sample sizes are large enough to detect meaningful variance differences and when the analyst has confirmed normality.


## Exercises

**Exercise 1.**
Describe the main concept of Pre-Test for Analysis of Variance Homoscedasticity and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Pre-Test for Analysis of Variance Homoscedasticity is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
