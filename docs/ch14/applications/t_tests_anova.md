# Normality in t-Tests and Analysis of Variance

## Where the Assumption Enters

The $t$-test and ANOVA both require normality, but the precise form of the assumption differs between the two. In the $t$-test, we assume that the underlying population (or populations) are normally distributed. In ANOVA, the assumption is that the observations within each group are drawn from normal populations. In both cases, the normality assumption ensures that the test statistic follows its reference distribution exactly, so that $p$-values and critical values are accurate.

Before applying these procedures, it is good practice to check normality. This section explains where the assumption enters, how to diagnose violations, and how robust each procedure is to departures.

## The One-Sample t-Test

For a sample $X_1, X_2, \ldots, X_n$ from a population with mean $\mu$ and variance $\sigma^2$, the one-sample $t$-test assumes

$$
X_i \overset{\text{iid}}{\sim} N(\mu, \sigma^2)
$$

Under this assumption, the test statistic

$$
t = \frac{\bar{X} - \mu_0}{S / \sqrt{n}}
$$

follows a $t$-distribution with $n - 1$ degrees of freedom exactly. The normality of $X_i$ guarantees two properties simultaneously: $\bar{X}$ is normally distributed, and $\bar{X}$ and $S^2$ are independent. Both are needed for the $t$-distribution to hold.

## The Two-Sample t-Test

The independent two-sample $t$-test assumes normality in both populations:

$$
X_{1i} \overset{\text{iid}}{\sim} N(\mu_1, \sigma^2), \quad X_{2j} \overset{\text{iid}}{\sim} N(\mu_2, \sigma^2)
$$

Under equal variances, the pooled $t$-statistic follows a $t$-distribution with $n_1 + n_2 - 2$ degrees of freedom. Welch's $t$-test relaxes the equal-variance assumption but still requires normality in each group for the approximate degrees of freedom to be valid.

## One-Way ANOVA

In one-way ANOVA with $k$ groups, the model is

$$
X_{ij} = \mu_i + \varepsilon_{ij}, \quad \varepsilon_{ij} \overset{\text{iid}}{\sim} N(0, \sigma^2)
$$

for $i = 1, \ldots, k$ and $j = 1, \ldots, n_i$. The normality assumption applies to the error terms $\varepsilon_{ij}$, which is equivalent to assuming that the observations within each group are normally distributed around the group mean. The $F$-statistic

$$
F = \frac{\text{MSB}}{\text{MSW}}
$$

follows an $F$-distribution with $k - 1$ and $N - k$ degrees of freedom under $H_0$ and the normality assumption.

## Checking Normality in Practice

### For the t-Test

Since the $t$-test assumes normality of the raw data (or each group's data), the check should be applied to the sample values directly:

1. **Q-Q plot** of the sample values against normal quantiles.
2. **Shapiro-Wilk test** on the sample values.
3. **Histogram** to visually assess symmetry and tail behavior.

### For ANOVA

In ANOVA, normality is an assumption about the residuals $\hat{\varepsilon}_{ij} = X_{ij} - \bar{X}_{i\cdot}$, where $\bar{X}_{i\cdot}$ is the group mean. The diagnostic procedure is:

1. **Compute residuals** by subtracting group means from each observation.
2. **Q-Q plot** of the pooled residuals against normal quantiles.
3. **Shapiro-Wilk test** on the pooled residuals.

Checking residuals rather than raw data is important because the raw data are a mixture of $k$ potentially different distributions (different means), even if all groups are normal.

??? tip "Check Residuals, Not Raw Data, in ANOVA"
    If the group means differ substantially, the combined raw data may appear non-normal (e.g., multimodal) even when each group is perfectly normal. Always check the residuals, which remove the group-mean differences and isolate the distributional assumption.

## Robustness to Non-Normality

### The t-Test

The one-sample and two-sample $t$-tests are robust to moderate departures from normality, especially when:

- The distribution is symmetric (even with heavy tails).
- The sample size is at least $n \geq 20$ per group.
- The departure is in the tails rather than in the form of skewness.

For skewed populations, the $t$-test can have an inflated Type I error rate in small samples. The distortion is typically modest: an actual $\alpha$ of 0.06--0.08 when the nominal level is 0.05.

### ANOVA

The $F$-test in ANOVA is moderately robust to non-normality when:

- Group sizes are equal (balanced design).
- The distributions are symmetric.
- The number of observations per group is at least 15--20.

Unbalanced designs combined with non-normality are more problematic. In such cases, the Welch ANOVA (which does not assume equal variances) combined with larger sample sizes provides better protection.

## Python Example

```python
import numpy as np
from scipy import stats

# ===================================================================
# Check normality of ANOVA residuals
# ===================================================================

np.random.seed(42)

# Three groups with slightly different means
group1 = np.random.normal(loc=5.0, scale=1.5, size=30)
group2 = np.random.normal(loc=5.5, scale=1.5, size=30)
group3 = np.random.normal(loc=6.0, scale=1.5, size=30)

# Compute residuals (observations minus group means)
residuals = np.concatenate([
    group1 - np.mean(group1),
    group2 - np.mean(group2),
    group3 - np.mean(group3),
])

# Shapiro-Wilk test on residuals
sw_stat, sw_p = stats.shapiro(residuals)

# One-way ANOVA
f_stat, f_p = stats.f_oneway(group1, group2, group3)

if __name__ == "__main__":
    print("Normality check on ANOVA residuals:")
    print(f"  Shapiro-Wilk: W = {sw_stat:.4f}, p = {sw_p:.4f}")
    print(f"\nOne-way ANOVA:")
    print(f"  F = {f_stat:.4f}, p = {f_p:.4f}")

    if sw_p > 0.05:
        print("\n  Residuals are consistent with normality (p > 0.05).")
    else:
        print("\n  Evidence of non-normality in residuals (p <= 0.05).")
```

## When Normality Fails

If diagnostics reveal non-normality in the $t$-test or ANOVA setting, several alternatives are available:

1. **Increase sample size.** The CLT provides asymptotic protection for mean-based tests.
2. **Apply a transformation.** Log or square-root transformations can reduce skewness.
3. **Use a nonparametric test.** The Wilcoxon rank-sum test replaces the two-sample $t$-test; the Kruskal-Wallis test replaces one-way ANOVA.
4. **Use a permutation test.** Permutation-based $p$-values do not require distributional assumptions.

## Summary

The normality assumption in $t$-tests applies to the raw data, while in ANOVA it applies to the residuals. Both procedures are moderately robust to non-normality, with the $t$-test being more robust than the $F$-test. Checking normality should involve both graphical methods (Q-Q plots of residuals) and formal tests (Shapiro-Wilk). When violations are detected, the practitioner should assess whether the departure is practically significant before switching to alternative methods.

## Exercises

**Exercise 1.**
A researcher applies a t-test to a sample of $n = 12$ observations that are moderately right-skewed. Should they be concerned about the normality assumption?

??? success "Solution to Exercise 1"
    With $n = 12$, the CLT provides only limited protection, so moderate skewness is a concern. The t-test assumes normality for exact validity; with skewed data and small $n$, the actual Type I error rate can deviate from the nominal $\alpha$, and confidence intervals may have incorrect coverage.

    The researcher should: (1) assess the degree of skewness with a Q-Q plot, (2) consider a nonparametric alternative (Wilcoxon signed-rank test), (3) try a log transformation if the data are positive and right-skewed. If the skewness is mild, the t-test may still be approximately valid, but reporting the sensitivity analysis is good practice.

---

**Exercise 2.**
ANOVA is often described as "robust to non-normality." Qualify this statement: when is ANOVA robust, and when is it not?

??? success "Solution to Exercise 2"
    ANOVA is robust when: (1) sample sizes are large ($n_j \geq 20$-$30$ per group), because the CLT normalizes group means; (2) group sizes are equal (balanced design), which protects against variance heterogeneity; (3) departures from normality are moderate (mild skewness, no extreme outliers).

    ANOVA is NOT robust when: (1) sample sizes are small and unequal with heterogeneous variances; (2) data are heavily skewed or have outliers in small samples; (3) the data are severely non-normal (e.g., bimodal, heavy-tailed). In these cases, the F-test's Type I error rate can be substantially inflated or deflated.

---

**Exercise 3.**
Compare the Welch t-test and the Mann-Whitney U test as alternatives when normality is questionable. When is each preferred?

??? success "Solution to Exercise 3"
    **Welch t-test:** Does not assume equal variances but still assumes approximate normality (or large $n$). It tests whether the means differ. Preferred when: data are approximately normal or $n$ is large, and the research question is about means.

    **Mann-Whitney U test:** A nonparametric test that does not assume normality. Under the assumption of equal shape distributions, it tests whether one group tends to have larger values (stochastic dominance). Preferred when: data are ordinal, heavily skewed, contain outliers, or $n$ is small.

    Key difference: the Welch test is about means; the Mann-Whitney is about the entire distribution (or medians under symmetry). If the distributions have different shapes, the Mann-Whitney tests something other than a location shift.

---

**Exercise 4.**
A one-way ANOVA with $k = 4$ groups and $n_j = 8$ per group rejects $H_0$ ($p = 0.02$). A Shapiro-Wilk test on the residuals gives $p = 0.04$. Should the ANOVA result be trusted?

??? success "Solution to Exercise 4"
    The Shapiro-Wilk rejection ($p = 0.04$) raises a concern but does not invalidate the ANOVA automatically. With $n_j = 8$ (small groups), the t/F-distribution approximations are more sensitive to non-normality.

    Recommended actions:

    1. Examine Q-Q plots and histograms of residuals to assess the degree of non-normality.
    2. Run a nonparametric alternative (Kruskal-Wallis test) as a sensitivity check. If it also rejects, the ANOVA conclusion is supported.
    3. If the non-normality is due to outliers, consider robust ANOVA methods or trimmed means.
    4. Report both the ANOVA and Kruskal-Wallis results, noting the normality concern.

    If the Kruskal-Wallis also gives $p < 0.05$, confidence in the conclusion increases. If results diverge, the normality violation may be responsible.
