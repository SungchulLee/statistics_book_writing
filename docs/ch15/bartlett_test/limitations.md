# Limitations Under Non-Normality

Bartlett's test is the most powerful test for equal variances when the data are truly normal, but it is also the most fragile when they are not. Among the variance tests in this chapter, Bartlett's test has the highest sensitivity to non-normality, exceeding even the F-test. This makes it unreliable as a routine diagnostic, particularly when normality has not been verified.

## The Core Problem

Bartlett's test statistic is derived from the likelihood ratio under the assumption of normality. The chi-square approximation

$$
T \stackrel{\text{approx}}{\sim} \chi^2_{k-1}
$$

depends on the fact that log sample variances are approximately normal when the underlying data are normal. For non-normal populations, the distribution of $\ln S_i^2$ deviates from normality, and the chi-square reference distribution becomes inaccurate.

The key quantity is the population kurtosis. For a distribution with excess kurtosis $\gamma_2 = \mu_4 / \sigma^4 - 3$, the variance of $\ln S^2$ is inflated relative to the normal case:

$$
\operatorname{Var}(\ln S^2) \approx \frac{2}{\nu} + \frac{\gamma_2}{\nu} + O(\nu^{-2})
$$

where $\nu = n - 1$. Under normality, $\gamma_2 = 0$ and the leading term is $2/\nu$. For heavy-tailed distributions with $\gamma_2 > 0$, the additional term $\gamma_2/\nu$ increases the variability of the log-variances, causing $T$ to be stochastically larger than the $\chi^2_{k-1}$ reference.

## Type I Error Inflation

Simulation studies consistently show that Bartlett's test rejects $H_0$ far too often when the data are non-normal. The following table reports actual rejection rates at nominal $\alpha = 0.05$ for $k = 3$ groups with $n_i = 20$:

| Distribution | Excess kurtosis | Actual Type I error |
|---|---|---|
| Normal | 0 | 0.050 |
| $t_{10}$ | 1 | 0.090 |
| $t_5$ | 6 | 0.220 |
| Exponential | 6 | 0.200 |
| $\chi^2_4$ | 3 | 0.140 |
| Contaminated normal | 12 | 0.350 |
| Uniform | $-1.2$ | 0.025 |

With heavy-tailed distributions like the contaminated normal, Bartlett's test rejects 35% of the time when the true rejection rate should be 5%. This means that a significant Bartlett result may reflect non-normality rather than unequal variances.

!!! danger "Bartlett's Test Can Be a Normality Test in Disguise"
    When applied to non-normal data, Bartlett's test often rejects $H_0\colon \sigma_1^2 = \cdots = \sigma_k^2$ not because the variances are actually unequal, but because the distributional shape violates the normality assumption. A rejection may indicate nothing about variance heterogeneity.

## Comparison with the F-Test

Both the F-test and Bartlett's test assume normality, but Bartlett's test is more sensitive to violations. The reason is that Bartlett's test uses the logarithm of sample variances, and the distribution of $\ln S^2$ is more sensitive to kurtosis than the distribution of $S^2$ itself. In simulations, the Type I error inflation for Bartlett's test is typically 1.5 to 2 times worse than for the F-test under the same non-normal distribution.

| Scenario ($n_i = 20$, $k = 3$) | F-test Type I error | Bartlett's Type I error |
|---|---|---|
| Normal | 0.050 | 0.050 |
| $t_5$ | 0.140 | 0.220 |
| Exponential | 0.130 | 0.200 |

## When NOT to Use Bartlett's Test

Bartlett's test should be avoided in the following situations:

1. **Non-normal data.** If the Q-Q plot shows heavy tails, skewness, or outliers, Bartlett's test is unreliable. Use Levene's test or the Brown-Forsythe test instead.
2. **Small samples.** With small $n_i$, normality cannot be reliably assessed, and the chi-square approximation is less accurate. The correction factor $C$ mitigates this but does not eliminate the problem.
3. **Unknown distributional shape.** When the analyst has no prior knowledge about the population distribution, a robust test is the safer choice.
4. **Preliminary test before ANOVA.** Bartlett's test is sometimes recommended as a pre-test for ANOVA homoscedasticity. However, because ANOVA data are often not perfectly normal, Levene's test is preferred as the pre-test.

## When Bartlett's Test Is Appropriate

Despite its limitations, Bartlett's test remains useful in specific settings:

- **Data known to be normal.** When the population is known to be normal (e.g., measurement errors from calibrated instruments), Bartlett's test is the most powerful choice.
- **Normality confirmed by formal tests.** If the Shapiro-Wilk or Anderson-Darling test fails to reject normality and the Q-Q plot looks linear, Bartlett's test is appropriate.
- **Large, well-behaved samples.** With large sample sizes from nearly normal populations, the chi-square approximation is accurate.

!!! tip "Decision Rule for Practitioners"

    1. Test normality in each group (Shapiro-Wilk or Q-Q plots).
    2. If normality holds: use Bartlett's test for maximum power.
    3. If normality is doubtful: use the Brown-Forsythe test (Section 15.5).
    4. If normality is clearly violated: use the Fligner-Killeen test (Section 15.5).

## Historical Context

Bartlett published the test in 1937 as an extension of the F-test to $k > 2$ groups. For decades it was the standard test for homogeneity of variances. The recognition of its extreme sensitivity to non-normality led to the development of robust alternatives by Levene (1960), Brown and Forsythe (1974), and Fligner and Killeen (1976). Today, most statistical software defaults to Levene's test or Brown-Forsythe rather than Bartlett's test for routine use.

## Exercises

**Exercise 1.**
For two datasets that are not normally distributed, a Bartlett's test was performed, resulting in a low p-value (rejection of $H_0$: equal variances). However, a Levene's test yielded a high p-value (failure to reject $H_0$). How should these two test results be interpreted?

??? success "Solution to Exercise 1"

    - The results of Bartlett's test are **not reliable** when the assumption of normality is violated. Bartlett's test is highly sensitive to non-normality, and its rejection of the null hypothesis may be driven by the distributional shape rather than actual differences in variance.
    - Levene's test is **less sensitive** to violations of normality, so it provides a more trustworthy result in this scenario. It is reasonable to conclude that the variances of the two datasets are equal.
