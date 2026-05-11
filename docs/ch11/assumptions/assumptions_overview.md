# Assumptions in ANOVA

The one-way ANOVA model decomposes each observation as $Y_{ij} = \mu_i + \epsilon_{ij}$, where $\mu_i$ is the population mean of group $i$ and $\epsilon_{ij}$ is a random error term. The validity of the F-test depends on specific conditions imposed on these error terms. Understanding why each assumption is needed -- and what goes wrong when it fails -- is essential before applying ANOVA to real data. This page summarizes the four key assumptions and their consequences; the remaining pages in this section provide detailed diagnostic procedures for each one.

## Summary of Assumptions

The four assumptions underlying the ANOVA F-test are:

1. **Normality:** The error terms $\epsilon_{ij}$ are normally distributed within each group, so that $Y_{ij} \mid \text{group } i \sim N(\mu_i, \sigma^2)$. In practice, this is assessed by examining the residuals $e_{ij} = Y_{ij} - \bar{Y}_{i\cdot}$, which estimate the unobservable errors. See [Checking Normality](normality.md) for diagnostic methods.

2. **Independence:** Observations are statistically independent of one another. This assumption is primarily ensured through proper experimental design -- random sampling and random assignment -- rather than tested after the fact. See [Checking Independence](independence.md) for details.

3. **Homoscedasticity:** The population variance $\sigma^2$ is the same across all groups. When group variances differ, the pooled mean-square-error no longer estimates a single common variance, and the F-ratio becomes unreliable. See [Checking Homoscedasticity](homoscedasticity.md) for Levene's test and related diagnostics.

4. **Linearity:** The relationship between predictors and the response is linear. This assumption is relevant when the ANOVA model includes continuous covariates, as in ANCOVA; for purely categorical factors, linearity is automatically satisfied. See [Checking Linearity](linearity.md) for guidance.

## Why Assumptions Matter

The F-test statistic follows an exact $F$-distribution under the null hypothesis only when normality, independence, and homoscedasticity all hold. Violations distort this reference distribution, leading to incorrect p-values and unreliable decisions.

- **Normality.** When residuals are non-normal, the sampling distribution of the F-statistic deviates from the theoretical $F$-distribution, particularly in small samples. The Central Limit Theorem provides some robustness as sample sizes grow, but skewed or heavy-tailed distributions can still inflate or deflate the Type I error rate with fewer than 20--30 observations per group.

- **Independence.** Dependence among observations is the most consequential violation. Positive correlation reduces the effective degrees of freedom below the nominal count, causing the standard error estimates to be too small. The result is an inflated Type I error rate -- the test rejects far more often than the nominal $\alpha$ level suggests.

- **Homoscedasticity.** Unequal group variances cause the pooled variance estimate to over-represent groups with larger variances. The F-test then becomes liberal (rejecting too often) when smaller groups have larger variances, and conservative (rejecting too rarely) in the reverse scenario. Welch's ANOVA, which does not pool variances, provides a robust alternative.

- **Linearity.** When continuous covariates are present and the true relationship is nonlinear, the model systematically misestimates group means. Residual plots reveal characteristic curved patterns that signal this type of misspecification.

??? example "Illustration: Effect of Unequal Variances on the F-Test"

    Consider three groups with $n_1 = n_2 = n_3 = 10$ and identical population means $\mu_1 = \mu_2 = \mu_3 = 0$. Under homoscedasticity ($\sigma_1^2 = \sigma_2^2 = \sigma_3^2 = 1$), the F-test rejects at the $\alpha = 0.05$ level approximately 5% of the time, as expected.

    Now suppose $\sigma_1^2 = 1$, $\sigma_2^2 = 1$, and $\sigma_3^2 = 9$ (group 3 has much higher variance). Even though all population means are still equal, the standard F-test rejects at roughly 8--10% instead of 5%, because the pooled variance estimate is distorted. Welch's ANOVA, which estimates each group's variance separately, maintains the correct 5% rejection rate in this scenario.

    This example shows why checking homoscedasticity before interpreting the standard ANOVA F-test is important: the test can produce spurious "significant" results when variances differ substantially.

## Diagnostic Workflow

A systematic approach to assumption checking proceeds as follows:

1. **Fit the ANOVA model** and compute the residuals $e_{ij} = Y_{ij} - \bar{Y}_{i\cdot}$.
2. **Check independence** by reviewing the study design. If observations might be correlated (e.g., repeated measures, time-series data), consider mixed-effects models or repeated-measures ANOVA instead.
3. **Check homoscedasticity** using Levene's test or a residual-vs-fitted-values plot. If variances are unequal, switch to Welch's ANOVA or apply a variance-stabilizing transformation.
4. **Check normality** using a Q-Q plot of the residuals and the Shapiro-Wilk test. For large samples ($n > 30$ per group), moderate non-normality is generally tolerable.
5. **Check linearity** if the model includes continuous covariates. Use scatter plots and partial-residual plots to detect curvature.

When one or more assumptions are violated, the appropriate remedy depends on which assumption fails. The [Handling Assumption Violations](../diagnostics/handling_violations.md) page provides a decision framework covering transformations, non-parametric alternatives (Kruskal-Wallis), and robust methods (Welch's ANOVA).

## Exercises

**Exercise 1.**
List the three standard assumptions for one-way ANOVA and explain what happens to the F-test if each is violated.

??? success "Solution to Exercise 1"
    The three assumptions are:

    1. **Independence:** Observations within and between groups are independent. Violation (e.g., repeated measures, clustered data) inflates the Type I error rate because the effective sample size is smaller than the nominal $n$.

    2. **Normality:** The residuals (or equivalently, the observations within each group) are normally distributed. Moderate violations are tolerable with large samples (by the CLT), but severe skewness or heavy tails can distort the F-distribution, especially with small or unequal group sizes.

    3. **Homoscedasticity (equal variances):** All groups have the same population variance $\sigma^2$. Unequal variances inflate or deflate the F-statistic depending on whether larger variances are associated with smaller or larger groups. The Welch ANOVA is a robust alternative when this assumption fails.

---

**Exercise 2.**
A Levene's test for homogeneity of variances yields a p-value of 0.02. Should you proceed with standard ANOVA? Suggest an alternative approach.

??? success "Solution to Exercise 2"
    A Levene's test p-value of 0.02 provides evidence against the equal-variance assumption at the 5% level. Proceeding with standard ANOVA risks an incorrect F-test: if the larger variance is in the smaller group, the F-test is liberal (too many false positives); if in the larger group, it is conservative (loss of power).

    Alternatives:

    1. **Welch's ANOVA:** Does not assume equal variances and adjusts the degrees of freedom accordingly. This is the recommended default.
    2. **Transformation:** Apply a variance-stabilizing transformation (e.g., log or square root) to equalize variances, then run standard ANOVA.
    3. **Nonparametric test:** Use the Kruskal-Wallis test, which does not assume normality or equal variances.

---

**Exercise 3.**
Explain why ANOVA is considered robust to moderate violations of the normality assumption when sample sizes are large and balanced.

??? success "Solution to Exercise 3"
    ANOVA's robustness comes from two factors:

    1. **Central Limit Theorem:** The group means $\bar{X}_j$ are approximately normal even when individual observations are not, as long as $n_j$ is reasonably large (often $n_j \geq 20-30$ suffices). Since the F-statistic is based on comparing group means, normality of the means is what matters.

    2. **Balanced designs:** When group sizes are equal ($n_1 = n_2 = \dots = n_k$), the F-test is robust to unequal variances because the pooled variance estimate weights each group equally. With unbalanced designs, the F-test becomes sensitive to heteroscedasticity.

    However, ANOVA is not robust to heavy tails in small samples, severe skewness, or outliers. In these cases, transformations or nonparametric alternatives should be considered.

---

**Exercise 4.**
Describe how to use residual plots to assess the assumptions of normality and homoscedasticity after fitting an ANOVA model.

??? success "Solution to Exercise 4"
    After fitting the ANOVA model, compute the residuals $e_{ij} = x_{ij} - \bar{x}_j$ (observation minus group mean). Then:

    1. **Normality check:** Create a Q-Q plot of the residuals against theoretical normal quantiles. If the points lie approximately along a straight line, the normality assumption is reasonable. Systematic curvature indicates skewness; heavy tails appear as S-shaped departures. Alternatively, apply a Shapiro-Wilk test to the residuals.

    2. **Homoscedasticity check:** Plot residuals against the fitted values (group means) or group labels. The spread of residuals should be approximately equal across groups. A funnel shape (spread increasing with the mean) suggests heteroscedasticity. Alternatively, compare group-wise residual variances numerically or use Levene's test.

    3. **Independence check:** If data have a natural ordering (time, space), plot residuals in that order to check for patterns. Autocorrelation in residuals indicates violation of independence.
