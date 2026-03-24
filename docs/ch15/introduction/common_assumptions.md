# Assumptions Common to Variance Tests

Every variance test discussed in this chapter rests on a small set of shared assumptions. Before examining each test individually, it is worth understanding these assumptions together, because violating any one of them affects the validity of the resulting $p$-values and confidence intervals. The severity of the consequences depends on which assumption is broken and which test is used.

## Independence

The most fundamental requirement is that the observations within each sample are **independent**. Formally, for a sample $X_1, X_2, \ldots, X_n$, we require

$$
\operatorname{Cov}(X_i, X_j) = 0 \quad \text{for all } i \neq j
$$

Independence is necessary because the sampling distribution of the sample variance $S^2$ is derived under the assumption that the squared deviations $(X_i - \bar{X})^2$ behave like independent (or nearly independent) random variables. When observations are correlated, the effective sample size is smaller than $n$, and the distribution of $(n-1)S^2/\sigma^2$ no longer follows a chi-square distribution with $n - 1$ degrees of freedom.

**Common violations.** Time series data, clustered data (students within classrooms), and repeated measurements on the same subject all introduce dependence. When dependence is present, standard variance tests produce inflated Type I error rates because they underestimate the true uncertainty in $S^2$.

## Normality

The chi-square test, F-test, and Bartlett's test all assume that the underlying population follows a normal distribution:

$$
X_i \sim N(\mu, \sigma^2)
$$

This assumption is needed because the exact result

$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}
$$

holds only when $X_1, \ldots, X_n$ are i.i.d. normal. The chi-square distribution emerges from the fact that a sum of $n - 1$ independent squared standard normal variables has a $\chi^2_{n-1}$ distribution. If the population is not normal, the distribution of $(n-1)S^2/\sigma^2$ differs from $\chi^2_{n-1}$, and the critical values used in the test are no longer correct.

**Sensitivity varies by test.** The chi-square and F-tests are highly sensitive to non-normality, particularly to heavy tails and skewness. Bartlett's test is even more sensitive. The robust tests in Section 15.5 (Levene, Brown-Forsythe, Fligner-Killeen) are designed to work under much weaker distributional assumptions.

!!! warning "Normality Matters More for Variance Tests Than for Mean Tests"
    The central limit theorem ensures that $\bar{X}$ is approximately normal for moderate $n$, making $t$-tests reasonably robust to non-normality. No analogous result rescues variance tests. The distribution of $S^2$ converges to normality much more slowly, and its sensitivity to the fourth moment (kurtosis) of the population means that even moderate non-normality can distort the chi-square or F-test.

## Random Sampling

The observations must be drawn as a random sample from the population of interest. Each observation $X_i$ should be identically distributed with the same variance $\sigma^2$. If the sampling mechanism introduces systematic biases (e.g., convenience sampling, voluntary response), the sample variance $S^2$ may not be a meaningful estimator of the true population variance.

For multi-sample tests (the F-test, Bartlett's test, and the robust tests), an additional requirement is that the $k$ samples are drawn independently of one another:

$$
X_{ij} \sim F_j(\mu_j, \sigma_j^2), \quad i = 1, \ldots, n_j, \quad j = 1, \ldots, k
$$

where the samples from different groups are mutually independent.

## Consequences of Violations

The table below summarizes how each assumption violation affects the main variance tests:

| Violation | Chi-square / F-test | Bartlett's test | Levene / Brown-Forsythe |
|---|---|---|---|
| Dependence | Inflated Type I error | Inflated Type I error | Inflated Type I error |
| Non-normality (moderate) | Moderate size distortion | Severe size distortion | Mild or no distortion |
| Non-normality (heavy tails) | Severe size distortion | Very severe distortion | Mild distortion |
| Non-random sampling | Biased inference | Biased inference | Biased inference |

"Size distortion" means the actual Type I error rate differs from the nominal significance level $\alpha$. A test with nominal $\alpha = 0.05$ that rejects 15% of the time under $H_0$ has severe size distortion.

## Checking the Assumptions

Before running a variance test, the analyst should verify the assumptions using the tools from earlier chapters:

1. **Independence.** Review the study design. If data are collected over time or within clusters, consider using a test designed for dependent data or adjusting the degrees of freedom.
2. **Normality.** Use the graphical and formal methods from Chapter 14: Q-Q plots, the Shapiro-Wilk test, and the Anderson-Darling test. If the data are clearly non-normal, choose a robust test from Section 15.5 or a bootstrap approach from Section 15.6.
3. **Random sampling.** Evaluate the data collection process. Non-random sampling cannot be corrected by a statistical test; it requires careful judgment about the population to which conclusions apply.

??? example "Quick Diagnostic Checklist"
    Before applying any variance test in this chapter, verify the following:

    - [ ] Observations within each group are independent
    - [ ] Data collection used random sampling or a randomized experiment
    - [ ] If using the chi-square test, F-test, or Bartlett's test: normality has been checked
    - [ ] If normality is questionable: use Levene's, Brown-Forsythe, or Fligner-Killeen instead
