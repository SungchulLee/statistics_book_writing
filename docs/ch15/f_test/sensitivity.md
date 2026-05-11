# Sensitivity to Non-Normality

The F-test for comparing two variances assumes that both populations are normally distributed. Among all common statistical tests, the F-test is one of the most sensitive to violations of its distributional assumption. While $t$-tests for means are reasonably robust to moderate non-normality (thanks to the central limit theorem), no analogous protection exists for variance ratio tests. This section explains why the F-test breaks down under non-normality and summarizes the simulation evidence.

## Why the F-Test Is Sensitive

The F-test statistic $F = S_1^2 / S_2^2$ follows an $F_{n_1-1, n_2-1}$ distribution only when both samples come from normal populations. The exact distribution of $S^2$ depends on all moments of the underlying distribution, not just the first two. In particular, the fourth central moment (related to kurtosis) plays a critical role.

For a population with kurtosis $\kappa$, the variance of the sample variance satisfies

$$
\operatorname{Var}(S^2) = \frac{1}{n}\left(\mu_4 - \frac{n-3}{n-1}\sigma^4\right)
$$

where $\mu_4 = E[(X - \mu)^4]$ is the fourth central moment. For a normal distribution, $\mu_4 = 3\sigma^4$, which simplifies this expression. For a heavy-tailed distribution with excess kurtosis $\gamma_2 = \mu_4/\sigma^4 - 3 > 0$, the variance of $S^2$ is inflated, causing the actual distribution of $F$ to have heavier tails than the nominal $F_{n_1-1, n_2-1}$ distribution.

!!! warning "The Kurtosis Effect"
    The F-test is primarily sensitive to **kurtosis** (heavy or light tails) rather than skewness. A symmetric distribution with heavy tails (such as the $t$-distribution with small degrees of freedom) can distort the F-test more severely than a moderately skewed distribution with normal-like tails.

## Type I Error Inflation

When the populations are non-normal, the actual Type I error rate of the F-test can differ substantially from the nominal significance level $\alpha$. The direction and magnitude of the distortion depend on the kurtosis:

- **Heavy-tailed distributions** ($\gamma_2 > 0$, leptokurtic): The actual rejection rate exceeds $\alpha$. The F-test is liberal, rejecting too often.
- **Light-tailed distributions** ($\gamma_2 < 0$, platykurtic): The actual rejection rate falls below $\alpha$. The F-test is conservative, rejecting too rarely.

The following table summarizes simulation results for the F-test at nominal $\alpha = 0.05$ with $n_1 = n_2 = 20$, testing $H_0\colon \sigma_1^2 = \sigma_2^2$ when the null is true:

| Distribution | Excess kurtosis | Actual Type I error |
|---|---|---|
| Normal | 0 | 0.050 |
| $t_5$ | 6 | 0.140 |
| $t_{10}$ | 1 | 0.075 |
| Exponential | 6 | 0.130 |
| Uniform | $-1.2$ | 0.028 |
| Laplace | 3 | 0.095 |

With $t_5$ data, the nominal 5% test rejects nearly 14% of the time under $H_0$ -- almost three times the intended rate.

## Simulation Evidence

Extensive simulation studies (Box, 1953; Markowski and Markowski, 1990) confirm the following patterns:

1. **Moderate non-normality.** Even mild departures from normality (excess kurtosis of 1--2) inflate the Type I error rate noticeably.
2. **Heavy tails dominate.** The distortion grows rapidly with kurtosis. For distributions like the $t_3$ or contaminated normal, the actual error rate can exceed 20% at a nominal 5% level.
3. **Skewness alone is less damaging.** Skewed distributions with moderate kurtosis (e.g., a mildly skewed population with $\gamma_2 \approx 0$) cause smaller distortions than symmetric heavy-tailed distributions.
4. **Sample size does not help much.** Unlike the $t$-test, increasing the sample size does not substantially reduce the sensitivity of the F-test to non-normality. The fundamental problem is that the distribution of $S^2$ depends on the fourth moment regardless of $n$.

## Comparison with the t-Test

The contrast with the $t$-test is instructive. For the one-sample $t$-test:

$$
T = \frac{\bar{X} - \mu}{S / \sqrt{n}} \sim t_{n-1}
$$

The central limit theorem ensures $\bar{X}$ is approximately normal for moderate $n$, making the numerator well-behaved. The denominator $S$ converges to $\sigma$ by the law of large numbers, so the overall statistic remains approximately $t$-distributed.

For the F-test, both the numerator and denominator are sample variances. Neither benefits from a CLT-like result for moderate sample sizes. The distribution of $S^2$ is determined by the population's fourth moment, and the ratio $S_1^2/S_2^2$ inherits distortions from both.

## When to Avoid the F-Test

The F-test should be avoided when:

- A normality test (Shapiro-Wilk, Anderson-Darling, or Q-Q plot inspection) rejects normality for either sample
- The data are known to come from a heavy-tailed distribution (financial returns, income data, survival times)
- The data contain outliers, which inflate both the sample variance and the kurtosis
- The sample sizes are small and normality cannot be reliably assessed

## Recommended Alternatives

When normality is in doubt, the following tests provide better control of the Type I error rate:

| Test | Robustness | Power under normality | Section |
|---|---|---|---|
| Levene's test | Good | Slightly lower than F-test | 15.5 |
| Brown-Forsythe test | Very good | Slightly lower than Levene's | 15.5 |
| Fligner-Killeen test | Excellent | Lower than Levene's | 15.5 |
| Bootstrap test | Good to excellent | Depends on implementation | 15.6 |

!!! tip "Practical Recommendation"
    Unless normality has been confirmed through formal testing and graphical inspection, prefer the Brown-Forsythe test over the F-test. The Brown-Forsythe test maintains its nominal Type I error rate across a wide range of distributions while retaining reasonable power when the data happen to be normal.


## Exercises

**Exercise 1.**
Describe the main concept of Sensitivity to Non-Normality and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Sensitivity to Non-Normality is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
