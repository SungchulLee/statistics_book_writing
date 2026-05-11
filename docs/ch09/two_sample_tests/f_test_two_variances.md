# F-Test for Two Variances

## Overview

Many statistical procedures — including the pooled two-sample $t$-test — assume that two populations share the same variance. Before applying such methods, we need a formal way to test whether this assumption is reasonable. The F-test for two variances provides exactly this: a hypothesis test that compares the variability of two normally distributed populations. Beyond assumption checking, comparing variances arises naturally in quality control (is one manufacturing process more variable than another?) and in experimental design (does a treatment affect variability, not just the mean?).

## Assumptions

The F-test requires the following conditions:

1. **Normality:** Both populations follow normal distributions, $X_1 \sim N(\mu_1, \sigma_1^2)$ and $X_2 \sim N(\mu_2, \sigma_2^2)$.
2. **Independence:** The two samples are independent of each other, and observations within each sample are independent.
3. **Random sampling:** Both samples are obtained via simple random sampling from their respective populations.

Let $X_{1,1}, \ldots, X_{1,n_1}$ be a random sample of size $n_1$ from the first population, and $X_{2,1}, \ldots, X_{2,n_2}$ a random sample of size $n_2$ from the second population. Define the sample variances as

$$
S_1^2 = \frac{1}{n_1 - 1}\sum_{j=1}^{n_1}(X_{1,j} - \bar{X}_1)^2, \qquad S_2^2 = \frac{1}{n_2 - 1}\sum_{j=1}^{n_2}(X_{2,j} - \bar{X}_2)^2
$$

## Hypotheses

The null hypothesis states that the two population variances are equal:

$$
H_0: \sigma_1^2 = \sigma_2^2
$$

The alternative hypothesis takes one of three forms:

- **Two-sided:** $H_1: \sigma_1^2 \neq \sigma_2^2$
- **Right-tailed:** $H_1: \sigma_1^2 > \sigma_2^2$
- **Left-tailed:** $H_1: \sigma_1^2 < \sigma_2^2$

## Test Statistic

The intuition behind the F-test is straightforward: if the two populations share the same variance, then the ratio of the sample variances should be close to one. Large deviations from one provide evidence against $H_0$.

Under $H_0: \sigma_1^2 = \sigma_2^2$ and the normality assumption, the test statistic

$$
F = \frac{S_1^2}{S_2^2}
$$

follows an $F$-distribution with $n_1 - 1$ degrees of freedom in the numerator and $n_2 - 1$ degrees of freedom in the denominator:

$$
F \sim F_{n_1 - 1,\; n_2 - 1}
$$

This result follows from the fact that $(n_i - 1)S_i^2 / \sigma_i^2 \sim \chi^2_{n_i - 1}$ for normal populations, and the ratio of two independent chi-squared random variables (each divided by its degrees of freedom) defines the F-distribution.

## Rejection Regions

Let $F_{\alpha, d_1, d_2}$ denote the upper $\alpha$ critical value of the $F_{d_1, d_2}$ distribution, where $d_1 = n_1 - 1$ and $d_2 = n_2 - 1$.

- **Two-sided** ($H_1: \sigma_1^2 \neq \sigma_2^2$): Reject $H_0$ if $F > F_{\alpha/2, d_1, d_2}$ or $F < F_{1 - \alpha/2, d_1, d_2}$.
- **Right-tailed** ($H_1: \sigma_1^2 > \sigma_2^2$): Reject $H_0$ if $F > F_{\alpha, d_1, d_2}$.
- **Left-tailed** ($H_1: \sigma_1^2 < \sigma_2^2$): Reject $H_0$ if $F < F_{1 - \alpha, d_1, d_2}$.

!!! tip "Convention for Two-Sided Tests"
    A common simplification is to place the larger sample variance in the numerator, ensuring $F \geq 1$. In that case, only the right tail needs to be checked, and the test rejects $H_0$ if $F > F_{\alpha/2, d_1, d_2}$.

!!! example "Numerical Example"
    A manufacturer compares the consistency of two machines. Machine A produces $n_1 = 10$ items with sample variance $S_1^2 = 4.2$, and Machine B produces $n_2 = 12$ items with sample variance $S_2^2 = 1.8$. Test $H_0: \sigma_1^2 = \sigma_2^2$ vs. $H_1: \sigma_1^2 \neq \sigma_2^2$ at $\alpha = 0.05$.

    The test statistic is

    $$
    F = \frac{S_1^2}{S_2^2} = \frac{4.2}{1.8} = 2.333
    $$

    The degrees of freedom are $d_1 = 9$ and $d_2 = 11$. The critical value $F_{0.025, 9, 11} \approx 3.59$. Since $2.333 < 3.59$, we fail to reject $H_0$ at the 5% significance level. There is insufficient evidence to conclude that the two machines differ in variability.

## Caution: Sensitivity to Non-Normality

The F-test is **extremely sensitive** to departures from normality. Even mild skewness or heavy tails in the underlying populations can cause the actual Type I error rate to far exceed the nominal level $\alpha$. Simulation studies have shown that for moderately skewed distributions, the true rejection rate under $H_0$ can be two to three times the nominal rate.

Because of this fragility, more robust alternatives are preferred in practice:

- **Levene's test** replaces each observation with its absolute deviation from the group mean (or median) and then applies a standard ANOVA F-test to these deviations. This approach is robust to non-normality.
- **Brown-Forsythe test** is a variant of Levene's test that uses deviations from the group median rather than the mean, providing even greater robustness for skewed distributions.

!!! warning "When to Use the F-Test"
    Use the classical F-test only when you have strong evidence that both populations are normally distributed (for example, from a Shapiro-Wilk test or Q-Q plots). Otherwise, prefer Levene's or Brown-Forsythe test.

## Exercises

**Exercise 1.**
Two independent samples from normal populations yield $s_1^2 = 25$ ($n_1 = 16$) and $s_2^2 = 10$ ($n_2 = 21$). Test $H_0: \sigma_1^2 = \sigma_2^2$ versus $H_a: \sigma_1^2 \neq \sigma_2^2$ at $\alpha = 0.05$.

??? success "Solution to Exercise 1"
    The F-statistic is:

    $$
    F = \frac{s_1^2}{s_2^2} = \frac{25}{10} = 2.5
    $$

    Under $H_0$, $F \sim F_{n_1 - 1, n_2 - 1} = F_{15, 20}$.

    For a two-sided test at $\alpha = 0.05$, the critical values are $F_{0.025, 15, 20} \approx 2.57$ and $F_{0.975, 15, 20} \approx 1/F_{0.025, 20, 15} \approx 1/2.76 \approx 0.362$.

    Since $0.362 < 2.5 < 2.57$, the test statistic falls within the acceptance region. We fail to reject $H_0$ at the 5% level. There is insufficient evidence to conclude the variances differ.

---

**Exercise 2.**
Explain why the F-test for comparing two variances is sensitive to departures from normality. What alternative test is more robust?

??? success "Solution to Exercise 2"
    The F-test relies on the ratio $s_1^2/s_2^2$ following an F-distribution under $H_0$, which requires both populations to be exactly normal. The sample variance is sensitive to heavy tails and outliers (since it involves squared deviations), and the F-distribution of the ratio is particularly fragile: even mild departures from normality can inflate the Type I error rate substantially.

    Simulation studies show that for heavy-tailed distributions (e.g., $t$-distributions with small degrees of freedom), the actual rejection rate of the F-test at nominal $\alpha = 0.05$ can exceed 15-20%.

    **Robust alternatives** include:

    - **Levene's test:** Based on absolute deviations from group means.
    - **Brown-Forsythe test:** Based on absolute deviations from group medians (even more robust).
    - **Bartlett's test:** More powerful under normality but also sensitive to non-normality.

---

**Exercise 3.**
Derive the F-test statistic from the chi-squared distributions of the sample variances. That is, show that $(s_1^2/\sigma_1^2)/(s_2^2/\sigma_2^2) \sim F_{n_1-1, n_2-1}$ under normality.

??? success "Solution to Exercise 3"
    Under normality:

    $$
    \frac{(n_1-1)s_1^2}{\sigma_1^2} \sim \chi^2_{n_1-1}, \quad \frac{(n_2-1)s_2^2}{\sigma_2^2} \sim \chi^2_{n_2-1}
    $$

    and these are independent (since the samples are independent). By definition, the ratio of two independent chi-squared random variables, each divided by their degrees of freedom, follows an F-distribution:

    $$
    F = \frac{\chi^2_{n_1-1}/(n_1-1)}{\chi^2_{n_2-1}/(n_2-1)} = \frac{s_1^2/\sigma_1^2}{s_2^2/\sigma_2^2}
    $$

    Under $H_0: \sigma_1^2 = \sigma_2^2$, this simplifies to $F = s_1^2/s_2^2 \sim F_{n_1-1, n_2-1}$. $\square$

---

**Exercise 4.**
If $F = s_1^2/s_2^2 \sim F_{d_1, d_2}$, show that $1/F = s_2^2/s_1^2 \sim F_{d_2, d_1}$. Why does this property matter for the two-sided test?

??? success "Solution to Exercise 4"
    If $F = (U/d_1)/(V/d_2)$ where $U \sim \chi^2_{d_1}$ and $V \sim \chi^2_{d_2}$ are independent, then:

    $$
    \frac{1}{F} = \frac{V/d_2}{U/d_1} \sim F_{d_2, d_1}
    $$

    by the definition of the F-distribution with the roles of numerator and denominator swapped.

    This matters for the two-sided test because the F-distribution is not symmetric. The lower critical value $F_{\alpha/2, d_1, d_2}$ can be computed as $1/F_{1-\alpha/2, d_2, d_1}$, which is useful since many tables only provide upper-tail critical values. It also means that convention typically places the larger variance in the numerator ($F \geq 1$), converting the two-sided test to a one-sided test with doubled significance level.
