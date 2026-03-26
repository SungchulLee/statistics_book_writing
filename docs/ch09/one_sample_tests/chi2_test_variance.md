# Chi-Square Test for the Population Variance


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

## Overview

In manufacturing, controlling process variability is often as important as controlling the process average. A machine that produces bolts with mean diameter on target but excessive variance yields too many parts outside tolerance --- and too many rejects. Similarly, in finance, an investor may want to test whether a portfolio's return volatility exceeds a specified threshold. The chi-square test for variance provides a formal procedure for testing whether a population variance $\sigma^2$ equals, exceeds, or falls below a hypothesized value $\sigma_0^2$. Unlike tests for the mean, which are relatively robust to mild non-normality, this test is highly sensitive to departures from the normal distribution.

## Hypotheses

Let $X_1, X_2, \ldots, X_n$ be a random sample from a $N(\mu, \sigma^2)$ population. The null hypothesis specifies a particular value for the variance:

$$
H_0\colon \sigma^2 = \sigma_0^2
$$

The alternative hypothesis takes one of three forms depending on the research question:

| Alternative | Interpretation |
|---|---|
| $H_1\colon \sigma^2 \neq \sigma_0^2$ | Two-sided: variance differs from $\sigma_0^2$ |
| $H_1\colon \sigma^2 > \sigma_0^2$ | Right-sided: variance exceeds $\sigma_0^2$ |
| $H_1\colon \sigma^2 < \sigma_0^2$ | Left-sided: variance is below $\sigma_0^2$ |

## Test Statistic

The test statistic measures how far the sample variance $S^2$ is from the hypothesized value $\sigma_0^2$, scaled by the degrees of freedom. Intuitively, if $\sigma^2 = \sigma_0^2$, then $S^2$ should be close to $\sigma_0^2$, and the ratio $(n-1)S^2 / \sigma_0^2$ should be close to $n-1$ (the mean of the $\chi^2_{n-1}$ distribution).

$$
\chi^2 = \frac{(n-1)S^2}{\sigma_0^2}
$$

Under $H_0$ and the normality assumption, this statistic follows a chi-square distribution with $n - 1$ degrees of freedom:

$$
\chi^2 = \frac{(n-1)S^2}{\sigma_0^2} \sim \chi^2_{n-1}
$$

This result follows from the fact that $(n-1)S^2/\sigma^2 = \sum_{i=1}^n (X_i - \bar{X})^2/\sigma^2$ is a sum of squared standard normals with one linear constraint (the deviations sum to zero), yielding $n-1$ degrees of freedom.

## Rejection Regions

At significance level $\alpha$, the rejection region depends on the alternative:

**Two-sided** ($H_1\colon \sigma^2 \neq \sigma_0^2$): Reject $H_0$ if

$$
\chi^2 < \chi^2_{1-\alpha/2,\, n-1} \quad \text{or} \quad \chi^2 > \chi^2_{\alpha/2,\, n-1}
$$

**Right-sided** ($H_1\colon \sigma^2 > \sigma_0^2$): Reject $H_0$ if

$$
\chi^2 > \chi^2_{\alpha,\, n-1}
$$

**Left-sided** ($H_1\colon \sigma^2 < \sigma_0^2$): Reject $H_0$ if

$$
\chi^2 < \chi^2_{1-\alpha,\, n-1}
$$

Here $\chi^2_{p,\, n-1}$ denotes the value such that $P(\chi^2_{n-1} \leq \chi^2_{p,\, n-1}) = p$, i.e., the $p$-th quantile of the chi-square distribution.

!!! note "Asymmetry of the Chi-Square Distribution"
    Unlike the normal and $t$ distributions, the chi-square distribution is not symmetric. This means the two-sided rejection region uses different critical values for the lower and upper tails, and the two-sided test cannot be expressed as a simple absolute-value condition.

## Worked Example

A manufacturer claims that the variance of fill weights for cereal boxes is $\sigma_0^2 = 4$ grams$^2$ (i.e., $\sigma_0 = 2$ grams). A quality inspector samples $n = 25$ boxes and finds a sample variance of $S^2 = 6.1$ grams$^2$. Test whether the variance exceeds the claimed value at $\alpha = 0.05$.

**Step 1.** State the hypotheses:

$$
H_0\colon \sigma^2 = 4 \qquad H_1\colon \sigma^2 > 4
$$

**Step 2.** Compute the test statistic:

$$
\chi^2 = \frac{(25 - 1)(6.1)}{4} = \frac{24 \times 6.1}{4} = \frac{146.4}{4} = 36.6
$$

**Step 3.** Find the critical value. For a right-sided test at $\alpha = 0.05$ with $24$ degrees of freedom:

$$
\chi^2_{0.05,\, 24} = 36.415
$$

**Step 4.** Make the decision. Since $\chi^2 = 36.6 > 36.415$, we reject $H_0$. There is sufficient evidence at the 5% level to conclude that the population variance exceeds 4 grams$^2$.

The $p$-value is $P(\chi^2_{24} > 36.6) \approx 0.048$.

## Assumptions and Sensitivity

The chi-square variance test requires:

- **Normality**: The population must be normally distributed. This is the most critical assumption.
- **Random sampling**: The observations must be independent and identically distributed.
- **Known $\sigma_0^2$**: The hypothesized variance $\sigma_0^2$ is specified, not estimated from data.

!!! warning "Extreme Sensitivity to Non-Normality"
    The chi-square test for variance is **not robust** to departures from normality. Even mild skewness or heavy tails can cause the actual Type I error rate to differ substantially from the nominal $\alpha$. The kurtosis of the population directly affects the variance of $S^2$: for a distribution with excess kurtosis $\kappa$, $\text{Var}(S^2) \approx 2\sigma^4(1 + \kappa/2)/(n-1)$, which can be much larger than the chi-square theory predicts ($\kappa = 0$ for the normal). When normality is in doubt, consider alternatives such as Levene's test or bootstrap-based methods.
