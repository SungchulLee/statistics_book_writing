# Introduction to Variance Testing

## Motivation

Several statistical tests have been developed to assess whether the observed differences in variances across groups or populations are statistically significant. These tests evaluate whether the variability in one or more samples is higher, lower, or equal compared to others. Each test has specific assumptions about the data (e.g., normality, independence) and is suited to different types of analysis.

## Overview of Variance Tests

### Chi-Square Test for Variance

This test determines if the variance of a single population differs from a specified value. It assumes that the data follow a normal distribution and is highly sensitive to deviations from this assumption.

The test statistic for a sample of size $n$ and sample variance $s^2$ is:

$$
\chi^2 = \frac{(n - 1) s^2}{\sigma_0^2}
$$

where $\sigma_0^2$ is the hypothesized population variance.

### F-Test

The F-test compares the variances of two independent samples. It tests the null hypothesis that the two population variances are equal. This test is also sensitive to the assumption of normality and requires that the two samples be independent.

The test statistic is the ratio of the two sample variances:

$$
F = \frac{s_1^2}{s_2^2}
$$

where $s_1^2$ and $s_2^2$ are the sample variances of the two groups.

### Bartlett's Test

Bartlett's test assesses equality of variances across multiple groups under the assumption that the data are normally distributed. It is highly sensitive to violations of normality, meaning even slight deviations from normality can lead to misleading results. Despite this sensitivity, Bartlett's test is often used when normality can be reasonably assumed.

The test statistic is:

$$
T = \frac{(N - k) \ln(S_p^2) - \sum_{i=1}^k (n_i - 1) \ln(S_i^2)}{1 + \frac{1}{3(k - 1)} \left( \sum_{i=1}^k \frac{1}{n_i - 1} - \frac{1}{N - k} \right)}
$$

where $N$ is the total sample size, $k$ is the number of groups, $S_p^2$ is the pooled variance, and $S_i^2$ are the sample variances.

### Levene's Test

Levene's test is a robust alternative to the F-test for equality of variances. It is less sensitive to deviations from normality, making it more appropriate when the data are not normally distributed. Levene's test is based on the absolute deviations of each observation from the group mean.

The hypotheses for Levene's test are:

- $H_0$: Population variances are equal.
- $H_1$: Population variances are not equal.

### Brown–Forsythe Test

This test is a modification of Levene's test, replacing the group mean with the group median to further reduce sensitivity to outliers. It is particularly useful when outliers are present in the data, as it provides a more robust measure of variance equality.

### Fligner–Killeen Test

The Fligner–Killeen test is a non-parametric test based on ranks. It is highly robust and appropriate when data do not meet the assumption of normality. Like the Brown–Forsythe test, it is effective for datasets with outliers or non-normal distributions.

## Choosing the Right Test

| Test | Samples | Normality Required | Robust to Outliers | Use Case |
|------|---------|-------------------|-------------------|----------|
| Chi-Square | 1 | Yes | No | Single population variance vs. hypothesized value |
| F-Test | 2 | Yes | No | Comparing variances of two independent groups |
| Bartlett's | $\geq 2$ | Yes | No | Multi-group variance equality (normal data) |
| Levene's | $\geq 2$ | No | Moderate | Multi-group variance equality (general) |
| Brown–Forsythe | $\geq 2$ | No | Yes | Multi-group variance equality (outliers present) |
| Fligner–Killeen | $\geq 2$ | No | Yes | Non-parametric multi-group variance equality |
