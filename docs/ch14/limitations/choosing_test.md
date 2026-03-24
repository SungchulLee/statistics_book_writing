# Choosing the Right Test

## Why Test Selection Matters

Several normality tests are available, each with different strengths and sensitivities. The Shapiro-Wilk test, Anderson-Darling test, Kolmogorov-Smirnov test, and Jarque-Bera test all assess normality but differ in what types of departures they detect most effectively, what sample sizes they accommodate, and how they are implemented in standard software. Choosing the right test depends on the sample size, the suspected type of departure, and the inferential context.

## Overview of Common Normality Tests

### Shapiro-Wilk Test

The Shapiro-Wilk test evaluates the null hypothesis $H_0$: "the data come from a normal distribution" by computing a test statistic $W$ that measures how well the ordered sample values match the expected order statistics of a normal distribution. The statistic is

$$
W = \frac{\left(\sum_{i=1}^{n} a_i X_{(i)}\right)^2}{\sum_{i=1}^{n} (X_i - \bar{X})^2}
$$

where the weights $a_i$ are derived from the covariance matrix of the normal order statistics. Values of $W$ close to 1 indicate normality; small values indicate departure.

**Strengths:** Generally the most powerful test for detecting departures from normality, particularly for small to moderate samples ($n \leq 50$). It is sensitive to both skewness and heavy tails.

**Limitations:** Most implementations restrict $n$ to at most 5000. The test does not indicate the type of departure (skewness vs. kurtosis).

### Anderson-Darling Test

The Anderson-Darling test is based on the empirical distribution function (EDF). It computes a weighted measure of the distance between the EDF $F_n(x)$ and the hypothesized normal CDF $\Phi(x)$:

$$
A^2 = -n - \sum_{i=1}^{n} \frac{2i - 1}{n} \left[\ln \Phi(Z_{(i)}) + \ln\left(1 - \Phi(Z_{(n+1-i)})\right)\right]
$$

where $Z_{(i)} = (X_{(i)} - \bar{X}) / S$ are the standardized order statistics. The Anderson-Darling statistic places more weight on the tails of the distribution compared to the Kolmogorov-Smirnov test.

**Strengths:** More sensitive to tail departures than the KS test. Available for larger sample sizes. Performs well across a range of alternatives.

**Limitations:** Slightly less powerful than Shapiro-Wilk for small samples. The critical values depend on whether the parameters are estimated or known.

### Kolmogorov-Smirnov Test

The Kolmogorov-Smirnov (KS) test measures the maximum absolute distance between the EDF and the hypothesized CDF:

$$
D = \sup_x \left| F_n(x) - \Phi(x) \right|
$$

When testing against a normal distribution with estimated parameters, the Lilliefors correction must be applied because the standard KS critical values assume fully specified parameters.

**Strengths:** Conceptually simple. Applicable to any continuous distribution. No sample size restriction.

**Limitations:** Generally less powerful than Shapiro-Wilk and Anderson-Darling for detecting normality violations. It is most sensitive to departures near the center of the distribution and relatively insensitive to tail departures.

### Jarque-Bera Test

The Jarque-Bera test is based on the sample skewness and kurtosis. The test statistic is

$$
JB = \frac{n}{6}\left(\hat{\gamma}^2 + \frac{(\hat{\kappa} - 3)^2}{4}\right)
$$

where $\hat{\gamma}$ is the sample skewness and $\hat{\kappa}$ is the sample kurtosis. Under $H_0$, $JB \overset{d}{\to} \chi^2_2$ as $n \to \infty$.

**Strengths:** Directly targets skewness and kurtosis, the two most common types of departure from normality. Computationally simple. Widely used in econometrics and finance.

**Limitations:** It is an asymptotic test, so it is unreliable for small samples ($n < 30$). It is insensitive to departures that do not affect skewness or kurtosis (e.g., bimodality with symmetric, mesokurtic components).

## Comparison Table

| Test | Best sample size | Sensitive to | Tail sensitivity | Computational cost |
|---|---|---|---|---|
| Shapiro-Wilk | $n \leq 5000$ | General departures | High | Moderate |
| Anderson-Darling | Any $n$ | Tail departures | Very high | Low |
| Kolmogorov-Smirnov (Lilliefors) | Any $n$ | Center departures | Low | Low |
| Jarque-Bera | $n \geq 30$ | Skewness and kurtosis | Moderate | Very low |

## Decision Framework

The following guidelines help select the appropriate test:

**Step 1: Consider the sample size.**

- If $n < 30$: Use the Shapiro-Wilk test. It has the best power in small samples. Supplement with a Q-Q plot.
- If $30 \leq n \leq 5000$: Shapiro-Wilk remains a strong default. The Anderson-Darling test is a good alternative, especially if tail behavior is the primary concern.
- If $n > 5000$: Use the Anderson-Darling test (no sample size restriction). The Jarque-Bera test is also appropriate if the concern is skewness or kurtosis.

**Step 2: Consider the suspected departure.**

- If heavy tails are suspected (e.g., financial data): Prefer Anderson-Darling or Jarque-Bera. The KS test is too insensitive in the tails.
- If skewness is the primary concern: Jarque-Bera directly tests for it.
- If the type of departure is unknown: Shapiro-Wilk is the most broadly powerful choice.

**Step 3: Consider the inferential context.**

- If the downstream analysis is a $t$-test or ANOVA: A general-purpose test (Shapiro-Wilk) is appropriate.
- If the downstream analysis involves variance or tail risk: A tail-sensitive test (Anderson-Darling) is more relevant.

??? tip "Always Supplement with Graphical Methods"
    No single normality test replaces graphical assessment. A Q-Q plot reveals the type and location of departures (tails, center, skewness), while a histogram shows the overall shape. The combination of a formal test and a Q-Q plot provides the most informative assessment.

## Python Example

```python
import numpy as np
from scipy import stats

# ===================================================================
# Run multiple normality tests on the same dataset and compare results
# ===================================================================

np.random.seed(42)
n = 100

# Generate data from a t-distribution (heavy tails)
data = stats.t.rvs(df=5, size=n)

if __name__ == "__main__":
    # Shapiro-Wilk
    sw_stat, sw_p = stats.shapiro(data)
    print(f"Shapiro-Wilk:      W = {sw_stat:.4f}, p = {sw_p:.4f}")

    # Anderson-Darling
    ad_result = stats.anderson(data, dist="norm")
    print(f"Anderson-Darling:  A2 = {ad_result.statistic:.4f}, "
          f"critical (5%) = {ad_result.critical_values[2]:.4f}")

    # Kolmogorov-Smirnov (Lilliefors via kstest with estimated params)
    ks_stat, ks_p = stats.kstest(data, "norm", args=(np.mean(data), np.std(data)))
    print(f"KS (estimated):    D = {ks_stat:.4f}, p = {ks_p:.4f}")

    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(data)
    print(f"Jarque-Bera:       JB = {jb_stat:.4f}, p = {jb_p:.4f}")
```

The output illustrates that different tests may give different $p$-values for the same dataset. The Shapiro-Wilk and Anderson-Darling tests tend to be the most sensitive, while the KS test may fail to detect the same departure.

## Summary

No single normality test is universally optimal. The Shapiro-Wilk test is the best default for small to moderate samples due to its broad power. The Anderson-Darling test excels at detecting tail departures and has no sample size restriction. The Jarque-Bera test is efficient when skewness or kurtosis is the suspected issue, but it requires at least moderate sample sizes. The Kolmogorov-Smirnov test, while widely known, is generally the least powerful for normality testing. In all cases, formal tests should be accompanied by graphical diagnostics.
