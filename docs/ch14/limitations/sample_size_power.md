# Sample Size Effects on Power

## The Core Problem

Normality tests are hypothesis tests, and like all hypothesis tests, their ability to detect departures from normality depends on the sample size $n$. This creates a practical dilemma: when $n$ is small and normality matters most, the tests lack power to detect violations; when $n$ is large and the CLT provides robustness, the tests become so powerful that they reject normality for trivial deviations. Understanding this relationship is essential for interpreting normality test results correctly.

## Power of a Normality Test

The **power** of a normality test is the probability of rejecting the null hypothesis $H_0$: "the data are normally distributed" when the data truly come from a non-normal distribution. Formally, if $F$ denotes the true distribution of the data and $\Phi$ denotes the normal CDF, then

$$
\text{Power} = P\left(\text{Reject } H_0 \mid F \neq \Phi\right)
$$

Power depends on three factors:

1. **Sample size $n$.** Larger samples provide more information about the shape of the distribution, increasing the ability to detect departures.
2. **Significance level $\alpha$.** A larger $\alpha$ increases power but also increases the Type I error rate.
3. **Degree of non-normality.** The further the true distribution $F$ is from normal, the easier it is to detect.

## Small-Sample Behavior

When $n$ is small (say $n < 30$), normality tests have low power. Even substantial departures from normality may go undetected. Consider a sample of $n = 15$ drawn from an exponential distribution, which is strongly right-skewed. The Shapiro-Wilk test may fail to reject $H_0$ in a large fraction of such samples, not because the data are normal, but because the test lacks sufficient evidence to detect the departure.

This low power is problematic because small-sample inference is precisely the setting where the normality assumption matters most. The $t$-distribution used in $t$-tests and confidence intervals is derived under exact normality, and the CLT provides little protection when $n$ is small.

??? warning "Failing to Reject Does Not Confirm Normality"
    A non-significant result on a normality test means "insufficient evidence to reject normality," not "the data are normal." This distinction is especially important with small samples, where the test may lack the power to detect even large departures.

## Large-Sample Behavior

As $n$ grows, the power of any consistent normality test converges to 1 for any fixed non-normal distribution. This means that, with enough data, the test will eventually reject $H_0$ no matter how close the true distribution is to normal. Formally, for any distribution $F \neq \Phi$ and any $\alpha > 0$,

$$
\lim_{n \to \infty} P\left(\text{Reject } H_0 \mid F\right) = 1
$$

This consistency property implies that for very large samples, normality tests almost always reject. A dataset of $n = 10{,}000$ drawn from a distribution with excess kurtosis of 0.1 (barely distinguishable from normal for practical purposes) will likely produce a significant Shapiro-Wilk or Anderson-Darling test result.

The rejection tells us that the data are not exactly normal, which is almost always true for real data. It does not tell us whether the departure is large enough to affect the validity of subsequent statistical procedures.

## The Practical Paradox

This creates a paradox in the standard workflow of "test normality, then proceed with parametric inference":

| Sample size | Power of normality test | Need for normality | Practical situation |
|---|---|---|---|
| Small ($n < 30$) | Low | High | Test cannot detect violations that matter |
| Moderate ($30 \leq n \leq 100$) | Moderate | Moderate | Test is informative and useful |
| Large ($n > 500$) | Very high | Low (CLT helps) | Test rejects for irrelevant departures |

The moderate sample size range is the "sweet spot" where normality tests provide the most useful information relative to the practical need.

## Quantifying the Effect

To illustrate, consider the power of the Shapiro-Wilk test at $\alpha = 0.05$ against a $t$-distribution with $\nu$ degrees of freedom. As $\nu \to \infty$, the $t$-distribution converges to the normal, so smaller $\nu$ represents a larger departure from normality.

For $\nu = 5$ (moderate departure with excess kurtosis $= 6$):

- At $n = 20$, power is approximately 0.30
- At $n = 50$, power is approximately 0.75
- At $n = 200$, power is approximately 0.99

For $\nu = 30$ (mild departure with excess kurtosis $= 0.4$):

- At $n = 20$, power is approximately 0.06 (barely above $\alpha$)
- At $n = 50$, power is approximately 0.08
- At $n = 200$, power is approximately 0.20
- At $n = 2000$, power is approximately 0.85

## Python Example

```python
import numpy as np
from scipy import stats

# ===================================================================
# Demonstrate how the power of the Shapiro-Wilk test changes with n
# ===================================================================

np.random.seed(42)
n_simulations = 5000
alpha = 0.05

sample_sizes = [15, 30, 50, 100, 200, 500, 1000]

if __name__ == "__main__":
    print("Power of Shapiro-Wilk test vs. t(5) distribution")
    print(f"{'n':>6}  {'Power':>8}")
    print("-" * 16)

    for n in sample_sizes:
        rejections = 0
        for _ in range(n_simulations):
            sample = stats.t.rvs(df=5, size=n)
            # Shapiro-Wilk limited to n <= 5000 in scipy
            if n <= 5000:
                _, p = stats.shapiro(sample)
                if p < alpha:
                    rejections += 1
        power = rejections / n_simulations
        print(f"{n:>6}  {power:>8.3f}")
```

## Guidance for Practitioners

Based on the sample size and power relationship, the following approach is recommended:

1. **For small samples ($n < 30$)**: Do not rely solely on formal normality tests. Use Q-Q plots and histograms for visual assessment. Consider nonparametric alternatives if the Q-Q plot shows clear departures.

2. **For moderate samples ($30 \leq n \leq 200$)**: Formal normality tests are most informative in this range. Combine a Shapiro-Wilk or Anderson-Darling test with graphical diagnostics. A significant result here likely reflects a meaningful departure.

3. **For large samples ($n > 500$)**: Expect normality tests to reject. Focus on the magnitude of the departure rather than the $p$-value. Use effect-size measures such as excess kurtosis and skewness, and assess whether the departure is large enough to affect your specific inferential procedure.

## Summary

The power of normality tests increases monotonically with sample size, creating a practical paradox. Small samples lack the power to detect violations that would invalidate parametric inference. Large samples detect every trivial departure, even when the CLT ensures that parametric procedures remain valid. Effective use of normality testing requires matching the interpretation to the sample size and supplementing formal tests with graphical methods and domain knowledge.
