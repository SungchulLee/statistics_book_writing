# Sample Size Effects on Power

## The Core Problem

Normality tests are hypothesis tests, and like all hypothesis tests, their ability to detect departures from normality depends on the sample size $n$. This creates a practical dilemma: when $n$ is small and normality matters most, the tests lack power to detect violations; when $n$ is large and the CLT provides robustness, the tests become so powerful that they reject normality for trivial deviations. Understanding this relationship is essential for interpreting normality test results correctly.

## Power of a Normality Test

The **power** of a normality test is the probability of rejecting the null hypothesis $H_0$: "the data are normally distributed" when the data truly come from a non-normal distribution. Formally, if $F$ denotes the true distribution of the data and $\mathcal{N}$ denotes the normal CDF, then

$$
\text{Power} = P\left(\text{Reject } H_0 \mid F \neq \mathcal{N}\right)
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

As $n$ grows, the power of any consistent normality test converges to 1 for any fixed non-normal distribution. This means that, with enough data, the test will eventually reject $H_0$ no matter how close the true distribution is to normal. Formally, for any distribution $F \neq \mathcal{N}$ and any $\alpha > 0$,

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


## Exercises

**Exercise 1.**
A Shapiro-Wilk test on $n = 500$ observations of mildly skewed data gives $p = 0.001$. On $n = 20$ observations from the same distribution, $p = 0.35$. Explain the discrepancy.

??? success "Solution to Exercise 1"
    The discrepancy is due to the relationship between power and sample size. With $n = 500$, the test has very high power to detect even small departures from normality. The mild skewness, though practically insignificant, is statistically detectable. With $n = 20$, the test has low power and cannot detect the same mild departure.

    The p-value is not a measure of the degree of non-normality; it measures the evidence against normality given the sample size. The same degree of skewness produces a tiny p-value with large $n$ and a large p-value with small $n$. This is why effect-size measures (actual skewness and kurtosis values) are more informative than p-values for assessing practical normality.

---

**Exercise 2.**
Plot the power of the Shapiro-Wilk test as a function of $n$ for a $t_5$ alternative. Describe the general shape of the curve.

??? success "Solution to Exercise 2"
    The power curve starts near $\alpha$ (the significance level) for very small $n$ (where the test cannot distinguish $t_5$ from normal) and increases monotonically toward 1 as $n$ grows.

    For the $t_5$ distribution (moderately heavy-tailed): power is approximately $\alpha = 0.05$ at $n = 5$, rises to about 0.3-0.4 at $n = 30$, reaches 0.8 around $n = 80$-$100$, and exceeds 0.95 by $n = 200$.

    The curve is S-shaped (sigmoid) on a linear scale: slow initial growth, steep middle section, and saturation near 1. The steepness depends on how different the alternative is from normal -- more extreme alternatives (e.g., $t_3$) produce steeper curves.

---

**Exercise 3.**
Explain the concept of a "power analysis for normality testing." Is it commonly performed in practice?

??? success "Solution to Exercise 3"
    A power analysis for normality testing would determine the sample size needed to detect a specific departure from normality (e.g., excess kurtosis of 2) with a given probability (e.g., 80% power) at a given significance level.

    It is **rarely performed** in practice for several reasons: (1) the researcher usually does not know the specific alternative distribution in advance; (2) the goal of normality testing is typically to assess whether normal-based methods are reliable, not to identify the true distribution; (3) power tables for normality tests against specific alternatives are available but not widely used.

    Instead, practitioners rely on rules of thumb: Shapiro-Wilk has good power for $n \geq 20$, and for $n > 200$, formal tests are overpowered and visual methods are preferred.

---

**Exercise 4.**
For a fixed degree of non-normality, how does the p-value of a normality test scale with sample size $n$? Give an approximate relationship.

??? success "Solution to Exercise 4"
    For a fixed alternative (fixed departure from normality), the test statistic grows approximately as $\sqrt{n}$ (for many normality tests, the standardized statistic scales with $\sqrt{n}$). This means the p-value decreases roughly exponentially with $n$.

    More precisely, the test statistic $T_n$ satisfies $T_n \approx \sqrt{n} \cdot \delta + Z$ where $\delta$ is the non-centrality parameter (measuring the departure) and $Z$ is noise. The p-value is approximately $P(Z > c - \sqrt{n}\delta)$, which decreases to 0 as $n \to \infty$ for any $\delta > 0$.

    Practical implication: doubling the sample size roughly doubles the evidence against normality (in terms of the test statistic), making rejection inevitable for large enough $n$, regardless of how close to normal the data are.
