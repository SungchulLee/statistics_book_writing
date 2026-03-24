# Sensitivity vs Practical Significance

## The Distinction

A normality test may reject the null hypothesis and still leave the practitioner with a perfectly valid analysis. The key insight is that **statistical significance of non-normality** and **practical significance of non-normality** are different things. A statistically significant departure means that the data deviate detectably from a normal distribution. A practically significant departure means that the deviation is large enough to meaningfully affect the validity of a downstream inferential procedure such as a $t$-test, ANOVA, or confidence interval.

This distinction mirrors the broader distinction between statistical and practical significance in hypothesis testing, but it carries special weight in the context of normality testing because the normality check is itself a preliminary step.

## Robustness of Parametric Procedures

A statistical procedure is said to be **robust** to violations of an assumption if its performance (Type I error rate, coverage probability, power) remains close to the nominal values even when the assumption is violated. Different procedures have very different levels of robustness to non-normality.

### Tests for Means

The one-sample $t$-test is remarkably robust to non-normality, particularly for symmetric distributions. Simulation studies show that, even for samples as small as $n = 15$, the actual Type I error rate of the $t$-test remains close to $\alpha = 0.05$ when the underlying distribution is symmetric with moderate tails. The reason is that the sampling distribution of $\bar{X}$ converges to normality via the CLT, and the $t$-statistic inherits this robustness.

For skewed distributions, the $t$-test is less robust. The actual Type I error rate can exceed $\alpha$ for small $n$ when the population is strongly skewed. However, even in this case, the distortion is typically modest (e.g., actual $\alpha$ of 0.07 when nominal $\alpha$ is 0.05) and decreases as $n$ grows.

### Tests for Variances

Tests involving variances are far more sensitive to non-normality. The chi-squared test for a single variance and the $F$-test for comparing two variances both rely on the assumption that

$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}
$$

This distributional result holds exactly only under normality. When the population has heavy tails, the actual distribution of $(n-1)S^2/\sigma^2$ has heavier tails than the chi-squared, leading to substantially inflated Type I error rates. A non-normality that barely affects the $t$-test can severely distort the $F$-test.

??? warning "The F-Test for Variances Is Not Robust"
    The $F$-test for comparing two population variances is one of the least robust classical procedures. Even mild non-normality can cause the actual Type I error rate to be two or three times the nominal $\alpha$. For this reason, the Levene or Brown-Forsythe test is preferred in practice.

### ANOVA

One-way ANOVA is moderately robust to non-normality when the group sizes are equal and the distributions have similar shapes. The $F$-statistic is based on a ratio of mean squares, and the CLT helps stabilize the numerator and denominator when $n$ is not too small. Unequal group sizes combined with non-normality and heteroscedasticity create the most problematic scenario.

## When Do Departures Matter?

The practical impact of non-normality depends on the combination of three factors:

1. **Type of departure.** Skewness tends to be more problematic than symmetric heavy tails for mean-based tests, because skewness shifts the center of the sampling distribution. Heavy tails primarily affect variance-based procedures.

2. **Sample size.** The CLT provides increasing protection as $n$ grows, but the rate of convergence depends on the severity of the departure. For the $t$-test, $n \geq 30$ is often sufficient for symmetric heavy-tailed distributions, while skewed distributions may require $n \geq 100$.

3. **The procedure being used.** The hierarchy of sensitivity, from least to most sensitive, is approximately:

    - Tests and CIs for means (most robust)
    - ANOVA $F$-tests (moderately robust with balanced designs)
    - Regression coefficient tests (moderately robust)
    - Tests for variances (least robust)

## A Decision Framework

The following framework helps determine whether a detected departure from normality is practically meaningful:

**Step 1: Identify the inferential goal.** Are you testing a mean, comparing means, testing a variance, or performing regression? The answer determines how sensitive your procedure is to non-normality.

**Step 2: Assess the sample size.** With large $n$, mean-based inference is protected by the CLT regardless of the test result.

**Step 3: Characterize the departure.** Is it skewness, heavy tails, outliers, or multimodality? Each has different implications.

**Step 4: Evaluate the magnitude.** Excess kurtosis below 1 and absolute skewness below 0.5 are generally considered mild. Excess kurtosis above 3 or absolute skewness above 1 warrants further investigation.

## Python Example

```python
import numpy as np
from scipy import stats

# ===================================================================
# Compare actual vs nominal Type I error rates under non-normality
# for the t-test (robust) and the chi-squared variance test (fragile)
# ===================================================================

np.random.seed(42)
n = 30
alpha = 0.05
n_simulations = 10_000
df_true = 5  # t-distribution with heavy tails

rejections_t = 0
rejections_chi2 = 0

for _ in range(n_simulations):
    sample = stats.t.rvs(df=df_true, size=n)

    # t-test for mean = 0 (true mean is 0)
    _, p_t = stats.ttest_1samp(sample, popmean=0)
    if p_t < alpha:
        rejections_t += 1

    # Chi-squared test for variance = df/(df-2) (true variance)
    true_var = df_true / (df_true - 2)
    chi2_stat = (n - 1) * np.var(sample, ddof=1) / true_var
    p_chi2 = 2 * min(
        stats.chi2.cdf(chi2_stat, df=n - 1),
        1 - stats.chi2.cdf(chi2_stat, df=n - 1)
    )
    if p_chi2 < alpha:
        rejections_chi2 += 1

if __name__ == "__main__":
    print(f"Data: t-distribution with df = {df_true}, n = {n}")
    print(f"Nominal alpha: {alpha}")
    print(f"Actual Type I error (t-test):          "
          f"{rejections_t / n_simulations:.4f}")
    print(f"Actual Type I error (chi-squared test): "
          f"{rejections_chi2 / n_simulations:.4f}")
```

The simulation demonstrates that the $t$-test maintains a Type I error rate close to $\alpha = 0.05$ even with heavy-tailed $t(5)$ data, while the chi-squared variance test has a substantially inflated error rate under the same conditions.

## Summary

Not all departures from normality are created equal. The practical significance of non-normality depends on which statistical procedure is being used, how large the sample is, and what type of departure is present. Mean-based tests are robust to a wide range of non-normal distributions, while variance-based tests are fragile. When a normality test rejects, the appropriate response is not automatically to abandon parametric methods, but rather to assess whether the specific departure matters for the specific analysis at hand.
