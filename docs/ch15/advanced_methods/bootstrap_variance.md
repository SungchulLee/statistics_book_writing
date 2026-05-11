# Bootstrap Variance Testing

The classical tests for variance (chi-square, F-test, Bartlett's) assume normality, and even the robust tests (Levene's, Brown-Forsythe) rely on asymptotic approximations for their reference distributions. The bootstrap offers an alternative that avoids distributional assumptions entirely by estimating the null distribution of the test statistic through resampling. This approach is especially useful when the sample size is small, the distribution is unknown, or the analyst wants to avoid the assumptions built into parametric tests.

## The Bootstrap Principle for Variance

The core idea is straightforward: if we want to test whether two populations have the same variance, we can resample the data under the null hypothesis and compute the test statistic many times. The collection of resampled test statistics approximates the null distribution, and the observed statistic is compared against this distribution to obtain a $p$-value.

The bootstrap does not assume any particular parametric form for the population. Instead, it treats the empirical distribution of the sample as an estimate of the true population distribution.

## One-Sample Bootstrap Test for Variance

To test $H_0\colon \sigma^2 = \sigma_0^2$ against $H_1\colon \sigma^2 \neq \sigma_0^2$:

**Step 1.** Compute the observed test statistic from the original sample of size $n$:

$$
T_{\text{obs}} = \frac{(n-1)S^2}{\sigma_0^2}
$$

**Step 2.** Generate $B$ bootstrap samples by resampling $n$ observations with replacement from the original sample. For each bootstrap sample $b = 1, \ldots, B$, compute the bootstrap sample variance $S_b^{*2}$ and the bootstrap test statistic:

$$
T_b^* = \frac{(n-1)S_b^{*2}}{S^2}
$$

The denominator uses $S^2$ (not $\sigma_0^2$) because the bootstrap samples are drawn from data that have variance $S^2$, not $\sigma_0^2$. This centers the bootstrap distribution around the null.

**Step 3.** Compute the bootstrap $p$-value as the proportion of bootstrap statistics at least as extreme as the observed statistic:

$$
p = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}(|T_b^* - (n-1)| \ge |T_{\text{obs}} - (n-1)|)
$$

## Two-Sample Bootstrap Test for Equal Variances

To test $H_0\colon \sigma_1^2 = \sigma_2^2$:

**Step 1.** Compute the observed variance ratio:

$$
F_{\text{obs}} = \frac{S_1^2}{S_2^2}
$$

**Step 2.** Pool the two samples to create a combined dataset that satisfies $H_0$ (equal variances). One approach is to center each group at zero and pool:

$$
\tilde{X}_{1j} = X_{1j} - \bar{X}_1, \qquad \tilde{X}_{2j} = X_{2j} - \bar{X}_2
$$

Then combine the centered residuals into a single pool $\{\tilde{X}_{11}, \ldots, \tilde{X}_{1n_1}, \tilde{X}_{21}, \ldots, \tilde{X}_{2n_2}\}$.

**Step 3.** For each bootstrap replicate $b = 1, \ldots, B$:

- Draw $n_1$ observations with replacement from the pool (bootstrap "group 1")
- Draw $n_2$ observations with replacement from the pool (bootstrap "group 2")
- Compute the bootstrap variance ratio $F_b^* = S_{1b}^{*2} / S_{2b}^{*2}$

**Step 4.** Compute the bootstrap $p$-value:

$$
p = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}\!\left(\left|\ln F_b^*\right| \ge \left|\ln F_{\text{obs}}\right|\right)
$$

Using the logarithm of the variance ratio ensures the test is symmetric: $F = 2$ and $F = 0.5$ represent the same degree of departure from equality.

!!! note "Why Pool Under the Null?"
    By pooling the centered residuals from both groups, we create a single population from which both bootstrap samples are drawn. This enforces $\sigma_1^2 = \sigma_2^2$ in the resampling scheme, generating the null distribution of the test statistic.

## Multi-Sample Extension

For $k > 2$ groups, the bootstrap approach generalizes naturally:

1. Center each group by subtracting its group mean.
2. Pool all centered residuals.
3. For each bootstrap replicate, draw $n_i$ observations from the pool for each group $i$.
4. Compute a test statistic (e.g., Bartlett's $T$ or Levene's $W$) on the bootstrap sample.
5. Compare the observed statistic to the bootstrap distribution.

This approach gives a bootstrap version of any classical or robust variance test, with the null distribution estimated nonparametrically.

## Advantages of the Bootstrap Approach

- **No distributional assumptions.** The bootstrap does not require normality or any other parametric form. It is valid for any continuous distribution.
- **Exact for finite samples.** Unlike asymptotic approximations, the bootstrap directly estimates the finite-sample null distribution.
- **Flexible.** Any test statistic can be bootstrapped, including custom statistics not covered by standard tables.

## Limitations

- **Computational cost.** Thousands of bootstrap replicates are needed for reliable $p$-values ($B \ge 1000$ for screening, $B \ge 10000$ for publication-quality results).
- **Discrete data.** For discrete distributions, resampling with replacement can produce bootstrap samples with unusual properties.
- **Very small samples.** With $n < 10$, the empirical distribution is a poor approximation of the population, and bootstrap results may be unreliable.

## Python Implementation

```python
import numpy as np

def bootstrap_variance_test(x, y, B=10000, seed=42):
    """Bootstrap test for equal variances in two samples."""
    rng = np.random.default_rng(seed)
    n1, n2 = len(x), len(y)

    # Observed statistic
    f_obs = np.var(x, ddof=1) / np.var(y, ddof=1)

    # Center and pool
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    pool = np.concatenate([x_centered, y_centered])

    # Bootstrap
    count = 0
    for _ in range(B):
        boot1 = rng.choice(pool, size=n1, replace=True)
        boot2 = rng.choice(pool, size=n2, replace=True)
        f_boot = np.var(boot1, ddof=1) / np.var(boot2, ddof=1)
        if abs(np.log(f_boot)) >= abs(np.log(f_obs)):
            count += 1

    p_value = count / B
    return f_obs, p_value

# Example
x = np.array([10, 12, 14, 11, 13, 15, 12, 10])
y = np.array([20, 28, 22, 35, 25, 18, 30, 22])

f_stat, p_val = bootstrap_variance_test(x, y)
print(f"Variance ratio: {f_stat:.4f}")
print(f"Bootstrap p-value: {p_val:.4f}")
```


## Exercises

**Exercise 1.**
Describe the bootstrap procedure for testing $H_0: \sigma_1^2 = \sigma_2^2$ for two independent samples.

??? success "Solution to Exercise 1"

    1. Compute the observed test statistic $T_{\text{obs}} = s_1^2/s_2^2$ (or $|s_1^2 - s_2^2|$).
    2. Pool both samples to create a combined dataset (enforcing $H_0$).
    3. For $b = 1, \dots, B$: draw bootstrap samples of sizes $n_1$ and $n_2$ from the pooled data, compute $T_b^*$.
    4. The p-value is the proportion of $T_b^*$ values as extreme as $T_{\text{obs}}$.

    This procedure is valid without normality because the bootstrap reference distribution adapts to the actual data distribution.

---

**Exercise 2.**
Why is the bootstrap particularly useful for variance testing compared to classical methods?

??? success "Solution to Exercise 2"
    Classical variance tests (chi-squared, F-test) are highly sensitive to non-normality -- even mild departures can severely distort Type I error rates. The bootstrap avoids distributional assumptions entirely, making it reliable for skewed, heavy-tailed, or otherwise non-normal data.

    Additionally, the bootstrap can test complex hypotheses about variances (ratios, functions of variances) that do not have simple classical test statistics.

---

**Exercise 3.**
How many bootstrap replicates $B$ are typically needed for reliable variance testing? What determines this choice?

??? success "Solution to Exercise 3"
    For hypothesis testing, $B = 1000$ is often sufficient for approximate p-values, but $B = 10{,}000$ or more is recommended for precise p-values (especially when testing at small $\alpha$ levels like 0.01).

    The required $B$ depends on: (1) the desired precision of the p-value ($\text{SE}(\hat{p}) \approx \sqrt{p(1-p)/B}$), (2) the significance level (smaller $\alpha$ requires larger $B$), and (3) the test statistic's variability. For $\alpha = 0.05$, $B = 2000$ gives p-value standard error of about 0.005.

---

**Exercise 4.**
A bootstrap test for equal variances produces a p-value of 0.047, while Levene's test gives $p = 0.12$. Discuss the possible reasons for disagreement.

??? success "Solution to Exercise 4"
    The tests may disagree because they test slightly different things and have different sensitivities. Levene's test is based on absolute deviations from group means/medians, while the bootstrap may use the variance ratio directly. Levene's test is designed to be robust to non-normality, but the bootstrap adapts more flexibly to the data's actual distribution.

    Other reasons: (1) the bootstrap p-value has Monte Carlo error (run with larger $B$ to check stability), (2) the data may have features (outliers, skewness) that affect the two statistics differently, (3) the tests have different power profiles against different alternatives.
