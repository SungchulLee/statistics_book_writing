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
