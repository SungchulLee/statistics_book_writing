# Bootstrap Hypothesis Tests

## Overview

Bootstrap hypothesis testing constructs the null distribution by resampling the observed data rather than relying on parametric assumptions. This page covers three core bootstrap tests: a one-sample test for the mean, a two-sample test for comparing means, and a bootstrap standard-error estimate for the median. These methods are especially valuable when the sampling distribution of the test statistic is unknown or difficult to derive analytically.

## One-Sample Bootstrap Test

We wish to test $H_0\colon \mu = \mu_0$ against a two-sided alternative. The procedure is:

1. **Center** the data under the null: $x_i^0 = x_i - \bar x + \mu_0$.
2. **Resample** from $\{x_1^0, \ldots, x_n^0\}$ with replacement, $B$ times, computing the mean of each resample.
3. **Compute** the $p$-value as the fraction of bootstrap means at least as extreme as the observed mean:

$$
p = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}\!\bigl(|\bar x^{*(b)} - \mu_0| \ge |\bar x - \mu_0|\bigr)
$$

The centering step is crucial: it ensures the resampled data have mean $\mu_0$ on average, thereby enforcing the null hypothesis in the bootstrap world.

```python
def bootstrap_mean_test(data, mu_0=0, n_boot=10_000, alpha=0.05):
    """Bootstrap test for H0: mean = mu_0. Returns p-value (two-sided)."""
    n = len(data)
    centered = data - data.mean() + mu_0
    boot_means = np.array([
        np.mean(centered[np.random.randint(0, n, n)])
        for _ in range(n_boot)
    ])
    obs_mean = data.mean()
    p_value = np.mean(np.abs(boot_means - mu_0) >= np.abs(obs_mean - mu_0))
    return obs_mean, p_value, boot_means
```

## Two-Sample Bootstrap Test

To test $H_0\colon \mu_x = \mu_y$, we pool the two samples and resample from the pooled data. Under $H_0$ the group labels are exchangeable:

1. **Pool** $\{x_1,\ldots,x_m,y_1,\ldots,y_n\}$ into a single set of size $m + n$.
2. **Resample** $m + n$ observations with replacement from the pool, assigning the first $m$ to group $X$ and the remaining $n$ to group $Y$.
3. **Compute** the difference in means $\bar x^{*(b)} - \bar y^{*(b)}$ for each replicate.
4. **$p$-value**: fraction of bootstrap differences at least as extreme as the observed difference.

$$
p = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}\!\bigl(|\bar x^{*(b)} - \bar y^{*(b)}| \ge |\bar x - \bar y|\bigr)
$$

```python
def bootstrap_two_sample(x, y, n_boot=10_000):
    """Bootstrap test for H0: mean(x) = mean(y)."""
    obs_diff = x.mean() - y.mean()
    pooled = np.concatenate([x, y])
    n_x = len(x)
    boot_diffs = []
    for _ in range(n_boot):
        perm = pooled[np.random.randint(0, len(pooled), len(pooled))]
        boot_diffs.append(perm[:n_x].mean() - perm[n_x:].mean())
    boot_diffs = np.array(boot_diffs)
    p_value = np.mean(np.abs(boot_diffs) >= np.abs(obs_diff))
    return obs_diff, p_value, boot_diffs
```

## Bootstrap Standard Error of the Median

The median has no simple closed-form standard error. The bootstrap provides a direct estimate:

$$
\widehat{\text{SE}}_{\text{boot}}(\text{median}) = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\bigl(\tilde x^{*(b)} - \overline{\tilde x^*}\bigr)^2}
$$

where $\tilde x^{*(b)}$ is the median of the $b$-th bootstrap sample. The bootstrap bias is:

$$
\widehat{\text{bias}} = \overline{\tilde x^*} - \tilde x
$$

```python
def bootstrap_se_median(data, n_boot=10_000):
    """Estimate the standard error of the median via bootstrap."""
    n = len(data)
    boot_medians = np.array([
        np.median(data[np.random.randint(0, n, n)])
        for _ in range(n_boot)
    ])
    se = boot_medians.std(ddof=1)
    bias = boot_medians.mean() - np.median(data)
    return se, bias, boot_medians
```

## Demonstration

The script applies the three methods to synthetic data:

```python
# 1. One-sample test: exponential data shifted by 2
data = np.random.exponential(scale=5, size=50) + 2
mu_0 = 5.0
obs, p, boots = bootstrap_mean_test(data, mu_0)

# 2. Two-sample test: two normal populations
x = np.random.normal(52, 10, 40)
y = np.random.normal(48, 10, 40)
diff, p2, boots2 = bootstrap_two_sample(x, y)

# 3. Bootstrap SE of the median: log-normal income
income = np.random.lognormal(mean=10.5, sigma=0.8, size=200)
se_med, bias, boot_med = bootstrap_se_median(income)
```

## Interpretation

- The one-sample bootstrap test rejects $H_0\colon \mu = 5$ when the data mean is far from 5. Because the data are exponentially distributed (not normal), the bootstrap approach avoids reliance on the $t$-distribution.
- The two-sample bootstrap test detects the 4-unit shift between the two normal populations. Its $p$-value is typically close to that of the two-sample $t$-test when normality holds.
- The bootstrap SE of the median is especially useful for skewed distributions like the log-normal. There is no simple formula for $\text{SE}(\text{median})$ in this case, making the bootstrap the method of choice.

A general principle: when parametric assumptions hold, bootstrap and classical tests agree. When assumptions are violated, the bootstrap is often more reliable.

## Exercises

**Exercise 1.** Draw a sample of size $n = 40$ from a standard normal distribution. Perform the one-sample bootstrap test for $H_0\colon \mu = 0$. Repeat for $H_0\colon \mu = 0.5$. Report the $p$-values and explain why the results differ.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    np.random.seed(7)
    data = np.random.normal(0, 1, 40)

    _, p0, _ = bootstrap_mean_test(data, mu_0=0.0)
    _, p05, _ = bootstrap_mean_test(data, mu_0=0.5)

    print(f"H0: mu = 0,   p-value = {p0:.4f}")
    print(f"H0: mu = 0.5, p-value = {p05:.4f}")
    ```

    Since the true mean is 0, the test for $H_0\colon \mu = 0$ yields a large $p$-value (typically $> 0.05$), meaning we fail to reject. For $H_0\colon \mu = 0.5$, the sample mean is far from 0.5 on average, so the $p$-value is smaller, often leading to rejection. The centering step shifts the resampled data to be centered at $\mu_0$, and if the observed mean is distant from $\mu_0$, the bootstrap null distribution rarely produces values that extreme. $\square$

---

**Exercise 2.** The two-sample bootstrap test in the code resamples from the pooled data *with replacement*. Explain the conceptual difference between this approach and a permutation test that shuffles labels *without replacement*. Under what conditions do the two approaches give similar $p$-values?

??? success "Solution to Exercise 2"

    In the bootstrap two-sample test, we draw $m + n$ observations *with replacement* from the pooled set, then split into groups of size $m$ and $n$. This means some observations appear multiple times while others may be absent from a given replicate.

    In a permutation test, we shuffle the $m + n$ labels without replacement, so every observation appears exactly once in each permuted dataset. This preserves the exact composition of the pooled sample.

    The two approaches give similar $p$-values when the sample sizes $m$ and $n$ are moderately large, because the bootstrap distribution of the difference in means converges to the permutation distribution as $m, n \to \infty$. For small samples, the permutation test is exact (conditional on the data), while the bootstrap test is approximate. The permutation test is also more natural under the null hypothesis of exchangeability, since it directly models the randomization mechanism. $\square$

---

**Exercise 3.** Derive the formula for the bootstrap bias of the median. Show that if the bootstrap distribution of the median is symmetric about $\tilde x$ (the sample median), the bias is zero.

??? success "Solution to Exercise 3"

    The bootstrap bias is defined as:

    $$
    \widehat{\text{bias}} = E^*[\tilde x^*] - \tilde x = \overline{\tilde x^*} - \tilde x
    $$

    where $E^*$ denotes expectation under the bootstrap distribution (the empirical distribution of the data) and $\overline{\tilde x^*} = \frac{1}{B}\sum_{b=1}^{B}\tilde x^{*(b)}$ estimates this expectation.

    If the bootstrap distribution of the median is symmetric about $\tilde x$, then for every bootstrap replicate producing $\tilde x^{*(b)} = \tilde x + \delta$, there is (approximately) a matching replicate producing $\tilde x - \delta$. Therefore $E^*[\tilde x^*] = \tilde x$, which gives:

    $$
    \widehat{\text{bias}} = \tilde x - \tilde x = 0
    $$

    $\square$

---

**Exercise 4.** Generate 200 observations from a $\text{Gamma}(2, 1)$ distribution. Use the bootstrap to estimate the standard error of both the mean and the median. Compare with the theoretical SE of the mean, $\sigma / \sqrt{n}$. Why is there no analogous formula for the median?

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    np.random.seed(12)
    data = np.random.gamma(shape=2, scale=1, size=200)

    # Bootstrap SE of the mean
    boot_means = np.array([
        np.mean(np.random.choice(data, 200, replace=True))
        for _ in range(10_000)
    ])
    se_mean_boot = boot_means.std(ddof=1)

    # Bootstrap SE of the median
    se_med, _, _ = bootstrap_se_median(data, n_boot=10_000)

    # Theoretical SE of the mean: sigma / sqrt(n)
    # For Gamma(2,1): sigma = sqrt(2)
    se_mean_theory = np.sqrt(2) / np.sqrt(200)

    print(f"Bootstrap SE(mean):    {se_mean_boot:.4f}")
    print(f"Theoretical SE(mean):  {se_mean_theory:.4f}")
    print(f"Bootstrap SE(median):  {se_med:.4f}")
    ```

    The bootstrap SE of the mean closely matches $\sigma/\sqrt{n} = \sqrt{2}/\sqrt{200} \approx 0.1$. There is no simple formula for the SE of the median because the median's sampling distribution depends on the density of the population at the median value, $f(m)$. The asymptotic formula is $\text{SE}(\text{median}) \approx 1/(2f(m)\sqrt{n})$, but this requires knowledge of $f(m)$, which is typically unknown. The bootstrap sidesteps this issue entirely by estimating the SE empirically. $\square$

---

**Exercise 5.** Prove that the one-sample bootstrap test is consistent: as $n \to \infty$, if $\mu \neq \mu_0$, the $p$-value converges to 0. (Hint: consider the behavior of $|\bar x - \mu_0|$ and the bootstrap distribution of $|\bar x^* - \mu_0|$ under the centered data.)

??? success "Solution to Exercise 5"

    Under the alternative $\mu \neq \mu_0$, by the law of large numbers $\bar x \to \mu$ as $n \to \infty$, so:

    $$
    |\bar x - \mu_0| \to |\mu - \mu_0| > 0
    $$

    The centered data $x_i^0 = x_i - \bar x + \mu_0$ have sample mean exactly $\mu_0$. By the bootstrap CLT, the resampled means $\bar x^{*(b)}$ from the centered data satisfy:

    $$
    \sqrt{n}(\bar x^{*(b)} - \mu_0) \xrightarrow{d} N(0, \sigma^2)
    $$

    where $\sigma^2$ is the population variance. Therefore $|\bar x^{*(b)} - \mu_0| = O_p(n^{-1/2})$, which converges to 0.

    Meanwhile, $|\bar x - \mu_0| \to |\mu - \mu_0| > 0$, a fixed positive constant. For large $n$, the probability that a bootstrap replicate produces $|\bar x^{*(b)} - \mu_0| \ge |\bar x - \mu_0|$ becomes negligible:

    $$
    p = P^*\!\bigl(|\bar x^{*(b)} - \mu_0| \ge |\bar x - \mu_0|\bigr) \to 0
    $$

    Hence the test is consistent. $\square$
