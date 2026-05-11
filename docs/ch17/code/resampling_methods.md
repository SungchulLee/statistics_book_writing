# Resampling Methods Comparison

## Overview

This page compares bootstrap and permutation resampling methods side by side. It covers bootstrap confidence intervals (normal, percentile, basic, and BCa) with coverage simulations, two-sample and paired permutation tests, a permutation test for correlation, and a direct comparison of bootstrap CIs versus permutation $p$-values. The goal is to show when and why each method is appropriate.

## Bootstrap Standard Error and Confidence Intervals

Given a sample $x_1, \ldots, x_n$, the bootstrap standard error of a statistic $\hat\theta$ is:

$$
\widehat{\text{SE}}_{\text{boot}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\bigl(\hat\theta^{*(b)} - \overline{\hat\theta^*}\bigr)^2}
$$

Four CI methods are available:

**Normal interval.** Uses the bootstrap SE with a normal quantile:

$$
\hat\theta \pm z_{1-\alpha/2}\cdot\widehat{\text{SE}}_{\text{boot}}
$$

**Percentile interval.** Reads directly from quantiles of the bootstrap distribution:

$$
\bigl[\hat\theta^*_{\alpha/2},\;\hat\theta^*_{1-\alpha/2}\bigr]
$$

**Basic (pivotal) interval.** Reflects quantiles around $\hat\theta$:

$$
\bigl[2\hat\theta - \hat\theta^*_{1-\alpha/2},\;2\hat\theta - \hat\theta^*_{\alpha/2}\bigr]
$$

**BCa interval.** Adjusts for bias ($z_0$) and acceleration ($a$) using the jackknife:

$$
\alpha_j = \mathcal{N}\!\left(z_0 + \frac{z_0 + z_{\alpha_j}}{1 - a(z_0 + z_{\alpha_j})}\right)
$$

```python
def bootstrap_ci_demo(data=None, B=10_000, alpha=0.05):
    """Demonstrate all four bootstrap CI methods for the mean and median."""
    if data is None:
        data = np.random.exponential(scale=3.0, size=50)

    n = len(data)
    theta_hat = np.mean(data)
    z = stats.norm.ppf(1 - alpha / 2)

    boot_means = np.array([
        np.mean(np.random.choice(data, n, replace=True))
        for _ in range(B)
    ])
    se_boot = boot_means.std(ddof=1)

    # Normal
    ci_normal = (theta_hat - z * se_boot, theta_hat + z * se_boot)
    # Percentile
    ci_pct = (np.percentile(boot_means, 100 * alpha / 2),
              np.percentile(boot_means, 100 * (1 - alpha / 2)))
    # Basic
    ci_basic = (2 * theta_hat - np.percentile(boot_means, 100 * (1 - alpha / 2)),
                2 * theta_hat - np.percentile(boot_means, 100 * alpha / 2))
```

## Bootstrap Coverage Simulation

A coverage simulation checks whether the nominal confidence level matches the actual proportion of intervals that contain the true parameter. For each of $N$ simulations:

1. Draw a fresh sample from the known population.
2. Build a bootstrap CI.
3. Check whether the true parameter falls inside.

The empirical coverage is:

$$
\widehat{\text{coverage}} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}\!\bigl(\theta \in \text{CI}_i\bigr)
$$

```python
def bootstrap_coverage(n=30, B_boot=2000, n_sim=2000, true_mu=3.0, true_scale=3.0):
    """Check empirical coverage of bootstrap CIs for the mean of Exp(3)."""
    alpha = 0.05
    z = stats.norm.ppf(1 - alpha / 2)
    coverage = {"Normal": 0, "Percentile": 0, "Basic": 0, "t-interval": 0}

    for _ in range(n_sim):
        data = np.random.exponential(true_scale, n)
        theta = data.mean()
        boot = np.array([
            np.random.choice(data, n, replace=True).mean()
            for _ in range(B_boot)
        ])
        se = boot.std(ddof=1)

        # Check each method
        lo, hi = theta - z * se, theta + z * se
        if lo <= true_mu <= hi:
            coverage["Normal"] += 1
        # ... (similarly for other methods)
```

Typical results for $n = 30$ from $\text{Exp}(3)$ show that the percentile and normal methods slightly under-cover (around 92--93%), while the parametric $t$-interval achieves close to 95%.

## Two-Sample Permutation Test

The permutation test pools both samples, shuffles labels, and computes the test statistic under each permutation:

```python
def permutation_test_two_sample(x, y, B=10_000, stat_func=None):
    """Two-sample permutation test for difference in means."""
    if stat_func is None:
        stat_func = lambda a, b: np.mean(a) - np.mean(b)
    t_obs = stat_func(x, y)
    pooled = np.concatenate([x, y])
    count = 0
    for _ in range(B):
        perm = np.random.permutation(pooled)
        t_perm = stat_func(perm[:len(x)], perm[len(x):])
        if abs(t_perm) >= abs(t_obs):
            count += 1
    p_value = count / B
    return t_obs, p_value
```

Applied to treatment versus control data, the permutation $p$-value is typically close to the Welch $t$-test $p$-value.

## Permutation Test for Correlation

To test $H_0\colon \rho = 0$, we permute one variable while keeping the other fixed:

$$
p = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}\!\bigl(|r^{(\pi_b)}| \ge |r_{\text{obs}}|\bigr)
$$

```python
def permutation_test_correlation(x, y, B=10_000):
    """Permutation test for Pearson correlation."""
    r_obs = np.corrcoef(x, y)[0, 1]
    count = 0
    for _ in range(B):
        y_perm = np.random.permutation(y)
        r_perm = np.corrcoef(x, y_perm)[0, 1]
        if abs(r_perm) >= abs(r_obs):
            count += 1
    return r_obs, count / B
```

## Paired Permutation Test (Sign-Flip)

For paired data $(x_i, y_i)$, the differences $d_i = x_i - y_i$ should be symmetric about 0 under $H_0$. We randomly flip signs:

$$
T^{(\pi)} = \frac{1}{n}\sum_{i=1}^{n} s_i\,d_i, \qquad s_i \in \{-1, +1\} \text{ uniformly}
$$

```python
def paired_permutation_test(x, y, B=10_000):
    """Paired permutation test via sign-flipping differences."""
    d = x - y
    t_obs = np.mean(d)
    n = len(d)
    count = 0
    for _ in range(B):
        signs = np.random.choice([-1, 1], size=n)
        t_perm = np.mean(signs * d)
        if abs(t_perm) >= abs(t_obs):
            count += 1
    return t_obs, count / B
```

## Bootstrap vs. Permutation: Side by Side

The two resampling strategies answer different questions:

| Aspect | Bootstrap | Permutation |
|---|---|---|
| **Goal** | Estimate a parameter or its uncertainty | Test a null hypothesis |
| **Output** | Confidence interval | $p$-value |
| **Resampling** | With replacement from each group | Shuffle labels without replacement |
| **Assumptions** | Sample is representative | Exchangeability under $H_0$ |

When both are applied to the same two-sample comparison, a bootstrap CI that excludes 0 and a permutation test that rejects at the same $\alpha$ should generally agree.

```python
def bootstrap_vs_permutation_comparison(x, y, B=10_000):
    """Compare bootstrap CI with permutation p-value."""
    diff_obs = np.mean(x) - np.mean(y)

    # Bootstrap CI for difference
    boot_diffs = np.array([
        np.mean(np.random.choice(x, len(x), True))
        - np.mean(np.random.choice(y, len(y), True))
        for _ in range(B)
    ])
    ci = np.percentile(boot_diffs, [2.5, 97.5])

    # Permutation test
    pooled = np.concatenate([x, y])
    count = sum(
        abs(np.mean(p[:len(x)]) - np.mean(p[len(x):])) >= abs(diff_obs)
        for p in (np.random.permutation(pooled) for _ in range(B))
    )
    p_perm = count / B
```

## Interpretation

- **Coverage simulations** reveal that bootstrap CIs can under-cover for skewed distributions with small $n$. The BCa and $t$-interval methods tend to be closer to the nominal level.
- **Permutation tests** for means and correlation produce $p$-values that closely match their parametric counterparts when distributional assumptions hold.
- **Sign-flip tests** for paired data are the permutation analogue of the paired $t$-test and are valid even when differences are non-normal.
- The **bootstrap and permutation approaches complement each other**: use the bootstrap for estimation (CIs, SEs) and permutation tests for hypothesis testing.

## Exercises

**Exercise 1.** Run the coverage simulation with $n = 100$ instead of $n = 30$. How does increasing the sample size affect the coverage of the percentile method? Explain using the central limit theorem.

??? success "Solution to Exercise 1"

    ```python
    bootstrap_coverage(n=100, B_boot=2000, n_sim=2000, true_mu=3.0, true_scale=3.0)
    ```

    With $n = 100$, the percentile method's coverage improves to approximately 94--95%, much closer to the nominal 95%. By the central limit theorem, $\bar x$ is approximately $N(\mu, \sigma^2/n)$ for large $n$. The bootstrap distribution of $\bar x^*$ mirrors this normal shape with decreasing skewness, so the percentile quantiles become accurate approximations of the true sampling quantiles. For $n = 30$ the exponential distribution's skewness causes noticeable distortion in the bootstrap distribution, but by $n = 100$ the normal approximation is much better. $\square$

---

**Exercise 2.** Modify `permutation_test_two_sample` to use the Welch $t$-statistic instead of the raw difference of means as the test statistic. Apply both versions to data with unequal variances: $X \sim N(5, 1)$ and $Y \sim N(5, 9)$ with $n_x = 20$ and $n_y = 50$. Compare the $p$-values.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(42)
    x = np.random.normal(5, 1, 20)
    y = np.random.normal(5, 3, 50)

    def welch_t(a, b):
        n1, n2 = len(a), len(b)
        return (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1)/n1 + b.var(ddof=1)/n2)

    t_obs_diff, p_diff, _ = permutation_test_two_sample(x, y, B=10_000)
    t_obs_welch, p_welch, _ = permutation_test_two_sample(
        x, y, B=10_000, stat_func=welch_t
    )

    print(f"Diff of means: p = {p_diff:.4f}")
    print(f"Welch t:       p = {p_welch:.4f}")
    ```

    Under $H_0$ (both means are 5), both versions should yield large $p$-values. However, the Welch $t$-statistic accounts for unequal variances, making it a more powerful test statistic when variances differ. In repeated simulations with a true mean difference, the Welch version will have higher power. Under $H_0$, both control the Type I error rate, but the Welch version is more robust to heteroscedasticity. $\square$

---

**Exercise 3.** The paired permutation test randomly flips signs of the differences $d_i$. With $n$ pairs, how many distinct permutations exist? For $n = 10$, is it feasible to enumerate all of them? Write code that computes the exact $p$-value by exhaustive enumeration.

??? success "Solution to Exercise 3"

    With $n$ pairs, each difference can be either kept or flipped, giving $2^n$ distinct sign assignments. For $n = 10$, $2^{10} = 1024$, which is easily enumerable.

    ```python
    import numpy as np
    from itertools import product

    before = np.array([82, 78, 91, 85, 73, 88, 79, 95, 84, 76])
    after  = np.array([88, 82, 95, 89, 78, 91, 84, 98, 90, 81])
    d = after - before
    t_obs = np.mean(d)

    count = 0
    total = 0
    for signs in product([-1, 1], repeat=len(d)):
        signs = np.array(signs)
        t_perm = np.mean(signs * d)
        if abs(t_perm) >= abs(t_obs):
            count += 1
        total += 1

    p_exact = count / total
    print(f"Exact p-value: {p_exact:.6f} ({count}/{total})")
    ```

    This gives the exact $p$-value without any Monte Carlo error. For $n > 20$ or so, exhaustive enumeration becomes impractical ($2^{20} > 10^6$), and random sampling of sign flips is preferred. $\square$

---

**Exercise 4.** The bootstrap-vs-permutation comparison claims that a bootstrap CI excluding 0 and a permutation test rejecting at $\alpha = 0.05$ should agree. Construct a scenario where they disagree. (Hint: consider a case where the CI is barely on one side of 0.)

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    np.random.seed(99)

    # Small sample, small effect
    x = np.random.normal(0.3, 1, 15)
    y = np.random.normal(0.0, 1, 15)

    # Bootstrap CI
    boot_diffs = np.array([
        np.mean(np.random.choice(x, len(x), True))
        - np.mean(np.random.choice(y, len(y), True))
        for _ in range(10_000)
    ])
    ci = np.percentile(boot_diffs, [2.5, 97.5])

    # Permutation test
    pooled = np.concatenate([x, y])
    diff_obs = x.mean() - y.mean()
    count = 0
    for _ in range(10_000):
        perm = np.random.permutation(pooled)
        if abs(perm[:15].mean() - perm[15:].mean()) >= abs(diff_obs):
            count += 1
    p_perm = count / 10_000

    print(f"Bootstrap 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"CI excludes 0: {ci[0] > 0 or ci[1] < 0}")
    print(f"Permutation p-value: {p_perm:.4f}")
    print(f"Perm rejects at 5%: {p_perm < 0.05}")
    ```

    Disagreement occurs when the observed difference is near the boundary. The bootstrap CI and permutation test use different resampling mechanisms: the bootstrap resamples within each group (preserving the group structure), while the permutation test pools and relabels. With small samples or borderline effects, the bootstrap CI may just include or exclude 0 while the permutation $p$-value hovers around 0.05. This is not a contradiction but reflects the different finite-sample properties of the two methods. $\square$

---

**Exercise 5.** Prove that for the permutation test of Pearson correlation, permuting $y$ while holding $x$ fixed generates the correct null distribution. Specifically, show that under $H_0\colon \rho = 0$ with the assumption that $(x_i, y_i)$ are independent, the joint distribution is invariant to permutations of the $y$ values.

??? success "Solution to Exercise 5"

    Under $H_0\colon \rho = 0$, the variables $X$ and $Y$ are independent. This means the joint density factors:

    $$
    f_{X,Y}(x_i, y_i) = f_X(x_i)\,f_Y(y_i)
    $$

    The joint likelihood of the observed data is:

    $$
    L = \prod_{i=1}^{n} f_X(x_i)\,f_Y(y_i) = \left(\prod_{i=1}^{n} f_X(x_i)\right)\left(\prod_{i=1}^{n} f_Y(y_i)\right)
    $$

    Now consider any permutation $\pi$ of $\{1, \ldots, n\}$. The likelihood of the permuted data $(x_i, y_{\pi(i)})$ is:

    $$
    L_\pi = \prod_{i=1}^{n} f_X(x_i)\,f_Y(y_{\pi(i)}) = \left(\prod_{i=1}^{n} f_X(x_i)\right)\left(\prod_{i=1}^{n} f_Y(y_{\pi(i)})\right)
    $$

    Since multiplication is commutative, $\prod_{i=1}^{n} f_Y(y_{\pi(i)}) = \prod_{i=1}^{n} f_Y(y_i)$. Therefore $L_\pi = L$ for all permutations $\pi$.

    This means all $n!$ pairings of $x$ and $y$ values are equally likely under $H_0$, which is exactly the exchangeability condition required by the permutation test. $\square$
