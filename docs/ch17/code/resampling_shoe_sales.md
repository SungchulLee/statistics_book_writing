# Resampling (Shoe Sales A/B Test)

## Overview

This page demonstrates three resampling techniques applied to a practical A/B testing scenario: an e-commerce company optimized shoe prices and wants to know whether weekly sales improved. We use a permutation test to assess statistical significance, compute an effect size for practical importance, and construct a bootstrap confidence interval for the mean difference. These nonparametric methods require no distributional assumptions and are well-suited for small-sample comparisons.

---

## The Data

Weekly shoe sales (units) for 12 weeks before and after price optimization:

$$
\text{Before}: \quad 23, 21, 19, 24, 35, 17, 18, 24, 33, 27, 21, 23
$$

$$
\text{After}: \quad 31, 28, 19, 24, 32, 27, 16, 28, 29, 26, 25, 27
$$

The observed difference in means is:

$$
\bar{x}_{\text{after}} - \bar{x}_{\text{before}} = 26.0 - 23.75 = 2.25 \text{ units}
$$

```python
import numpy as np

np.random.seed(42)
BEFORE = np.array([23, 21, 19, 24, 35, 17, 18, 24, 33, 27, 21, 23])
AFTER  = np.array([31, 28, 19, 24, 32, 27, 16, 28, 29, 26, 25, 27])

print(f"Before mean: {BEFORE.mean():.2f}")
print(f"After  mean: {AFTER.mean():.2f}")
print(f"Difference:  {AFTER.mean() - BEFORE.mean():.2f}")
```

---

## Permutation Test

The permutation test evaluates whether the observed difference could have arisen by chance. Under the null hypothesis $H_0\colon F_{\text{before}} = F_{\text{after}}$, the "before/after" labels are exchangeable.

**Procedure:**

1. Pool all $n_1 + n_2 = 24$ observations.
2. Randomly assign 12 to "before" and 12 to "after."
3. Compute the permuted mean difference.
4. Repeat $B$ times and compute the $p$-value:

$$
p = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}\!\bigl(\Delta^{(\pi_b)} \ge \Delta_{\text{obs}}\bigr)
$$

This is a one-sided test because we expect the optimization to increase (not decrease) sales.

```python
def permutation_test(before, after, n_perm=100_000):
    observed_diff = after.mean() - before.mean()
    combined = np.concatenate([before, after])
    n_before = len(before)
    count = 0
    perm_diffs = np.empty(n_perm)

    for i in range(n_perm):
        np.random.shuffle(combined)
        perm_diffs[i] = combined[n_before:].mean() - combined[:n_before].mean()
        if perm_diffs[i] >= observed_diff:
            count += 1

    p_value = count / n_perm
    return observed_diff, perm_diffs, p_value

obs_diff, perm_diffs, p_val = permutation_test(BEFORE, AFTER)
print(f"Observed difference: {obs_diff:.2f}")
print(f"Permutation p-value: {p_val:.4f}")
```

---

## Effect Size

Statistical significance alone does not tell us whether the effect is practically meaningful. The absolute difference and percentage change provide context:

$$
\text{Absolute difference} = \bar{x}_{\text{after}} - \bar{x}_{\text{before}}
$$

$$
\text{Percentage change} = \frac{\bar{x}_{\text{after}} - \bar{x}_{\text{before}}}{\bar{x}_{\text{before}}} \times 100\%
$$

```python
diff = AFTER.mean() - BEFORE.mean()
pct = diff / BEFORE.mean() * 100
print(f"Mean difference: {diff:.2f} units")
print(f"Percentage increase: {pct:.2f}%")
```

A 2.25-unit increase on a base of 23.75 corresponds to roughly a 9.5% lift, which may or may not be economically significant depending on profit margins and the cost of the optimization.

---

## Bootstrap Confidence Interval

The bootstrap provides a confidence interval for the true mean difference without assuming normality. We resample with replacement from each group independently:

1. Draw $n_1$ observations with replacement from "before."
2. Draw $n_2$ observations with replacement from "after."
3. Compute $\Delta^{(b)} = \bar{x}_{\text{after}}^{(b)} - \bar{x}_{\text{before}}^{(b)}$.
4. Repeat $B$ times and take percentiles:

$$
\text{CI}_{1-\alpha} = \left[q_{\alpha/2},\;\; q_{1-\alpha/2}\right]
$$

```python
def bootstrap_ci(before, after, n_boot=100_000, ci=90):
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        b = np.random.choice(before, size=len(before), replace=True)
        a = np.random.choice(after, size=len(after), replace=True)
        diffs[i] = a.mean() - b.mean()
    lo = (100 - ci) / 2
    hi = 100 - lo
    return diffs, np.percentile(diffs, [lo, hi])

boot_diffs, ci90 = bootstrap_ci(BEFORE, AFTER, ci=90)
_, ci95 = bootstrap_ci(BEFORE, AFTER, ci=95)

print(f"90% Bootstrap CI: [{ci90[0]:.2f}, {ci90[1]:.2f}]")
print(f"95% Bootstrap CI: [{ci95[0]:.2f}, {ci95[1]:.2f}]")
```

---

## Interpretation

- The **permutation test** $p$-value tells us whether the observed 2.25-unit increase is unlikely under random relabeling. If $p < 0.05$, we conclude the increase is statistically significant.
- The **effect size** of approximately 9.5% quantifies the practical magnitude of the change.
- The **bootstrap CI** tells us the range of plausible values for the true mean difference. If the 95% CI excludes zero, this is consistent with the permutation test rejecting $H_0$.
- Together, these three analyses provide a complete picture: significance, magnitude, and uncertainty. The resampling approach makes no normality assumption, which is important with only $n = 12$ observations per group.

---

## Exercises

**Exercise 1.** Convert the one-sided permutation test above into a two-sided test. Re-run it on the shoe sales data and compare the $p$-value to the one-sided version.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np

    np.random.seed(42)
    BEFORE = np.array([23, 21, 19, 24, 35, 17, 18, 24, 33, 27, 21, 23])
    AFTER  = np.array([31, 28, 19, 24, 32, 27, 16, 28, 29, 26, 25, 27])

    def permutation_test_two_sided(before, after, n_perm=100_000):
        observed_diff = after.mean() - before.mean()
        combined = np.concatenate([before, after])
        n_before = len(before)
        perm_diffs = np.empty(n_perm)
        for i in range(n_perm):
            np.random.shuffle(combined)
            perm_diffs[i] = combined[n_before:].mean() - combined[:n_before].mean()
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        return observed_diff, p_value

    diff, p_two = permutation_test_two_sided(BEFORE, AFTER)
    print(f"Two-sided p-value: {p_two:.4f}")
    ```

    The two-sided $p$-value counts permutations where $|\Delta^{(\pi)}| \ge |\Delta_{\text{obs}}|$, so it is approximately twice the one-sided $p$-value. The two-sided test is appropriate when we do not have a prior expectation about the direction of the effect. $\square$

---

**Exercise 2.** The data are paired (same weeks, before and after). Implement a paired permutation test that randomly flips the sign of each within-week difference $d_i = x_i^{\text{after}} - x_i^{\text{before}}$. Compare the $p$-value to the unpaired test.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np

    np.random.seed(42)
    BEFORE = np.array([23, 21, 19, 24, 35, 17, 18, 24, 33, 27, 21, 23])
    AFTER  = np.array([31, 28, 19, 24, 32, 27, 16, 28, 29, 26, 25, 27])

    def paired_perm_test(before, after, n_perm=100_000):
        d = after - before
        t_obs = np.mean(d)
        count = 0
        for _ in range(n_perm):
            signs = np.random.choice([-1, 1], size=len(d))
            if np.mean(signs * d) >= t_obs:
                count += 1
        return t_obs, count / n_perm

    t_obs, p_paired = paired_perm_test(BEFORE, AFTER)
    print(f"Paired differences: {AFTER - BEFORE}")
    print(f"Mean difference: {t_obs:.2f}")
    print(f"Paired permutation p-value: {p_paired:.4f}")
    ```

    The paired test uses within-week differences, removing week-to-week variability. If the "before" and "after" measurements are positively correlated across weeks, the paired test will be more powerful (smaller $p$-value) than the unpaired test. $\square$

---

**Exercise 3.** The bootstrap CI above uses the percentile method. Implement the **basic bootstrap CI** (also called the "pivotal" method), defined as:

$$
\text{CI} = \bigl(2\hat\theta - q_{1-\alpha/2},\;\; 2\hat\theta - q_{\alpha/2}\bigr)
$$

where $\hat\theta$ is the observed statistic and $q$ are bootstrap percentiles. Apply it to the shoe sales data and compare to the percentile CI.

??? success "Solution to Exercise 3"

    ```python
    import numpy as np

    np.random.seed(42)
    BEFORE = np.array([23, 21, 19, 24, 35, 17, 18, 24, 33, 27, 21, 23])
    AFTER  = np.array([31, 28, 19, 24, 32, 27, 16, 28, 29, 26, 25, 27])

    theta_hat = AFTER.mean() - BEFORE.mean()
    n_boot = 100_000
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        b = np.random.choice(BEFORE, len(BEFORE), replace=True)
        a = np.random.choice(AFTER, len(AFTER), replace=True)
        boot_diffs[i] = a.mean() - b.mean()

    # Percentile CI
    q_lo, q_hi = np.percentile(boot_diffs, [2.5, 97.5])
    print(f"Percentile 95% CI: [{q_lo:.2f}, {q_hi:.2f}]")

    # Basic (pivotal) CI
    basic_lo = 2 * theta_hat - q_hi
    basic_hi = 2 * theta_hat - q_lo
    print(f"Basic 95% CI:      [{basic_lo:.2f}, {basic_hi:.2f}]")
    ```

    The basic CI "reflects" the bootstrap distribution around the observed statistic. When the bootstrap distribution is symmetric, both methods give similar results. When it is skewed, they can differ, and the basic CI may have better coverage properties. $\square$

---

**Exercise 4.** Suppose the company wants to detect a minimum lift of \$2 per week (in units) with 80% power. Using a simulation-based approach, estimate the required sample size (number of weeks per group) for the permutation test at $\alpha = 0.05$.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np

    def power_estimate(n_weeks, true_diff=2.0, n_sims=1000, n_perm=2000):
        rejections = 0
        for _ in range(n_sims):
            before = np.random.normal(24, 5, n_weeks)
            after = np.random.normal(24 + true_diff, 5, n_weeks)
            obs = after.mean() - before.mean()
            combined = np.concatenate([before, after])
            count = 0
            for _ in range(n_perm):
                np.random.shuffle(combined)
                if combined[n_weeks:].mean() - combined[:n_weeks].mean() >= obs:
                    count += 1
            if count / n_perm < 0.05:
                rejections += 1
        return rejections / n_sims

    np.random.seed(42)
    for n in [12, 20, 30, 50, 75, 100]:
        pwr = power_estimate(n, true_diff=2.0, n_sims=500, n_perm=1000)
        print(f"n={n:3d}: power = {pwr:.3f}")
    ```

    With $\sigma \approx 5$ and a true difference of 2, approximately 50--75 weeks per group are needed to achieve 80% power. The original 12-week design has low power for detecting a difference this small, which explains why the $p$-value may be borderline. $\square$

---

**Exercise 5.** Prove that the permutation test controls the Type I error at exactly $\alpha$ when the null hypothesis of exchangeability holds. That is, show $P(p \le \alpha \mid H_0) \le \alpha$ for all $\alpha \in (0, 1)$.

??? success "Solution to Exercise 5"

    Under $H_0$, the "before" and "after" labels are exchangeable: all $\binom{n_1+n_2}{n_1}$ label assignments are equally likely. Let $T_0$ be the observed test statistic and $T_1, \ldots, T_B$ be the statistics from $B$ random permutations.

    By exchangeability, the augmented set $\{T_0, T_1, \ldots, T_B\}$ consists of $B + 1$ exchangeable random variables. The $p$-value is:

    $$
    p = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}(T_b \ge T_0)
    $$

    Consider the rank of $T_0$ among $\{T_0, T_1, \ldots, T_B\}$. By exchangeability, $T_0$ is equally likely to have any rank from 1 to $B + 1$. The event $\{p \le \alpha\}$ occurs when at least $(1 - \alpha)B$ of the $T_b$ exceed $T_0$, i.e., when $T_0$ has rank at most $\lfloor \alpha B \rfloor + 1$.

    Therefore:

    $$
    P(p \le \alpha) = \frac{\lfloor \alpha B \rfloor + 1}{B + 1} \le \frac{\alpha B + 1}{B + 1} \le \alpha + \frac{1}{B+1}
    $$

    For any practical $B$ (e.g., $B = 100{,}000$), this is essentially $\alpha$. In the exact permutation test (enumerating all permutations), the bound holds with equality: $P(p \le \alpha) \le \alpha$ exactly. $\square$
