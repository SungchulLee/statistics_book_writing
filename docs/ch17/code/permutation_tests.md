# Permutation Test Demonstrations

## Overview

Permutation tests assess statistical significance by comparing an observed test statistic to its distribution under random relabeling of the data. Unlike parametric tests, they make no assumptions about the underlying distribution of the data. This page demonstrates three permutation tests: a two-sample test for difference of means, a multi-group test analogous to one-way ANOVA, and a test for comparing two proportions in an A/B testing context.

## Two-Sample Permutation Test

Given two samples $x_1, \ldots, x_m$ and $y_1, \ldots, y_n$, we test:

$$
H_0\colon F_X = F_Y \quad \text{vs} \quad H_1\colon F_X \neq F_Y
$$

The test statistic is the difference of means $T_{\text{obs}} = \bar x - \bar y$. Under $H_0$ the group labels are exchangeable, so we:

1. **Pool** all $m + n$ observations.
2. **Shuffle** the pool and assign the first $m$ to group $X$, the rest to group $Y$.
3. **Compute** $T^{(\pi)} = \bar x^{(\pi)} - \bar y^{(\pi)}$ for each permutation.
4. **$p$-value** (two-sided):

$$
p = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}\!\bigl(|T^{(\pi_b)}| \ge |T_{\text{obs}}|\bigr)
$$

```python
def perm_test_two_sample(x, y, n_perm=5000):
    """Two-sample permutation test for difference of means."""
    obs_diff = x.mean() - y.mean()
    pooled = np.concatenate([x, y])
    n_x = len(x)
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        np.random.shuffle(pooled)
        perm_diffs[i] = pooled[:n_x].mean() - pooled[n_x:].mean()
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    return obs_diff, p_value, perm_diffs
```

## Multi-Group Permutation Test

For $k$ groups with sizes $n_1, \ldots, n_k$, an ANOVA-like permutation test uses the **variance of group means** as the test statistic:

$$
T = \text{Var}(\bar x_1, \bar x_2, \ldots, \bar x_k)
$$

Under $H_0$ (all group distributions are identical), permuting group labels should not systematically change this statistic. The $p$-value is one-sided because larger values of $T$ indicate greater between-group differences:

$$
p = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}\!\bigl(T^{(\pi_b)} \ge T_{\text{obs}}\bigr)
$$

```python
def perm_test_multi_group(groups, n_perm=5000):
    """Multi-group permutation test (variance of group means)."""
    pooled = np.concatenate(groups)
    sizes = [len(g) for g in groups]
    obs_var = np.var([g.mean() for g in groups])
    perm_vars = np.empty(n_perm)
    for i in range(n_perm):
        np.random.shuffle(pooled)
        idx = 0
        means = []
        for s in sizes:
            means.append(pooled[idx:idx + s].mean())
            idx += s
        perm_vars[i] = np.var(means)
    p_value = np.mean(perm_vars >= obs_var)
    return obs_var, p_value, perm_vars
```

## Permutation Test for Proportions

In an A/B test with binary outcomes (conversion or not), group A has $n_A$ users with $c_A$ conversions, and group B has $n_B$ users with $c_B$ conversions. The observed difference in conversion rates is:

$$
T_{\text{obs}} = \frac{c_A}{n_A} - \frac{c_B}{n_B}
$$

We create a binary vector of length $n_A + n_B$ with $c_A + c_B$ ones (total conversions), shuffle it, and split:

```python
def perm_test_proportion(n_a, conv_a, n_b, conv_b, n_perm=5000):
    """Permutation test for two proportions (A/B test)."""
    obs_diff = conv_a / n_a - conv_b / n_b
    pooled = np.zeros(n_a + n_b)
    pooled[:conv_a + conv_b] = 1
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        np.random.shuffle(pooled)
        perm_diffs[i] = pooled[:n_a].mean() - pooled[n_a:].mean()
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    return obs_diff, p_value, perm_diffs
```

## Worked Examples

### Two-sample: page load times

Two web pages are compared. Page A has $n = 36$ observations drawn from $N(120, 30^2)$; page B has $n = 40$ from $N(135, 30^2)$.

```python
page_a = np.random.normal(120, 30, size=36)
page_b = np.random.normal(135, 30, size=40)
diff, p, perms = perm_test_two_sample(page_a, page_b)
```

### Multi-group: four treatment arms

Four groups of 30 observations each, drawn from normal distributions with means 160, 170, 155, and 180.

```python
groups = [np.random.normal(mu, 25, 30) for mu in [160, 170, 155, 180]]
var_obs, p_multi, perm_vars = perm_test_multi_group(groups)
```

### Proportions: conversion rates

Control group: 200 conversions out of 23,739 users. Treatment: 182 conversions out of 22,588.

```python
diff_ab, p_ab, perms_ab = perm_test_proportion(23739, 200, 22588, 182)
```

## Interpretation

- The **two-sample test** detects the 15-unit shift between the page groups. Its $p$-value should be small, consistent with a two-sample $t$-test.
- The **multi-group test** identifies that the four group means are not all equal. The variance-of-means statistic is analogous to the $F$-statistic in ANOVA.
- The **proportion test** typically yields a large $p$-value for this data, indicating no significant difference in conversion rates -- consistent with a chi-squared test of independence.

Permutation tests are exact in the sense that they control the Type I error rate at exactly $\alpha$ for finite samples, provided the null hypothesis of exchangeability holds. For continuous data, ties are negligible and the permutation distribution is discrete but rich enough to approximate the continuous null.

## Exercises

**Exercise 1.** Generate two samples of size 50 from the same $N(0, 1)$ distribution. Perform the two-sample permutation test and record the $p$-value. Repeat this 1000 times and plot the histogram of $p$-values. What distribution should they follow under the null?

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(0)
    pvals = []
    for _ in range(1000):
        x = np.random.normal(0, 1, 50)
        y = np.random.normal(0, 1, 50)
        _, p, _ = perm_test_two_sample(x, y, n_perm=1000)
        pvals.append(p)

    plt.hist(pvals, bins=20, edgecolor='k')
    plt.xlabel('p-value')
    plt.ylabel('Frequency')
    plt.title('Distribution of p-values under H0')
    plt.show()
    ```

    Under the null hypothesis, $p$-values follow a $\text{Uniform}(0, 1)$ distribution. The histogram should be approximately flat across the interval $[0, 1]$. This is because under $H_0$ the observed test statistic is just another draw from the permutation distribution, so the probability of it exceeding any fraction $\alpha$ of the permuted values is exactly $\alpha$. $\square$

---

**Exercise 2.** Prove that the two-sided permutation $p$-value satisfies $P(p \le \alpha) \le \alpha$ under $H_0$ for any $\alpha \in (0, 1)$. (This establishes that the permutation test controls the Type I error rate.)

??? success "Solution to Exercise 2"

    Let $T_0 = T_{\text{obs}}$ and $T_1, T_2, \ldots, T_B$ be the permuted test statistics. Under $H_0$, all permutations are equally likely, so $T_0, T_1, \ldots, T_B$ are exchangeable.

    The $p$-value is:

    $$
    p = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}(|T_b| \ge |T_0|)
    $$

    Consider the extended set $\{|T_0|, |T_1|, \ldots, |T_B|\}$. By exchangeability, $|T_0|$ is equally likely to occupy any rank among these $B + 1$ values. Therefore:

    $$
    P(p \le \alpha) = P\!\left(\frac{\#\{b : |T_b| \ge |T_0|\}}{B} \le \alpha\right) = P\!\left(\text{rank of } |T_0| \ge (1 - \alpha)B\right)
    $$

    Since $|T_0|$ is uniformly distributed over the $B + 1$ ranks, $P(p \le \alpha) \le (\lfloor \alpha B \rfloor + 1)/(B + 1) \le \alpha + 1/(B+1)$, which for standard choices of $B$ is at most $\alpha$. $\square$

---

**Exercise 3.** Explain why the multi-group test uses a one-sided $p$-value (counting only $T^{(\pi)} \ge T_{\text{obs}}$) while the two-sample test uses a two-sided $p$-value.

??? success "Solution to Exercise 3"

    The two-sample test statistic is the *difference* of means, $\bar x - \bar y$, which can be positive or negative. Under the alternative, the difference could go in either direction (group $X$ could have a larger or smaller mean than group $Y$). Therefore, we use a two-sided test that counts permutations where $|T^{(\pi)}| \ge |T_{\text{obs}}|$.

    The multi-group test statistic is the *variance* of group means, which is always non-negative. Under $H_0$ all group means are similar, so $T$ is small. Under the alternative (at least one group differs), $T$ increases. There is no notion of a "negative direction" for the variance. Therefore, only large values of $T$ provide evidence against $H_0$, and we use a one-sided $p$-value counting $T^{(\pi)} \ge T_{\text{obs}}$. $\square$

---

**Exercise 4.** The conversion rate example uses 23,739 and 22,588 users. With such large sample sizes, the permutation distribution should closely approximate a normal distribution by the CLT. Verify this empirically by overlaying a normal density (with the mean and standard deviation of the permutation distribution) on the histogram of permuted differences.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    np.random.seed(1)
    diff_ab, p_ab, perms_ab = perm_test_proportion(23739, 200, 22588, 182, n_perm=5000)

    mu = perms_ab.mean()
    sigma = perms_ab.std()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(perms_ab * 100, bins=40, density=True, alpha=0.7, edgecolor='k')
    x = np.linspace(perms_ab.min() * 100, perms_ab.max() * 100, 200)
    ax.plot(x, stats.norm.pdf(x, mu * 100, sigma * 100), 'r-', lw=2,
            label='Normal approximation')
    ax.set_xlabel('Difference in rate (%)')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    plt.show()
    ```

    The normal density closely matches the histogram, confirming that the permutation distribution is approximately normal for large sample sizes. This is a consequence of the central limit theorem applied to the difference of sample proportions. $\square$

---

**Exercise 5.** Consider a paired experiment with $n$ subjects measured before and after treatment. Design a permutation test that respects the paired structure. (Hint: for each subject, randomly flip the sign of the difference $d_i = x_i^{\text{after}} - x_i^{\text{before}}$.) Implement your test and apply it to the data `before = [82, 78, 91, 85, 73]` and `after = [88, 82, 95, 89, 78]`.

??? success "Solution to Exercise 5"

    Under $H_0$ (no treatment effect), each difference $d_i$ is equally likely to be positive or negative. We randomly flip the signs:

    ```python
    import numpy as np

    def paired_perm_test(before, after, n_perm=10_000):
        d = np.array(after) - np.array(before)
        t_obs = np.mean(d)
        n = len(d)
        count = 0
        for _ in range(n_perm):
            signs = np.random.choice([-1, 1], size=n)
            t_perm = np.mean(signs * d)
            if abs(t_perm) >= abs(t_obs):
                count += 1
        return t_obs, count / n_perm

    before = [82, 78, 91, 85, 73]
    after = [88, 82, 95, 89, 78]
    t_obs, p = paired_perm_test(before, after)
    print(f"Mean difference: {t_obs:.1f}")
    print(f"Permutation p-value: {p:.4f}")
    ```

    The mean difference is $5.0$. With only $n = 5$ pairs, there are $2^5 = 32$ possible sign-flip combinations. The exact permutation test can enumerate all of them. The observed mean difference of 5.0 is relatively large compared to the permutation distribution, and the $p$-value will typically be small (around 0.03--0.06), providing moderate evidence for a treatment effect. $\square$
