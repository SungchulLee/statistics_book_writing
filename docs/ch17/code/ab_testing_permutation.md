# A/B Testing Permutation

## Overview

Permutation tests provide a natural framework for A/B testing because they directly model the randomization inherent in experimental design. This page walks through three complete A/B testing scenarios: web page session duration (stickiness), conversion rate testing with binary outcomes, and multi-headline click-through rate comparison. Each example includes both a permutation test and a parametric alternative for comparison.

## Web Page Stickiness Test

An experiment compares session durations (in seconds) for two web page designs with $n_A = 10$ and $n_B = 10$ users. The test statistic is the difference of sample means:

$$
T_{\text{obs}} = \bar x_B - \bar x_A
$$

Under $H_0$ (both pages produce the same session time distribution), the "Page A" and "Page B" labels are arbitrary. We shuffle labels $B$ times and compute:

$$
p = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}\!\bigl(|T^{(\pi_b)}| \ge |T_{\text{obs}}|\bigr)
$$

```python
session_data = {
    'Time': [185, 188, 142, 160, 161, 157, 182, 181, 159, 167,
             173, 181, 182, 170, 169, 177, 168, 183, 169, 164],
    'Page': ['Page A'] * 10 + ['Page B'] * 10
}

def perm_test_two_sample_means(data_col, group_col, nA, nB, n_perms=1000):
    """Permutation test for difference of means between two groups."""
    obs_diff = (data_col[group_col == 'Page A'].mean()
                - data_col[group_col == 'Page B'].mean())
    pooled = data_col.values.copy()
    perm_diffs = []
    for _ in range(n_perms):
        np.random.shuffle(pooled)
        diff = pooled[:nA].mean() - pooled[nA:].mean()
        perm_diffs.append(diff)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    return p_value, perm_diffs, obs_diff
```

The result is compared with a Welch $t$-test:

$$
t = \frac{\bar x_A - \bar x_B}{\sqrt{s_A^2/n_A + s_B^2/n_B}}
$$

With small sample sizes ($n = 10$ per group) and potentially non-normal data, the permutation test is more trustworthy than the $t$-test.

## Conversion Rate A/B Test

For binary outcomes, we test whether the treatment changes the conversion rate. Let:

- Control: $c_0 = 200$ conversions out of $n_0 = 23{,}739$ users
- Treatment: $c_1 = 182$ conversions out of $n_1 = 22{,}588$ users

The observed difference in rates is:

$$
T_{\text{obs}} = \frac{c_1}{n_1} - \frac{c_0}{n_0}
$$

The permutation approach creates a binary vector of length $n_0 + n_1$ with $c_0 + c_1$ ones and shuffles it:

```python
def perm_test_proportion(n_control, conv_control, n_treatment, conv_treatment,
                         n_perms=1000):
    """Permutation test for difference in conversion rates."""
    binary = np.zeros(n_control + n_treatment, dtype=int)
    binary[:conv_control + conv_treatment] = 1
    obs_diff = conv_treatment / n_treatment - conv_control / n_control

    perm_diffs = []
    for _ in range(n_perms):
        np.random.shuffle(binary)
        perm_rate_control = binary[:n_control].mean()
        perm_rate_treatment = binary[n_control:].mean()
        perm_diffs.append(perm_rate_treatment - perm_rate_control)

    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))
    return p_value, perm_diffs, obs_diff
```

The comparison uses a chi-squared test of independence on the $2 \times 2$ contingency table:

$$
\chi^2 = \sum_{i,j}\frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

## Multi-Headline Click-Through Test

Three headlines are tested with 1,000 impressions each:

| Headline | Clicks | No Clicks |
|---|---|---|
| A | 14 | 986 |
| B | 8 | 992 |
| C | 12 | 988 |

With more than two groups, a chi-squared test of independence is the standard parametric approach. A permutation analogue would use the variance of group proportions as the test statistic (similar to the multi-group permutation test).

## Power Analysis

The power of an A/B test depends on the effect size $d$, sample size $n$, and significance level $\alpha$. For a two-sample test with equal group sizes:

$$
\text{Power} \approx 1 - \mathcal{N}\!\left(z_{1-\alpha/2} - d\sqrt{\frac{n}{2}}\right)
$$

where $d = (\mu_1 - \mu_2)/\sigma$ is Cohen's $d$. Key benchmarks:

- $d = 0.2$: small effect -- requires $n \approx 400$ per group for 80% power
- $d = 0.5$: medium effect -- requires $n \approx 65$ per group
- $d = 0.8$: large effect -- requires $n \approx 25$ per group

## Interpretation

The three examples illustrate different facets of permutation-based A/B testing:

1. **Session duration**: With continuous data and small samples, the permutation test closely mirrors the $t$-test but makes no normality assumption.
2. **Conversion rates**: With binary data and large samples, the permutation $p$-value aligns with the chi-squared test. Both typically fail to reject $H_0$ for this dataset, since the observed rate difference is tiny.
3. **Multi-headline**: When comparing more than two groups, chi-squared is standard, but a permutation test on the variance of group click rates provides a distribution-free alternative.

In practice, the permutation framework is especially attractive for A/B tests because:

- It directly models the randomization used to assign users to groups.
- It requires no distributional assumptions.
- The $p$-value has a transparent interpretation: the fraction of random relabelings producing a result as extreme as observed.

## Exercises

**Exercise 1.** The session-duration example uses $n = 10$ per group. Simulate a scenario where Page B truly has 15 seconds longer mean session time. Run the permutation test 1000 times with $n = 10$ and $n = 50$ per group. Estimate the power at $\alpha = 0.05$ for each sample size.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np

    np.random.seed(42)

    def power_estimate(n_per_group, true_diff=15, sigma=20, n_sims=1000, n_perms=1000):
        reject = 0
        for _ in range(n_sims):
            a = np.random.normal(170, sigma, n_per_group)
            b = np.random.normal(170 + true_diff, sigma, n_per_group)
            obs = a.mean() - b.mean()
            pooled = np.concatenate([a, b])
            perm_diffs = []
            for __ in range(n_perms):
                np.random.shuffle(pooled)
                perm_diffs.append(pooled[:n_per_group].mean() - pooled[n_per_group:].mean())
            p = np.mean(np.abs(perm_diffs) >= np.abs(obs))
            if p < 0.05:
                reject += 1
        return reject / n_sims

    power_10 = power_estimate(10)
    power_50 = power_estimate(50)
    print(f"Power with n=10: {power_10:.3f}")
    print(f"Power with n=50: {power_50:.3f}")
    ```

    With $n = 10$, the power is typically around 0.30--0.45 (under-powered). With $n = 50$, the power increases to approximately 0.90--0.99. This illustrates the well-known principle that larger sample sizes yield greater power to detect a given effect size. $\square$

---

**Exercise 2.** In the conversion-rate example, the observed rate difference is approximately $-0.04\%$. Calculate the minimum detectable effect (MDE) at 80% power and $\alpha = 0.05$ for the given sample sizes ($n_0 = 23{,}739$, $n_1 = 22{,}588$), assuming equal proportions under $H_0$.

??? success "Solution to Exercise 2"

    Under equal proportions, the pooled rate is:

    $$
    \hat p = \frac{200 + 182}{23739 + 22588} = \frac{382}{46327} \approx 0.00824
    $$

    The standard error of the difference under $H_0$ is:

    $$
    \text{SE} = \sqrt{\hat p(1 - \hat p)\left(\frac{1}{n_0} + \frac{1}{n_1}\right)} = \sqrt{0.00824 \times 0.99176 \times \left(\frac{1}{23739} + \frac{1}{22588}\right)}
    $$

    $$
    \text{SE} \approx \sqrt{0.00817 \times 0.0000841} \approx \sqrt{6.87 \times 10^{-7}} \approx 0.000829
    $$

    The MDE at 80% power with $\alpha = 0.05$ (two-sided) is:

    $$
    \text{MDE} = (z_{0.975} + z_{0.80})\times\text{SE} = (1.96 + 0.84)\times 0.000829 \approx 2.80 \times 0.000829 \approx 0.00232
    $$

    So the minimum detectable absolute difference is about 0.23 percentage points. The observed difference of $-0.04\%$ is well below this threshold, which explains the failure to reject $H_0$. $\square$

---

**Exercise 3.** The chi-squared test for the headline data has 2 degrees of freedom. Explain why, and show that the chi-squared test is equivalent to a likelihood ratio test for the multinomial model. Under what conditions does the permutation approach have an advantage?

??? success "Solution to Exercise 3"

    The contingency table has 2 rows (click / no-click) and 3 columns (headlines A, B, C). The degrees of freedom for the chi-squared test of independence are:

    $$
    df = (r - 1)(c - 1) = (2 - 1)(3 - 1) = 2
    $$

    Under $H_0$, the expected count in cell $(i, j)$ is $E_{ij} = R_i C_j / N$, where $R_i$ and $C_j$ are the row and column totals and $N$ is the grand total.

    The likelihood ratio test statistic is:

    $$
    G^2 = 2\sum_{i,j} O_{ij}\ln\frac{O_{ij}}{E_{ij}}
    $$

    By a Taylor expansion, $G^2 \approx \chi^2$ for large samples, so the two tests are asymptotically equivalent.

    The permutation approach has an advantage when:

    - Expected cell counts are small (the chi-squared approximation breaks down below $E_{ij} \approx 5$).
    - The number of categories is large relative to the sample size.
    - One wants an exact $p$-value without relying on asymptotic approximations. $\square$

---

**Exercise 4.** Implement a permutation-based A/B test for the difference in *medians* (rather than means) of session durations. Compare its $p$-value with the mean-based test. When would a median-based test be preferred?

??? success "Solution to Exercise 4"

    ```python
    import numpy as np

    np.random.seed(42)
    times = np.array([185, 188, 142, 160, 161, 157, 182, 181, 159, 167,
                       173, 181, 182, 170, 169, 177, 168, 183, 169, 164])

    obs_diff_mean = times[:10].mean() - times[10:].mean()
    obs_diff_median = np.median(times[:10]) - np.median(times[10:])

    perm_means = []
    perm_medians = []
    for _ in range(10_000):
        np.random.shuffle(times)
        perm_means.append(times[:10].mean() - times[10:].mean())
        perm_medians.append(np.median(times[:10]) - np.median(times[10:]))

    p_mean = np.mean(np.abs(perm_means) >= np.abs(obs_diff_mean))
    p_median = np.mean(np.abs(perm_medians) >= np.abs(obs_diff_median))

    print(f"Mean-based p-value:   {p_mean:.4f}")
    print(f"Median-based p-value: {p_median:.4f}")
    ```

    The median-based test is preferred when:

    - The data contain outliers that could inflate the mean.
    - The distribution is heavily skewed (e.g., session times with a few very long sessions).
    - The research question concerns the "typical" user experience rather than the average.

    In general, the median-based test has lower power than the mean-based test for symmetric distributions (by the asymptotic relative efficiency of the median versus the mean, which is $\pi/2 \approx 63.7\%$ for normal data), but it is more robust to contamination. $\square$

---

**Exercise 5.** Prove that if the two groups have the same distribution (i.e., $H_0$ is true), the expected value of the permutation $p$-value is $E[p] = (B + 1)^{-1}\lceil \alpha(B+1)\rceil$ when ties are absent. More simply, show that $P(p \le \alpha) \le \alpha$ for any valid permutation test.

??? success "Solution to Exercise 5"

    Let $T_0 = |T_{\text{obs}}|$ and $T_1, \ldots, T_B$ be the absolute values of the permuted test statistics. Under $H_0$ and the assumption of no ties, all $(B + 1)$ values $T_0, T_1, \ldots, T_B$ are exchangeable and almost surely distinct.

    The $p$-value is:

    $$
    p = \frac{1}{B}\sum_{b=1}^{B}\mathbf{1}(T_b \ge T_0) = \frac{B + 1 - R}{B}
    $$

    where $R$ is the rank of $T_0$ among $\{T_0, T_1, \ldots, T_B\}$ (from largest to smallest). By exchangeability, $R$ is uniformly distributed on $\{1, 2, \ldots, B+1\}$.

    Then:

    $$
    P(p \le \alpha) = P\!\left(\frac{B + 1 - R}{B} \le \alpha\right) = P\bigl(R \ge B + 1 - \alpha B\bigr) = P\bigl(R \ge (1-\alpha)B + 1\bigr)
    $$

    Since $R$ is uniform on $\{1, \ldots, B+1\}$:

    $$
    P(p \le \alpha) = \frac{\lfloor \alpha B \rfloor + 1}{B + 1} \le \frac{\alpha B + 1}{B + 1} = \alpha + \frac{1 - \alpha}{B + 1} \le \alpha + \frac{1}{B + 1}
    $$

    For practical values of $B$ (e.g., $B = 1000$), the excess over $\alpha$ is negligible. More precisely, the test is conservative: $P(p \le \alpha) \le \alpha + 1/(B+1)$, which converges to $\alpha$ as $B \to \infty$. $\square$
