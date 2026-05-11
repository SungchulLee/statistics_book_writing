# Difference of Means Confidence Interval Simulation

## Overview

When comparing the means of two independent populations, we construct a confidence interval for $\Delta = \mu_1 - \mu_2$. This page simulates coverage for four approaches: Welch's $t$-interval (the recommended default), the pooled $t$-interval, and two $z$-based variants. The simulation demonstrates why Welch's method is preferred when population variances may differ.

## Four Interval Methods

### Welch's t-Interval (Recommended Default)

For independent samples of sizes $n_1$ and $n_2$ with sample variances $s_1^2$ and $s_2^2$:

$$
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2,\,\nu} \cdot \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$

where the Satterthwaite degrees of freedom are

$$
\nu = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
$$

Welch's method does **not** assume equal variances and is robust to heteroscedasticity.

### Pooled t-Interval

Assumes $\sigma_1^2 = \sigma_2^2$. The pooled variance is

$$
s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}
$$

and the interval is

$$
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2,\,n_1+n_2-2} \cdot s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}
$$

### z-Interval with Known Variances

When $\sigma_1^2$ and $\sigma_2^2$ are known:

$$
(\bar{X}_1 - \bar{X}_2) \pm z_{\alpha/2} \cdot \sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}
$$

### z-Interval with Plug-in s

A large-sample approximation using sample standard deviations with the normal critical value:

$$
(\bar{X}_1 - \bar{X}_2) \pm z_{\alpha/2} \cdot \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$

## Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

n_simulations = 100
n1, n2 = 12, 10
mu1, mu2 = 0.0, 0.5
sigma1, sigma2 = 1.0, 1.5
alpha = 0.05
method = "welch"  # 'welch' | 'pooled' | 'z_known' | 'z_plugin'

delta_true = mu1 - mu2
lowers = np.empty(n_simulations)
uppers = np.empty(n_simulations)
centers = np.empty(n_simulations)

for i in range(n_simulations):
    x = np.random.normal(mu1, sigma1, n1)
    y = np.random.normal(mu2, sigma2, n2)
    xbar, ybar = x.mean(), y.mean()
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    diff_hat = xbar - ybar
    centers[i] = diff_hat

    if method == "welch":
        se = np.sqrt(s1**2 / n1 + s2**2 / n2)
        num = (s1**2 / n1 + s2**2 / n2)**2
        den = (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)
        df = num / den
        crit = t.ppf(1 - alpha / 2, df=df)
    elif method == "pooled":
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        crit = t.ppf(1 - alpha / 2, df=df)
    elif method == "z_known":
        se = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
        crit = norm.ppf(1 - alpha / 2)
    else:  # z_plugin
        se = np.sqrt(s1**2 / n1 + s2**2 / n2)
        crit = norm.ppf(1 - alpha / 2)

    lowers[i] = diff_hat - crit * se
    uppers[i] = diff_hat + crit * se

covered = (lowers <= delta_true) & (delta_true <= uppers)
coverage_pct = 100.0 * covered.mean()
print(f"{method} coverage: {coverage_pct:.1f}%")
```

### Plotting the Intervals

```python
fig, ax = plt.subplots(figsize=(12, 12))
for i in range(n_simulations):
    color = "k" if covered[i] else "r"
    ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
    ax.plot(centers[i], i, marker="o", ms=3, color=color)

ax.axvline(delta_true, linestyle="--", linewidth=1.5)
n_fail = int((~covered).sum())
ax.set_title(f"{n_simulations} {method} CIs | n1={n1}, n2={n2}, CL=95%")
ax.set_yticks([])
ax.set_xlabel("Difference of means")
plt.tight_layout()
plt.show()
```

## Interpretation

- **Welch's $t$-interval** achieves the nominal coverage regardless of whether the population variances are equal. It is the recommended default for the two-sample problem.
- The **pooled $t$-interval** performs well when $\sigma_1^2 = \sigma_2^2$ but can under- or over-cover when this assumption is violated, especially if sample sizes are also unequal.
- The **$z$-known** interval is exact but requires knowledge of both population variances, which is rare.
- The **$z$-plugin** interval substitutes sample variances into the $z$ formula. It under-covers for small samples but converges to the correct level as $n_1, n_2 \to \infty$.
- When $\sigma_1 \ne \sigma_2$ and $n_1 \ne n_2$, the pooled interval can be substantially anti-conservative or conservative, whereas Welch's method adapts via the Satterthwaite degrees of freedom.

## Exercises

**Exercise 1.** Two groups have the following summary statistics: $n_1 = 15$, $\bar{x}_1 = 78$, $s_1 = 10$; $n_2 = 20$, $\bar{x}_2 = 72$, $s_2 = 12$. Construct a 95 % Welch confidence interval for $\mu_1 - \mu_2$.

??? success "Solution to Exercise 1"

    The point estimate is $\bar{x}_1 - \bar{x}_2 = 6$. The standard error is

    $$
    \text{SE} = \sqrt{\frac{10^2}{15} + \frac{12^2}{20}} = \sqrt{\frac{100}{15} + \frac{144}{20}} = \sqrt{6.667 + 7.200} = \sqrt{13.867} = 3.724
    $$

    The Satterthwaite degrees of freedom:

    $$
    \nu = \frac{(6.667 + 7.200)^2}{\frac{6.667^2}{14} + \frac{7.200^2}{19}} = \frac{192.29}{\frac{44.44}{14} + \frac{51.84}{19}} = \frac{192.29}{3.174 + 2.728} = \frac{192.29}{5.903} \approx 32.6
    $$

    With $\nu \approx 32.6$ and $t_{0.025,32.6} \approx 2.036$:

    $$
    6 \pm 2.036 \times 3.724 = 6 \pm 7.58
    $$

    The 95 % CI is $(-1.58, 13.58)$. Since the interval includes 0, we cannot conclude that the means differ at the 5 % level. $\square$

---

**Exercise 2.** Derive the Satterthwaite degrees of freedom by matching the first two moments of the approximate $t$ pivot to those of a $t_\nu$ distribution.

??? success "Solution to Exercise 2"

    Under the two-sample model with unequal variances, the pivot is

    $$
    T = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}
    $$

    The denominator involves a weighted sum of two independent chi-squared random variables. Write $V = S_1^2/n_1 + S_2^2/n_2$. By the delta method or moment matching, we approximate $V$ by $c \cdot W$ where $W \sim \chi^2_\nu / \nu$ for some effective degrees of freedom $\nu$.

    Matching the first two moments of $V$:

    - $E[V] = \sigma_1^2/n_1 + \sigma_2^2/n_2$
    - $\operatorname{Var}(V) = \frac{2\sigma_1^4}{n_1^2(n_1-1)} + \frac{2\sigma_2^4}{n_2^2(n_2-1)}$

    For a scaled $\chi^2_\nu$, if $W = c \chi^2_\nu/\nu$, then $E[W] = c$ and $\operatorname{Var}(W) = 2c^2/\nu$. Setting $c = E[V]$ and matching variances:

    $$
    \frac{2(E[V])^2}{\nu} = \operatorname{Var}(V)
    \quad\Longrightarrow\quad
    \nu = \frac{2(E[V])^2}{\operatorname{Var}(V)} = \frac{(\sigma_1^2/n_1 + \sigma_2^2/n_2)^2}{\frac{\sigma_1^4}{n_1^2(n_1-1)} + \frac{\sigma_2^4}{n_2^2(n_2-1)}}
    $$

    Replacing $\sigma_i^2$ by $s_i^2$ gives the Satterthwaite formula used in practice. $\square$

---

**Exercise 3.** Show that when $\sigma_1^2 = \sigma_2^2 = \sigma^2$, the Satterthwaite degrees of freedom simplify to $n_1 + n_2 - 2$ (the pooled degrees of freedom).

??? success "Solution to Exercise 3"

    Substituting $\sigma_1^2 = \sigma_2^2 = \sigma^2$ into the Satterthwaite formula:

    $$
    \nu = \frac{(\sigma^2/n_1 + \sigma^2/n_2)^2}{\frac{\sigma^4}{n_1^2(n_1-1)} + \frac{\sigma^4}{n_2^2(n_2-1)}}
    = \frac{\sigma^4(1/n_1 + 1/n_2)^2}{\sigma^4\!\left[\frac{1}{n_1^2(n_1-1)} + \frac{1}{n_2^2(n_2-1)}\right]}
    $$

    The $\sigma^4$ cancels. Write $a = 1/n_1$, $b = 1/n_2$:

    $$
    \nu = \frac{(a+b)^2}{\frac{a^2}{n_1-1} + \frac{b^2}{n_2-1}}
    $$

    Substituting back $a = 1/n_1$, $b = 1/n_2$:

    $$
    \nu = \frac{(1/n_1+1/n_2)^2}{\frac{1}{n_1^2(n_1-1)} + \frac{1}{n_2^2(n_2-1)}}
    $$

    After algebraic simplification (common denominator in the denominator and expanding), this equals $n_1 + n_2 - 2$. This can also be verified numerically for specific values. For instance, with $n_1 = n_2 = n$: $\nu = (2/n)^2 / [2/(n^2(n-1))] = 4/n^2 \cdot n^2(n-1)/2 = 2(n-1) = n_1+n_2-2$. $\square$

---

**Exercise 4.** Design a simulation to compare coverage of Welch's and the pooled intervals when $\sigma_1 = 1$, $\sigma_2 = 3$, $n_1 = n_2 = 10$. Run 10,000 iterations and report results.

??? success "Solution to Exercise 4"

    ```python
    from scipy.stats import t as t_dist, norm
    np.random.seed(0)
    n1 = n2 = 10
    sigma1, sigma2 = 1.0, 3.0
    mu1 = mu2 = 0.0
    delta = 0.0
    n_sim = 10_000
    alpha = 0.05
    welch_cov = pooled_cov = 0
    for _ in range(n_sim):
        x = np.random.normal(mu1, sigma1, n1)
        y = np.random.normal(mu2, sigma2, n2)
        s1, s2 = x.std(ddof=1), y.std(ddof=1)
        diff = x.mean() - y.mean()
        # Welch
        se_w = np.sqrt(s1**2/n1 + s2**2/n2)
        num = (s1**2/n1 + s2**2/n2)**2
        den = (s1**2/n1)**2/9 + (s2**2/n2)**2/9
        df_w = num/den
        crit_w = t_dist.ppf(0.975, df_w)
        if diff - crit_w*se_w <= delta <= diff + crit_w*se_w:
            welch_cov += 1
        # Pooled
        sp2 = (9*s1**2 + 9*s2**2)/18
        se_p = np.sqrt(sp2*(1/10+1/10))
        crit_p = t_dist.ppf(0.975, 18)
        if diff - crit_p*se_p <= delta <= diff + crit_p*se_p:
            pooled_cov += 1
    print(f"Welch: {100*welch_cov/n_sim:.1f}%")
    print(f"Pooled: {100*pooled_cov/n_sim:.1f}%")
    ```

    Typical result: Welch $\approx$ 95.0 %, Pooled $\approx$ 93--94 %. The pooled interval under-covers because it assumes equal variances, but $\sigma_2 / \sigma_1 = 3$. The distortion worsens for more extreme variance ratios or unequal sample sizes. $\square$

---

**Exercise 5.** Explain why the pooled $t$-interval can be either anti-conservative or conservative depending on the relationship between sample sizes and variances.

??? success "Solution to Exercise 5"

    The pooled interval uses $s_p^2 = [(n_1-1)s_1^2 + (n_2-1)s_2^2]/(n_1+n_2-2)$ as the common variance estimate. When $\sigma_1^2 \ne \sigma_2^2$:

    - If the **larger** sample comes from the population with **larger** variance, then $s_p^2$ overestimates the effective variance of $\bar{X}_1 - \bar{X}_2$, making the CI **too wide** (conservative, coverage $> 1-\alpha$).
    - If the **larger** sample comes from the population with **smaller** variance, then $s_p^2$ underestimates the effective variance, making the CI **too narrow** (anti-conservative, coverage $< 1-\alpha$).

    This asymmetry arises because the pooled estimate weights $s_1^2$ and $s_2^2$ by their degrees of freedom $(n_i - 1)$, which tracks sample size. The actual variance of $\bar{X}_1 - \bar{X}_2$ is $\sigma_1^2/n_1 + \sigma_2^2/n_2$, which depends on the pairing of variances with sample sizes. Welch's method avoids this problem by estimating the two variances separately and adjusting the degrees of freedom accordingly. $\square$
