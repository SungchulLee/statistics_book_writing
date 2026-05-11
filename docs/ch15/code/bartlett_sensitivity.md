# Bartlett Test Non-Normality Sensitivity

## Overview

Bartlett's test for equality of variances is the uniformly most powerful test under normality, but it is notoriously sensitive to departures from normality. When the data are skewed or heavy-tailed, Bartlett's test has an inflated false-positive rate, rejecting the null hypothesis of equal variances far more often than the nominal significance level $\alpha$ would suggest. This page demonstrates this sensitivity through simulation and compares Bartlett's test with Levene's and Fligner--Killeen tests under lognormal data.

## The Problem

Under $H_0: \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2$ and the normality assumption, the Bartlett statistic $T$ has an approximate $\chi^2(k-1)$ distribution. When the data are non-normal, the actual distribution of $T$ is stochastically larger than $\chi^2(k-1)$, because the excess kurtosis inflates the variability of $S_i^2$. The result is that

$$
P(T > \chi^2_{1-\alpha}(k-1) \mid H_0, \text{non-normal data}) \gg \alpha.
$$

## Simulation Design

To demonstrate this effect:

1. Generate $k = 3$ groups of size $n$ from a **lognormal** distribution with equal parameters (so the null hypothesis of equal variances truly holds).
2. Apply Bartlett's, Levene's, and Fligner--Killeen tests at level $\alpha = 0.05$.
3. Repeat many times and record the proportion of rejections (the empirical false-positive rate).

Under correct calibration, all tests should reject at approximately 5%.

## Code

```python
import numpy as np
from scipy.stats import bartlett, levene, fligner

rng = np.random.default_rng(0)


def simulate_once(n=20, sigmas=(1.0, 1.0, 1.0), skew=True):
    """Generate k groups from lognormal (skewed) or normal."""
    if skew:
        groups = [rng.lognormal(mean=0.0, sigma=s, size=n) for s in sigmas]
    else:
        groups = [rng.normal(loc=0.0, scale=s, size=n) for s in sigmas]
    return groups


def trial(n=20, sigmas=(1.0, 1.0, 1.0), skew=True):
    """Run all three tests on one simulated dataset."""
    g1, g2, g3 = simulate_once(n=n, sigmas=sigmas, skew=skew)
    _, p_bartlett = bartlett(g1, g2, g3)
    _, p_levene = levene(g1, g2, g3, center='mean')
    _, p_fligner = fligner(g1, g2, g3)
    return p_bartlett, p_levene, p_fligner


# Run simulation
n_sims, alpha = 5000, 0.05
ps_b, ps_lv, ps_fl = [], [], []

for _ in range(n_sims):
    p_b, p_l, p_f = trial(n=20, sigmas=(1.0, 1.0, 1.0), skew=True)
    ps_b.append(p_b)
    ps_lv.append(p_l)
    ps_fl.append(p_f)

fp_b = np.mean(np.array(ps_b) < alpha)
fp_lv = np.mean(np.array(ps_lv) < alpha)
fp_fl = np.mean(np.array(ps_fl) < alpha)

print("False-positive rates under skewed (lognormal) data:")
print(f"  Bartlett       : {fp_b:.3f}")
print(f"  Levene (mean)  : {fp_lv:.3f}")
print(f"  Fligner-Killeen: {fp_fl:.3f}")
```

## Interpretation

- **Bartlett's test** typically shows a false-positive rate of 0.20 or higher under lognormal data -- roughly four times the nominal $\alpha = 0.05$. This makes it unreliable as a diagnostic for equal variances when normality is doubtful.
- **Levene's test** (mean-centered) may also be somewhat inflated (e.g., 0.06--0.08) under strong skewness.
- **Fligner--Killeen** maintains a false-positive rate very close to 0.05 regardless of the underlying distribution, thanks to its rank-based construction.
- The key takeaway is that Bartlett's sensitivity to non-normality is not a minor nuisance; it can lead to fundamentally wrong conclusions.

## Exercises

**Exercise 1.** Run the simulation with `skew=False` (normal data) and verify that all three tests have false-positive rates close to 0.05.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy.stats import bartlett, levene, fligner

    rng = np.random.default_rng(0)
    n_sims, alpha = 5000, 0.05
    rej = {"Bartlett": 0, "Levene": 0, "Fligner": 0}

    for _ in range(n_sims):
        g1 = rng.normal(0, 1, 20)
        g2 = rng.normal(0, 1, 20)
        g3 = rng.normal(0, 1, 20)
        _, p = bartlett(g1, g2, g3)
        if p < alpha: rej["Bartlett"] += 1
        _, p = levene(g1, g2, g3, center='mean')
        if p < alpha: rej["Levene"] += 1
        _, p = fligner(g1, g2, g3)
        if p < alpha: rej["Fligner"] += 1

    for name, count in rej.items():
        print(f"{name}: {count/n_sims:.3f}")
    ```

    All three rates should be close to 0.05, confirming that under normality all tests are properly calibrated.

---

**Exercise 2.** Increase the sample size to $n = 100$ while keeping the lognormal distribution. Does the inflated false-positive rate for Bartlett's test improve? Explain why or why not.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy.stats import bartlett

    rng = np.random.default_rng(0)
    n_sims = 5000
    rej = 0
    for _ in range(n_sims):
        g1 = rng.lognormal(0, 1, 100)
        g2 = rng.lognormal(0, 1, 100)
        g3 = rng.lognormal(0, 1, 100)
        _, p = bartlett(g1, g2, g3)
        if p < 0.05:
            rej += 1
    print(f"Bartlett FP rate (n=100): {rej/n_sims:.3f}")
    ```

    The false-positive rate remains elevated or may even increase. Larger samples give more precise variance estimates, but they also give the test more statistical power. The fundamental problem is that Bartlett's $\chi^2$ reference distribution is wrong for non-normal data, and this does not improve with sample size. The bias is asymptotic, not a finite-sample issue.

---

**Exercise 3.** Modify the simulation to use $t(3)$ data instead of lognormal. Is Bartlett's test still inflated? How does the severity compare with the lognormal case?

??? success "Solution to Exercise 3"

    ```python
    import numpy as np
    from scipy.stats import bartlett, t as tdist

    rng = np.random.default_rng(0)
    n_sims = 5000
    rej = 0
    for _ in range(n_sims):
        g1 = tdist(df=3).rvs(20, random_state=rng)
        g2 = tdist(df=3).rvs(20, random_state=rng)
        g3 = tdist(df=3).rvs(20, random_state=rng)
        _, p = bartlett(g1, g2, g3)
        if p < 0.05:
            rej += 1
    print(f"Bartlett FP rate (t(3)): {rej/n_sims:.3f}")
    ```

    The $t(3)$ distribution has excess kurtosis $\kappa = \infty$ (or very large for practical purposes), so Bartlett's test is severely inflated -- potentially even worse than under the lognormal. The lognormal has a skewness problem; the $t(3)$ has a kurtosis problem. Both violate Bartlett's normality assumption, but in different ways.

---

**Exercise 4.** The lognormal distribution with parameter $\sigma$ has variance $(e^{\sigma^2} - 1)e^{2\mu + \sigma^2}$. When all groups use the same $\sigma$ and $\mu$, the population variances are equal. Compute the population variance, skewness, and kurtosis for the lognormal with $\mu = 0, \sigma = 1$.

??? success "Solution to Exercise 4"

    For $X \sim \text{Lognormal}(\mu, \sigma^2)$ with $\mu = 0, \sigma = 1$:

    $$
    E[X] = e^{\mu + \sigma^2/2} = e^{1/2} \approx 1.6487.
    $$

    $$
    \operatorname{Var}(X) = (e^{\sigma^2} - 1)e^{2\mu + \sigma^2} = (e - 1)e^1 \approx 1.7183 \times 2.7183 \approx 4.6708.
    $$

    The skewness is

    $$
    \gamma_1 = (e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1} = (e + 2)\sqrt{e - 1} \approx 4.7183 \times 1.3108 \approx 6.185.
    $$

    The excess kurtosis is

    $$
    \gamma_2 = e^{4\sigma^2} + 2e^{3\sigma^2} + 3e^{2\sigma^2} - 6 = e^4 + 2e^3 + 3e^2 - 6 \approx 54.598 + 40.171 + 22.167 - 6 = 110.936.
    $$

    The enormous skewness and kurtosis explain why Bartlett's test fails so dramatically: the $\chi^2$ approximation for $T$ is wildly inaccurate when the underlying data have this level of non-normality. $\square$

---

**Exercise 5.** Propose a corrected version of Bartlett's test that adjusts the critical value based on the observed kurtosis. Discuss the feasibility and limitations of this approach.

??? success "Solution to Exercise 5"

    One approach is Box's (1953) correction, which replaces the $\chi^2(k-1)$ reference with $\chi^2(k-1)/C$, where

    $$
    C = 1 + \frac{\hat{\kappa}}{2(k+1)}\left(\sum_{i=1}^k \frac{1}{n_i - 1} - \frac{1}{N-k}\right) + \text{higher-order terms},
    $$

    and $\hat{\kappa}$ is an estimate of the common excess kurtosis pooled across groups.

    ```python
    import numpy as np
    from scipy.stats import bartlett, kurtosis, chi2

    def bartlett_corrected(*groups, alpha=0.05):
        stat, _ = bartlett(*groups)
        k = len(groups)
        all_data = np.concatenate(groups)
        kappa_hat = kurtosis(all_data, fisher=True)
        ns = [len(g) for g in groups]
        N = sum(ns)
        correction = max(1 + kappa_hat / (2*(k+1)) *
                         (sum(1/(n-1) for n in ns) - 1/(N-k)), 0.5)
        adjusted_stat = stat / correction
        p = chi2(df=k-1).sf(adjusted_stat)
        return adjusted_stat, p
    ```

    **Limitations**: (1) Estimating kurtosis itself requires large samples to be reliable. (2) The correction assumes a single kurtosis value across groups; if groups have different distributions, the correction is approximate. (3) For extremely non-normal data, no simple correction fully fixes the problem -- nonparametric tests like Fligner--Killeen are more reliable.
