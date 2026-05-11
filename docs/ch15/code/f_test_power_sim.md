# F-Test Power Simulation

## Overview

The power of a hypothesis test is the probability of correctly rejecting the null hypothesis when it is false. For the F-test of equality of variances, power depends on the true variance ratio, the sample sizes, and the significance level. Since closed-form power expressions for the F-test are complex, Monte Carlo simulation provides a practical and transparent way to estimate power under specific parameter configurations.

## Power of the F-Test

Consider the two-sided F-test with

$$
H_0: \sigma_1^2 = \sigma_2^2 \quad \text{vs.} \quad H_1: \sigma_1^2 \neq \sigma_2^2.
$$

The power is

$$
\beta(\sigma_1, \sigma_2) = P\!\left(\text{reject } H_0 \mid \sigma_1^2 \neq \sigma_2^2\right).
$$

Power increases when:

- The true variance ratio $\sigma_1^2/\sigma_2^2$ is farther from 1.
- The sample sizes $n_1$ and $n_2$ increase.
- The significance level $\alpha$ increases.

## Monte Carlo Estimation

The Monte Carlo approach to estimating power proceeds as follows:

1. Fix the true parameters $\sigma_1, \sigma_2, n_1, n_2$ and the significance level $\alpha$.
2. For each of $B$ replications, generate $X_1 \sim N(\mu_1, \sigma_1^2)$ and $X_2 \sim N(\mu_2, \sigma_2^2)$.
3. Compute the F-statistic and the two-sided $p$-value.
4. Record whether $p < \alpha$.
5. The estimated power is the proportion of rejections: $\hat{\beta} = (\text{number of rejections}) / B$.

The standard error of $\hat{\beta}$ is $\sqrt{\hat{\beta}(1-\hat{\beta})/B}$.

## Code

```python
import numpy as np
from scipy.stats import f

rng = np.random.default_rng(0)


def f_test_two_sided(x1, x2, alpha=0.05):
    """Return True if the two-sided F-test rejects H0."""
    n1, n2 = x1.size, x2.size
    df1, df2 = n1 - 1, n2 - 1
    F_obs = x1.var(ddof=1) / x2.var(ddof=1)
    p_left = f.cdf(F_obs, df1, df2)
    p_right = f.sf(F_obs, df1, df2)
    p_two = 2 * min(p_left, p_right)
    return p_two < alpha


def estimate_power(n1=12, n2=12, sigma1=1.0, sigma2=1.5,
                   n_sims=2000, alpha=0.05):
    """Estimate power of the two-sided F-test via simulation."""
    hits = 0
    for _ in range(n_sims):
        x1 = rng.normal(0, sigma1, size=n1)
        x2 = rng.normal(0, sigma2, size=n2)
        if f_test_two_sided(x1, x2, alpha=alpha):
            hits += 1
    return hits / n_sims


# Example: sigma1/sigma2 = 1/2, n1 = n2 = 10
power = estimate_power(n1=10, n2=10, sigma1=1.0, sigma2=2.0,
                       n_sims=5000, alpha=0.05)
se = np.sqrt(power * (1 - power) / 5000)
print(f"Estimated power: {power:.3f} (SE: {se:.3f})")
```

To explore how power varies with sample size:

```python
for n in [10, 20, 30, 50, 100]:
    pw = estimate_power(n1=n, n2=n, sigma1=1.0, sigma2=1.5, n_sims=3000)
    print(f"n1=n2={n:3d}: power = {pw:.3f}")
```

## Interpretation

- With $\sigma_1 = 1$ and $\sigma_2 = 2$ (a 2:1 ratio of standard deviations, or 4:1 ratio of variances), even small samples ($n = 10$) yield moderate power.
- For a more subtle difference like $\sigma_1/\sigma_2 = 1/1.5$ (a 2.25:1 variance ratio), larger samples are needed.
- The simulation naturally accounts for the finite-sample behavior of the F-test, unlike asymptotic approximations.

## Exercises

**Exercise 1.** Using the `estimate_power` function, create a table showing power for $n_1 = n_2 \in \{10, 25, 50, 100\}$ and $\sigma_2/\sigma_1 \in \{1.25, 1.5, 2.0, 3.0\}$ with $\alpha = 0.05$.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy.stats import f

    rng = np.random.default_rng(0)

    def f_test_two_sided(x1, x2, alpha=0.05):
        F = x1.var(ddof=1) / x2.var(ddof=1)
        df1, df2 = x1.size - 1, x2.size - 1
        p = 2 * min(f.cdf(F, df1, df2), f.sf(F, df1, df2))
        return p < alpha

    print(f"{'n':>5s}", end="")
    for ratio in [1.25, 1.5, 2.0, 3.0]:
        print(f"  ratio={ratio:.2f}", end="")
    print()

    for n in [10, 25, 50, 100]:
        print(f"{n:5d}", end="")
        for ratio in [1.25, 1.5, 2.0, 3.0]:
            hits = 0
            for _ in range(5000):
                x1 = rng.normal(0, 1.0, n)
                x2 = rng.normal(0, ratio, n)
                if f_test_two_sided(x1, x2):
                    hits += 1
            print(f"  {hits/5000:10.3f}", end="")
        print()
    ```

    The table will show that power increases with both $n$ and the variance ratio. For ratio = 1.25 and $n = 10$, power is very low; for ratio = 3.0 and $n = 50$, power is near 1.

---

**Exercise 2.** Modify the simulation to estimate the power of **Levene's test** (median-centered) for the same parameter grid. Compare the results with the F-test.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy.stats import levene

    rng = np.random.default_rng(0)

    for n in [10, 25, 50, 100]:
        for ratio in [1.5, 2.0]:
            hits_f, hits_l = 0, 0
            for _ in range(3000):
                x1 = rng.normal(0, 1.0, n)
                x2 = rng.normal(0, ratio, n)
                # F-test
                from scipy.stats import f as fdist
                F = np.var(x1, ddof=1) / np.var(x2, ddof=1)
                p = 2 * min(fdist(n-1,n-1).cdf(F), fdist(n-1,n-1).sf(F))
                if p < 0.05:
                    hits_f += 1
                # Levene
                _, p_l = levene(x1, x2, center='median')
                if p_l < 0.05:
                    hits_l += 1
            print(f"n={n:3d}, ratio={ratio}: "
                  f"F-test={hits_f/3000:.3f}, Levene={hits_l/3000:.3f}")
    ```

    Under normality, the F-test will have slightly higher power than Levene's test, since the F-test is the optimal test when its assumptions are met. The difference is typically small (a few percentage points).

---

**Exercise 3.** Derive the standard error formula $\text{SE}(\hat{\beta}) = \sqrt{\hat{\beta}(1-\hat{\beta})/B}$ for the Monte Carlo power estimate. How many simulations $B$ are needed to estimate power to within $\pm 0.01$ with 95% confidence?

??? success "Solution to Exercise 3"

    Each replication is a Bernoulli trial with success probability $\beta$ (the true power). The estimator $\hat{\beta} = \sum_{i=1}^B I_i / B$ is a sample proportion, so

    $$
    \operatorname{Var}(\hat{\beta}) = \frac{\beta(1-\beta)}{B}, \quad \text{SE}(\hat{\beta}) = \sqrt{\frac{\hat{\beta}(1-\hat{\beta})}{B}}.
    $$

    For a 95% confidence interval of half-width $\delta = 0.01$, we need $1.96 \cdot \text{SE} \le 0.01$, so

    $$
    B \ge \frac{1.96^2 \cdot \beta(1-\beta)}{0.01^2}.
    $$

    The worst case is $\beta = 0.5$, giving $B \ge 1.96^2 \cdot 0.25 / 0.0001 = 9604$. So $B = 10{,}000$ simulations suffice. $\square$

---

**Exercise 4.** Plot a power curve: fix $n_1 = n_2 = 20$, $\alpha = 0.05$, and let $\sigma_2/\sigma_1$ range from 1.0 to 3.0. Plot estimated power on the $y$-axis against the variance ratio on the $x$-axis.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import f as fdist

    rng = np.random.default_rng(0)
    n, n_sims, alpha = 20, 3000, 0.05
    ratios = np.arange(1.0, 3.05, 0.1)
    powers = []

    for r in ratios:
        hits = 0
        for _ in range(n_sims):
            x1 = rng.normal(0, 1.0, n)
            x2 = rng.normal(0, r, n)
            F = np.var(x1, ddof=1) / np.var(x2, ddof=1)
            p = 2 * min(fdist(n-1, n-1).cdf(F), fdist(n-1, n-1).sf(F))
            if p < alpha:
                hits += 1
        powers.append(hits / n_sims)

    plt.figure(figsize=(8, 4))
    plt.plot(ratios, powers, "o-", markersize=4)
    plt.axhline(alpha, ls="--", color="red", label=f"alpha = {alpha}")
    plt.xlabel("sigma2 / sigma1")
    plt.ylabel("Estimated power")
    plt.title("F-test power curve (n1 = n2 = 20)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    ```

    At ratio = 1.0 the "power" equals $\alpha = 0.05$ (this is the size of the test). Power increases rapidly with the ratio, approaching 1 for ratios above about 2.5.

---

**Exercise 5.** Explain why the F-test may have poor power for detecting variance differences when the sample sizes are very unequal (e.g., $n_1 = 5$, $n_2 = 100$). Verify with a simulation.

??? success "Solution to Exercise 5"

    The F-test statistic $F = S_1^2/S_2^2$ has degrees of freedom $d_1 = n_1 - 1$ and $d_2 = n_2 - 1$. When $n_1$ is very small, $S_1^2$ is estimated from very few observations and has high variability. The $F(d_1, d_2)$ distribution with small $d_1$ is very spread out, so the critical values are far apart and the rejection region is narrow.

    ```python
    import numpy as np
    from scipy.stats import f as fdist

    rng = np.random.default_rng(0)
    n_sims = 5000

    for n1, n2 in [(5, 100), (50, 50)]:
        hits = 0
        for _ in range(n_sims):
            x1 = rng.normal(0, 1.0, n1)
            x2 = rng.normal(0, 2.0, n2)
            F = np.var(x1, ddof=1) / np.var(x2, ddof=1)
            p = 2 * min(fdist(n1-1, n2-1).cdf(F), fdist(n1-1, n2-1).sf(F))
            if p < 0.05:
                hits += 1
        print(f"n1={n1:3d}, n2={n2:3d}: power = {hits/n_sims:.3f}")
    ```

    With $n_1 = 5, n_2 = 100$, the power will be much lower than with $n_1 = n_2 = 50$, even though the total sample size is larger in the first case. The bottleneck is the small group: the test's ability to detect variance differences is limited by the least precisely estimated variance.
