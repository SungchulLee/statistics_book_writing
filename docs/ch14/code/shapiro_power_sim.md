# Shapiro-Wilk Power Simulation

## Overview

The power of a normality test is the probability of correctly rejecting the null hypothesis when the data truly come from a non-normal distribution. This page uses Monte Carlo simulation to estimate the empirical power of the Shapiro-Wilk test across a range of sample sizes, using a lognormal alternative. The resulting power curve illustrates how larger samples dramatically improve detection of non-normality.

## Power of a Test

For a normality test at significance level $\alpha$, the power against a specific alternative distribution $F_1$ is

$$
\text{Power}(n, \alpha, F_1) = P\bigl(\text{reject } H_0 \mid X_1, \ldots, X_n \sim F_1\bigr).
$$

The power depends on three factors:

1. **Sample size $n$:** larger samples provide more information.
2. **Significance level $\alpha$:** a larger $\alpha$ increases power but also increases Type I error.
3. **Degree of non-normality:** distributions further from normal are easier to detect.

## Simulation Procedure

For each sample size $n$:

1. Repeat $M$ times:
    - Draw $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Lognormal}(0, \sigma)$.
    - Run the Shapiro-Wilk test at level $\alpha$.
    - Record whether $H_0$ was rejected.
2. Estimate power as the fraction of rejections:

$$
\widehat{\text{Power}}(n) = \frac{\text{number of rejections}}{M}.
$$

### Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def power_for_n(n, sims=500, sigma_ln=0.6, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(sims):
        x = rng.lognormal(mean=0.0, sigma=sigma_ln, size=n)
        W, p = stats.shapiro(x)
        if p < alpha:
            rejections += 1
    return rejections / sims

ns = [20, 30, 50, 80, 120, 200, 300]
powers = [power_for_n(n, sims=400, sigma_ln=0.6, alpha=0.05,
                      seed=42 + n) for n in ns]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(ns, powers, marker="o")
ax.set_ylim(0, 1)
ax.set_xlabel("Sample size (n)")
ax.set_ylabel("Empirical power (alpha = 0.05)")
ax.set_title("Shapiro-Wilk power vs n (lognormal alt, sigma = 0.6)")
plt.tight_layout()
plt.show()

for n, pw in zip(ns, powers):
    print(f"n = {n:>3}: power = {pw:.3f}")
```

## Reading the Power Curve

The power curve rises from near $\alpha$ (for very small $n$, the test barely exceeds the nominal rejection rate) toward 1.0 as $n$ increases. Key takeaways:

- At $n = 20$, power may be only 0.3--0.5: the test misses more than half of lognormal samples.
- By $n = 100$--$200$, power approaches 1.0: the test reliably detects the lognormal alternative.
- The steepness of the curve depends on $\sigma$ (the lognormal shape parameter). Larger $\sigma$ means greater departure from normality and a steeper power curve.

## Interpretation

The simulation demonstrates that the Shapiro-Wilk test's effectiveness depends critically on sample size. For small samples, failing to reject normality should not be interpreted as strong evidence for normality -- the test may simply lack power. Conversely, for very large samples, the test will reject even for distributions that are "close enough" to normal for practical purposes.

## Exercises

**Exercise 1.** Run the power simulation for $\sigma = 0.3$ (a lognormal closer to normal) and compare the power curve with the $\sigma = 0.6$ curve. At what sample size does power reach 0.8 for each?

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    def power_curve(sigma, ns, sims=500, alpha=0.05):
        powers = []
        for n in ns:
            rng = np.random.default_rng(42 + n)
            rej = sum(1 for _ in range(sims)
                      if stats.shapiro(rng.lognormal(0, sigma, n))[1] < alpha)
            powers.append(rej / sims)
        return powers

    ns = [20, 30, 50, 80, 120, 200, 300, 500]
    p03 = power_curve(0.3, ns)
    p06 = power_curve(0.6, ns)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, p03, marker="o", label="sigma = 0.3")
    ax.plot(ns, p06, marker="s", label="sigma = 0.6")
    ax.axhline(0.8, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Power")
    ax.set_title("Power Curves: Lognormal Alternatives")
    ax.legend()
    plt.tight_layout()
    plt.show()
    ```

    For $\sigma = 0.6$, power reaches 0.8 around $n = 50$--$80$. For $\sigma = 0.3$, the lognormal is closer to normal (less skew), so power reaches 0.8 only around $n = 200$--$300$. This illustrates the fundamental trade-off: subtler departures require larger samples. $\square$

---

**Exercise 2.** Modify the simulation to compare the power of the Shapiro-Wilk test and the D'Agostino $K^2$ test against the lognormal alternative. Plot both power curves on the same axes.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    ns = [20, 30, 50, 80, 120, 200, 300]
    sims, alpha, sigma = 500, 0.05, 0.6
    pw_sw, pw_k2 = [], []

    for n in ns:
        rng = np.random.default_rng(42 + n)
        r_sw = r_k2 = 0
        for _ in range(sims):
            x = rng.lognormal(0, sigma, n)
            if stats.shapiro(x)[1] < alpha:
                r_sw += 1
            if stats.normaltest(x)[1] < alpha:
                r_k2 += 1
        pw_sw.append(r_sw / sims)
        pw_k2.append(r_k2 / sims)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, pw_sw, marker="o", label="Shapiro-Wilk")
    ax.plot(ns, pw_k2, marker="s", label="D'Agostino K^2")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Power")
    ax.legend()
    plt.tight_layout()
    plt.show()
    ```

    Both tests should have high power for $n \geq 100$. For smaller $n$ (20--50), the Shapiro-Wilk test typically has higher power against the lognormal alternative. $\square$

---

**Exercise 3.** Explain mathematically why the power of any consistent test converges to 1 as $n \to \infty$ for any fixed alternative $F_1 \neq \mathcal{N}$.

??? success "Solution to Exercise 3"

    A test is consistent if its test statistic $T_n$ diverges (in probability or almost surely) from its null distribution as $n \to \infty$ under $F_1$. Formally, for a consistent test, the Shapiro-Wilk statistic satisfies $W_n \xrightarrow{p} c < 1$ when $F_1 \neq \mathcal{N}$. Since the rejection region is $\{W < w_\alpha\}$ where $w_\alpha$ is the critical value, and $w_\alpha \to 1$ more slowly than $W_n \to c$, eventually $W_n < w_\alpha$ with probability approaching 1:

    $$
    \text{Power}(n) = P(W_n < w_\alpha \mid F_1) \to P(c < 1) = 1.
    $$

    The same argument applies to any consistent test: the statistic converges to a value in the rejection region under the alternative, so the rejection probability tends to 1. $\square$

---

**Exercise 4.** Estimate the power of the Shapiro-Wilk test at $n = 100$ against a $t_\nu$ alternative for $\nu \in \{3, 5, 10, 30, 100\}$. Plot power versus $\nu$.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    n, sims, alpha = 100, 1000, 0.05
    dfs = [3, 5, 10, 30, 100]
    powers = []

    for df in dfs:
        rej = sum(1 for _ in range(sims)
                  if stats.shapiro(rng.standard_t(df, n))[1] < alpha)
        powers.append(rej / sims)
        print(f"df = {df:>3}: power = {rej/sims:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dfs, powers, marker="o")
    ax.set_xlabel("Degrees of freedom (nu)")
    ax.set_ylabel("Power")
    ax.set_title("SW Power vs t(nu) at n = 100")
    plt.tight_layout()
    plt.show()
    ```

    Power decreases as $\nu$ increases because $t_\nu \to \mathcal{N}(0,1)$. For $\nu = 3$ (infinite kurtosis), power is very high. For $\nu = 100$, the $t$ distribution is nearly indistinguishable from normal, and power is close to $\alpha = 0.05$. $\square$

---

**Exercise 5.** Derive a formula for the Monte Carlo standard error of the estimated power and compute the 95% confidence interval for the power estimate when $\widehat{\text{Power}} = 0.72$ with $M = 400$ simulations.

??? success "Solution to Exercise 5"

    Each simulation is a Bernoulli trial with success probability $\pi = \text{Power}$. The estimator $\hat{\pi} = \widehat{\text{Power}}$ has variance $\pi(1-\pi)/M$, so the standard error is

    $$
    \text{SE}(\hat{\pi}) = \sqrt{\frac{\hat{\pi}(1 - \hat{\pi})}{M}}.
    $$

    For $\hat{\pi} = 0.72$ and $M = 400$:

    $$
    \text{SE} = \sqrt{\frac{0.72 \times 0.28}{400}} = \sqrt{\frac{0.2016}{400}} = \sqrt{0.000504} \approx 0.0225.
    $$

    The 95% confidence interval is $\hat{\pi} \pm 1.96 \times \text{SE} = 0.72 \pm 0.044 = (0.676, 0.764)$. To halve the standard error, we would need $M = 1600$ simulations. $\square$
