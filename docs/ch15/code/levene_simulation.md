# Levene Test Normal vs Skewed Simulation

## Overview

Levene's test with median centering (Brown--Forsythe variant) is often recommended for testing equality of variances because of its robustness to non-normality. This page investigates the Type I error rate of the median-centered Levene test under both normal and skewed (lognormal) data via Monte Carlo simulation. The goal is to verify that the test maintains its nominal size of $\alpha = 0.05$ across different distributional settings.

## Simulation Design

The simulation proceeds as follows:

1. Generate $k = 3$ groups of size $n$ from either a normal or lognormal distribution, all with the same parameters (so $H_0$ holds).
2. Apply the median-centered Levene test at level $\alpha = 0.05$.
3. Record whether $H_0$ is rejected.
4. Repeat $B$ times and compute the empirical Type I error rate.

If the test is well calibrated, the rejection rate should be close to 0.05 regardless of the distribution.

## Code

```python
import numpy as np
from scipy.stats import levene

rng = np.random.default_rng(0)


def simulate_once(n=20, dist="normal"):
    """Generate 3 groups from the specified distribution."""
    if dist == "normal":
        g1 = rng.normal(0, 1.0, size=n)
        g2 = rng.normal(0, 1.0, size=n)
        g3 = rng.normal(0, 1.0, size=n)
    else:
        g1 = rng.lognormal(0, 1.0, size=n)
        g2 = rng.lognormal(0, 1.0, size=n)
        g3 = rng.lognormal(0, 1.0, size=n)
    _, p = levene(g1, g2, g3, center='median')
    return p


alpha = 0.05
n_sims = 5000

for dist in ["normal", "lognormal"]:
    pvals = [simulate_once(20, dist) for _ in range(n_sims)]
    type1 = np.mean(np.array(pvals) < alpha)
    print(f"Type I error (median-centered) under {dist}: {type1:.3f}")
```

## Interpretation

- Under **normal** data, the rejection rate is close to 0.05, confirming that the test is correctly sized.
- Under **lognormal** data, the rejection rate remains close to 0.05 (perhaps slightly above), demonstrating the robustness of the median-centered Levene test.
- This contrasts sharply with Bartlett's test, which would show a rejection rate of 0.20 or higher under the same lognormal scenario.
- The median-centered variant achieves robustness because the median is not influenced by the extreme values in the right tail of the lognormal distribution.

## Exercises

**Exercise 1.** Extend the simulation to also compute the Type I error for the **mean-centered** Levene test. Compare the two centering options under lognormal data.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy.stats import levene

    rng = np.random.default_rng(0)
    n_sims, n, alpha = 5000, 20, 0.05

    for center in ['mean', 'median']:
        rej = 0
        for _ in range(n_sims):
            g1 = rng.lognormal(0, 1, n)
            g2 = rng.lognormal(0, 1, n)
            g3 = rng.lognormal(0, 1, n)
            _, p = levene(g1, g2, g3, center=center)
            if p < alpha:
                rej += 1
        print(f"Levene ({center:6s}): Type I error = {rej/n_sims:.3f}")
    ```

    The mean-centered version will show a slightly elevated rejection rate (perhaps 0.06--0.08) under lognormal data, while the median-centered version stays closer to 0.05. The difference arises because the mean is pulled toward the heavy right tail.

---

**Exercise 2.** Vary the group size $n \in \{10, 20, 50, 100\}$ and plot the Type I error rate of the median-centered Levene test under lognormal data. Does the test improve with larger samples?

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy.stats import levene
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    n_sims, alpha = 5000, 0.05
    sizes = [10, 20, 50, 100]
    rates = []

    for n in sizes:
        rej = 0
        for _ in range(n_sims):
            g1 = rng.lognormal(0, 1, n)
            g2 = rng.lognormal(0, 1, n)
            g3 = rng.lognormal(0, 1, n)
            _, p = levene(g1, g2, g3, center='median')
            if p < alpha:
                rej += 1
        rates.append(rej / n_sims)

    plt.figure(figsize=(6, 3))
    plt.plot(sizes, rates, "o-")
    plt.axhline(0.05, ls="--", color="red")
    plt.xlabel("Group size n")
    plt.ylabel("Type I error rate")
    plt.title("Brown-Forsythe under lognormal data")
    plt.tight_layout()
    plt.show()
    ```

    The Type I error rate should stay close to 0.05 for all sample sizes, confirming the robustness of the median-centered test across a range of $n$.

---

**Exercise 3.** Repeat the simulation using data from a $t(3)$ distribution (symmetric but heavy-tailed). How does the median-centered Levene test perform compared to the mean-centered version?

??? success "Solution to Exercise 3"

    ```python
    import numpy as np
    from scipy.stats import levene, t as tdist

    rng = np.random.default_rng(0)
    n_sims, n, alpha = 5000, 20, 0.05

    for center in ['mean', 'median']:
        rej = 0
        for _ in range(n_sims):
            g1 = tdist(df=3).rvs(n, random_state=rng)
            g2 = tdist(df=3).rvs(n, random_state=rng)
            g3 = tdist(df=3).rvs(n, random_state=rng)
            _, p = levene(g1, g2, g3, center=center)
            if p < alpha:
                rej += 1
        print(f"t(3), Levene ({center}): {rej/n_sims:.3f}")
    ```

    Under symmetric heavy tails ($t(3)$), both centering options perform similarly since the mean and median are both at 0. The rejection rates should be close to 0.05 for both, though the mean-centered version may be very slightly inflated due to the heavy tails affecting the variance of the absolute deviations.

---

**Exercise 4.** Explain why the $F$-distribution approximation for the Levene test statistic is valid even when the original data are non-normal.

??? success "Solution to Exercise 4"

    The Levene test applies a one-way ANOVA F-test to the transformed observations $Z_{ij} = |X_{ij} - c_i|$, not to the original $X_{ij}$. Even though the $X_{ij}$ may be non-normal, the central limit theorem ensures that the group means $\bar{Z}_{i\cdot}$ are approximately normal for moderate $n$, and the F-statistic computed from the $Z_{ij}$ approximately follows an $F(k-1, N-k)$ distribution.

    Moreover, the absolute-deviation transformation is a variance-stabilizing operation: it maps the problem from comparing variances (which requires normality for exact theory) to comparing means (which is robust due to the CLT). The ANOVA F-test is known to be robust to moderate non-normality of the input data, especially with balanced designs and moderate sample sizes. $\square$

---

**Exercise 5.** Design a simulation to estimate the **power** of the median-centered Levene test under lognormal data with group $\sigma$ parameters $(1.0, 1.0, 1.5)$ (so variances are genuinely unequal). Compare with the power under normal data with the same standard deviations.

??? success "Solution to Exercise 5"

    ```python
    import numpy as np
    from scipy.stats import levene

    rng = np.random.default_rng(0)
    n_sims, n, alpha = 5000, 30, 0.05

    for dist in ["normal", "lognormal"]:
        rej = 0
        for _ in range(n_sims):
            if dist == "normal":
                g1 = rng.normal(0, 1.0, n)
                g2 = rng.normal(0, 1.0, n)
                g3 = rng.normal(0, 1.5, n)
            else:
                g1 = rng.lognormal(0, 1.0, n)
                g2 = rng.lognormal(0, 1.0, n)
                g3 = rng.lognormal(0, 1.5, n)
            _, p = levene(g1, g2, g3, center='median')
            if p < alpha:
                rej += 1
        print(f"{dist:10s}: power = {rej/n_sims:.3f}")
    ```

    Under normal data, the power will be moderate (the test detects the difference in $\sigma$). Under lognormal data, the power may be higher or lower depending on the interplay between the skewness and the variance difference. The important point is that the test's power is "honest" -- the Type I error rate is controlled, so rejections represent genuine detection of variance inequality.
