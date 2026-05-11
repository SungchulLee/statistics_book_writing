# Bootstrap Standard Error

## Overview

When the theoretical standard error formula is unavailable or the statistic of interest is complex (e.g., a median, ratio, or regression coefficient), the **bootstrap** provides a powerful computational method for estimating the standard error. The idea is simple: resample with replacement from the observed data many times, compute the statistic each time, and use the standard deviation of these bootstrap replicates as the estimated standard error. This page compares the bootstrap SE with the classical formula for the sample mean.

## The Bootstrap Principle

Given an observed sample $x_1, x_2, \ldots, x_n$, the bootstrap algorithm for estimating $\text{SE}(\hat{\theta})$ is:

1. Draw a bootstrap sample $x_1^*, x_2^*, \ldots, x_n^*$ by sampling **with replacement** from the original data.
2. Compute the statistic of interest: $\hat{\theta}^* = T(x_1^*, \ldots, x_n^*)$.
3. Repeat steps 1--2 a total of $B$ times, producing $\hat{\theta}_1^*, \hat{\theta}_2^*, \ldots, \hat{\theta}_B^*$.
4. The bootstrap standard error is:

$$
\widehat{\text{SE}}_{\text{boot}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^B \left(\hat{\theta}_b^* - \bar{\hat{\theta}}^*\right)^2}
$$

where $\bar{\hat{\theta}}^* = \frac{1}{B}\sum_{b=1}^B \hat{\theta}_b^*$.

!!! info "Why Resample with Replacement?"
    Sampling with replacement from the data mimics the process of drawing new samples from the (unknown) population. The empirical distribution of the data serves as a nonparametric estimate of the population distribution.

## Classical Comparison

For the sample mean $\bar{x}$, the classical standard error formula is:

$$
\text{SE}(\bar{x}) = \frac{s}{\sqrt{n}}
$$

where $s$ is the sample standard deviation. The bootstrap SE should approximate this value when applied to the mean.

## Alternative: Squared-Error Approach

An equivalent formulation computes:

$$
\widehat{\text{SE}} = \sqrt{\frac{1}{B}\sum_{b=1}^B \left(\bar{x}_b^* - \bar{x}\right)^2}
$$

This replaces the bootstrap grand mean $\bar{\hat{\theta}}^*$ with the original sample mean $\bar{x}$. For large $B$, both approaches give nearly identical results.

## Simulation

The following code applies both the classical and bootstrap approaches to a sample of 31 price observations.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# Sample data (31 price observations)
data = np.array([
    245.02, 244.88, 244.76, 244.65, 244.53, 244.42, 244.30,
    244.18, 244.08, 243.97, 243.85, 243.74, 243.63, 243.52,
    243.40, 243.28, 243.17, 243.06, 242.95, 242.83, 242.72,
    242.61, 242.49, 242.38, 242.27, 242.15, 242.04, 241.93,
    241.81, 241.70, 241.59,
])

n = len(data)
n_boot = 10_000

# Classical SE
se_classical = data.std(ddof=1) / np.sqrt(n)

# Bootstrap SE (standard approach)
boot_means = np.array([
    np.random.choice(data, size=n, replace=True).mean()
    for _ in range(n_boot)
])
se_bootstrap = boot_means.std(ddof=1)

# Bootstrap SE (squared-error approach)
sq_errors = np.array([
    (np.random.choice(data, size=n, replace=True).mean() - data.mean()) ** 2
    for _ in range(n_boot)
])
se_squared_error = np.sqrt(sq_errors.mean())

print(f"Classical SE:       {se_classical:.4f}")
print(f"Bootstrap SE:       {se_bootstrap:.4f}")
print(f"Squared-error SE:   {se_squared_error:.4f}")
```

## Visualization

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left panel: bootstrap distribution
ax = axes[0]
ax.hist(boot_means, bins=40, edgecolor="white", alpha=0.7)
ax.axvline(data.mean(), color="red", linestyle="--",
           label=f"Sample mean = {data.mean():.2f}")
ax.set_xlabel("Bootstrap sample mean")
ax.set_ylabel("Frequency")
ax.set_title(f"Bootstrap Distribution (SE = {se_bootstrap:.3f})")
ax.legend()

# Right panel: SE vs. sample size
ax = axes[1]
sizes = np.arange(5, n + 1)
se_vals = [data[:k].std(ddof=1) / np.sqrt(k) for k in sizes]
ax.plot(sizes, se_vals, marker="o", markersize=4)
ax.set_xlabel("Sample size n")
ax.set_ylabel("SE (classical)")
ax.set_title("Standard Error Decreases with n")

plt.tight_layout()
plt.show()
```

## Interpretation

!!! note "Key Observations"

    1. The classical SE, bootstrap SE, and squared-error SE all produce very similar values, confirming the consistency of the bootstrap method for the sample mean.
    2. The bootstrap distribution of $\bar{x}^*$ is approximately normal, centred at the original sample mean.
    3. As the sample size increases (right panel), the standard error decreases following the familiar $1/\sqrt{n}$ curve.

!!! tip "When to Use Bootstrap"
    The bootstrap is most valuable when:

    - The statistic has no simple formula for its standard error (e.g., median, correlation, percentiles).
    - The theoretical SE depends on unknown quantities that are hard to estimate (e.g., the population's fourth moment for $\text{SE}(S^2)$).
    - The data distribution is complex or the sample size is small and you want to avoid distributional assumptions.

### Bootstrap SE vs. Sample Size

The code also demonstrates how SE decreases as we use more of the data:

| $n$ | Classical SE | Bootstrap SE |
|---|---|---|
| 5 | larger | similar |
| 15 | moderate | similar |
| 31 (full) | smallest | similar |

The bootstrap and classical methods converge to very similar values at each sample size.

## Exercises

**Exercise 1.** Explain why bootstrap samples are drawn **with replacement** rather than without replacement from the original data of size $n$.

??? success "Solution to Exercise 1"
    If we sampled without replacement from $n$ data points and took all $n$ values, we would always get the exact same dataset, and every bootstrap replicate of any statistic would be identical to the original. There would be no variability to measure.

    Sampling with replacement introduces variability: some original observations appear multiple times in a bootstrap sample while others are omitted entirely. On average, about $1 - (1 - 1/n)^n \approx 1 - e^{-1} \approx 63.2\%$ of the original observations appear in each bootstrap sample. This variability mimics the variability that would arise from drawing new samples from the true population.

    Formally, the bootstrap treats the empirical distribution $\hat{F}_n$ (which places mass $1/n$ on each observed value) as a stand-in for the true population distribution $F$. Sampling with replacement from the data is equivalent to sampling from $\hat{F}_n$. $\square$

---

**Exercise 2.** For the price data above ($n = 31$, $s \approx 1.01$), compute the classical standard error by hand and verify it matches the simulation output.

??? success "Solution to Exercise 2"
    The sample standard deviation is:

    $$
    s = \sqrt{\frac{1}{30}\sum_{i=1}^{31}(x_i - \bar{x})^2}
    $$

    The data ranges from 241.59 to 245.02 with roughly equal spacing. The sample mean is approximately $\bar{x} \approx 243.31$. Computing $s$ (or noting it from the code output):

    $$
    s \approx 1.013
    $$

    The classical SE is:

    $$
    \text{SE} = \frac{s}{\sqrt{n}} = \frac{1.013}{\sqrt{31}} = \frac{1.013}{5.568} \approx 0.182
    $$

    This should match the simulation output closely. $\square$

---

**Exercise 3.** How many bootstrap replicates $B$ are recommended in practice? Discuss the trade-off between computation time and accuracy of the bootstrap SE estimate.

??? success "Solution to Exercise 3"
    The standard error of the bootstrap SE estimate is approximately:

    $$
    \text{SE}(\widehat{\text{SE}}_{\text{boot}}) \approx \frac{\widehat{\text{SE}}_{\text{boot}}}{\sqrt{2B}}
    $$

    Common recommendations:

    - **$B = 1{,}000$**: Sufficient for a rough estimate. The SE of the SE is about $\widehat{\text{SE}} / 44.7$, giving about 2.2% relative precision.
    - **$B = 10{,}000$**: Good for most applications. Relative precision is about 0.7%.
    - **$B = 50{,}000$ or more**: Used for bootstrap confidence intervals (which require accurate tail estimates).

    The trade-off: doubling $B$ improves precision by a factor of $\sqrt{2} \approx 1.41$ but doubles computation time. For standard error estimation, $B = 1{,}000$ to $10{,}000$ is typically sufficient. For bootstrap confidence intervals or hypothesis tests, larger $B$ is needed because tail probabilities are estimated less precisely. $\square$

---

**Exercise 4.** Apply the bootstrap to estimate the standard error of the **sample median** for the price data. Why is the bootstrap particularly useful for the median?

??? success "Solution to Exercise 4"
    ```python
    import numpy as np
    np.random.seed(42)

    data = np.array([245.02, 244.88, 244.76, 244.65, 244.53, 244.42,
                     244.30, 244.18, 244.08, 243.97, 243.85, 243.74,
                     243.63, 243.52, 243.40, 243.28, 243.17, 243.06,
                     242.95, 242.83, 242.72, 242.61, 242.49, 242.38,
                     242.27, 242.15, 242.04, 241.93, 241.81, 241.70,
                     241.59])

    boot_medians = np.array([
        np.median(np.random.choice(data, size=len(data), replace=True))
        for _ in range(10_000)
    ])
    se_median = boot_medians.std(ddof=1)
    print(f"Bootstrap SE of the median: {se_median:.4f}")
    ```

    The bootstrap is particularly useful for the median because:

    1. There is no simple, widely-known closed-form formula for the standard error of the median that works for arbitrary distributions.
    2. The asymptotic formula $\text{SE}(\text{median}) \approx 1 / (2f(m)\sqrt{n})$ requires knowing the population density $f$ at the median $m$, which is itself difficult to estimate.
    3. The bootstrap automatically accounts for the distribution shape and provides a nonparametric estimate without requiring density estimation.

    $\square$

---

**Exercise 5.** Prove that for the sample mean, the bootstrap SE converges to $s/\sqrt{n}$ as $B \to \infty$, where $s$ is the sample standard deviation.

??? success "Solution to Exercise 5"
    In a bootstrap sample, each $x_i^*$ is drawn independently and uniformly from $\{x_1, \ldots, x_n\}$. The bootstrap sample mean is $\bar{x}^* = \frac{1}{n}\sum_{j=1}^n x_j^*$.

    Under the bootstrap distribution (conditional on the data):

    $$
    E^*[x_j^*] = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}
    $$

    $$
    \text{Var}^*(x_j^*) = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2 = \frac{n-1}{n} s^2
    $$

    Since the $x_j^*$ are i.i.d. under the bootstrap:

    $$
    \text{Var}^*(\bar{x}^*) = \frac{1}{n} \cdot \frac{n-1}{n} s^2 = \frac{(n-1)s^2}{n^2}
    $$

    As $B \to \infty$, the bootstrap SE converges to:

    $$
    \widehat{\text{SE}}_{\text{boot}} \to \sqrt{\frac{(n-1)s^2}{n^2}} = \frac{s\sqrt{n-1}}{n}
    $$

    This is very close to $s/\sqrt{n}$ for large $n$ (differing by a factor of $\sqrt{(n-1)/n}$). Using $\text{ddof}=1$ in the standard deviation of the bootstrap means gives exactly $s/\sqrt{n}$ in the limit. The minor discrepancy arises from the distinction between dividing by $n$ vs. $n-1$ in the variance, and vanishes as $n \to \infty$. $\square$
