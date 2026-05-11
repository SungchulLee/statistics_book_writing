# CI for σ²

## Confidence Interval for the Population Variance

When the goal is to estimate the variability in a population, we construct a confidence interval for the population variance $\sigma^2$ (or equivalently, the population standard deviation $\sigma$).

### Formula

$$
\left[\frac{(n-1)s^2}{\chi^2_{\alpha/2,\,n-1}},\;\frac{(n-1)s^2}{\chi^2_{1-\alpha/2,\,n-1}}\right]
$$

where

- $s^2$ is the sample variance (with Bessel's correction, $\text{ddof}=1$),
- $n - 1$ is the degrees of freedom,
- $\chi^2_{\alpha/2, n-1}$ and $\chi^2_{1-\alpha/2, n-1}$ are the lower and upper critical values from the chi-square distribution.

### Sampling Distribution

The pivotal quantity is

$$
\frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}
$$

This result holds **exactly** when the population is normally distributed.

### Conditions for Validity

$$
\text{Chi-square CI for } \sigma^2
\quad\text{if}\quad
\begin{cases}
\text{population distribution is normal, so that sampling distribution is known exactly} \\
n \le 0.1N \text{ (IID approximation)}
\end{cases}
$$

!!! warning "Critical Normality Assumption"
    This CI is **exact only for Normal data**. The pivotal result $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$ holds **if and only if** the population is Normal. If the population is skewed or heavy-tailed, this relationship breaks and the chi-square CI can under- or over-cover, even for large $n$. The CLT that helps means does **not** rescue this variance CI.

### When to Use

- Data plausibly come from a **Normal population** (check with histogram or Q-Q plot; look for symmetry and light tails).
- Measurement-error or process data that are well-modeled by Normal noise.
- Teaching and demonstration of exact small-sample inference under Normality.

### When to Be Cautious

- **Skewed or heavy-tailed** data or notable outliers → chi-square CI can miscover. Consider a bootstrap CI for $\sigma$ or $\sigma^2$ (percentile or BCa), or use a robust scale estimator (e.g., MAD) with bootstrap.
- **Transformations** (e.g., log) may normalize, but then the CI is for the variance on the transformed scale.
- For comparing two variances, the F-interval has the same Normality requirement.

### Python Code

```python
import numpy as np
from scipy.stats import chi2

# Given data
n = 12
sigma = 2.0       # true population std dev (for simulation)
alpha = 0.05       # significance level

# Simulate a sample
rng = np.random.default_rng(42)
x = rng.normal(loc=0, scale=sigma, size=n)

# Sample variance
s2 = x.var(ddof=1)
df = n - 1

# Chi-square critical values
chi2_lo = chi2(df=df).ppf(alpha / 2.0)
chi2_hi = chi2(df=df).ppf(1 - alpha / 2.0)

# Confidence interval for σ²
ci_lower = df * s2 / chi2_hi
ci_upper = df * s2 / chi2_lo

print(f"95% CI for σ²: ({ci_lower:.4f}, {ci_upper:.4f})")
print(f"95% CI for σ:  ({np.sqrt(ci_lower):.4f}, {np.sqrt(ci_upper):.4f})")
```

---

## Simulation: Variance CI Coverage

The following script simulates many samples from a Normal population, constructs chi-square CIs for $\sigma^2$, and tracks how many intervals capture the true variance.

```python
#!/usr/bin/env python3
"""
Variance CI via Chi-square simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

rng_seed = None
n_simulations = 100
n_samples = 12
mu = 0.0
sigma = 2.0
alpha = 0.05
report_sigma_not_sigma2 = False  # if True, show CI for σ instead of σ²


def main():
    if rng_seed is not None:
        np.random.seed(rng_seed)

    true_var = sigma**2
    lowers = np.empty(n_simulations)
    uppers = np.empty(n_simulations)
    centers = np.empty(n_simulations)

    df = n_samples - 1
    chi2_lo = chi2(df=df).ppf(alpha / 2.0)
    chi2_hi = chi2(df=df).ppf(1 - alpha / 2.0)

    for i in range(n_simulations):
        x = np.random.normal(loc=mu, scale=sigma, size=n_samples)
        s2 = x.var(ddof=1)
        lowers[i] = df * s2 / chi2_hi
        uppers[i] = df * s2 / chi2_lo
        centers[i] = s2

    covered = (lowers <= true_var) & (true_var <= uppers)
    n_fail = int((~covered).sum())
    coverage_pct = 100.0 * covered.mean()

    if report_sigma_not_sigma2:
        lowers, uppers, centers = np.sqrt(lowers), np.sqrt(uppers), np.sqrt(centers)
        true_ref = np.sqrt(true_var)
        x_label = "Standard Deviation (σ)"
    else:
        true_ref = true_var
        x_label = "Variance (σ²)"

    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(n_simulations):
        color = "k" if covered[i] else "r"
        ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
        ax.plot(centers[i], i, marker="o", ms=3, color=color)

    ax.axvline(true_ref, linestyle="--", linewidth=1.5, color="r")
    ax.set_title(
        f"{n_simulations} Chi-square CIs | n={n_samples}, df={df}, "
        f"CL={int((1 - alpha) * 100)}% | Fail={n_fail} (Coverage ≈ {coverage_pct:.1f}%)")
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel(x_label)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

## Exercises

**Exercise 1.**
Ball-bearings, $n = 15$, $s^2 = 0.0025$ mm². (a) 95% CI for $\sigma^2$. (b) 95% CI for $\sigma$.

??? success "Solution to Exercise 1"
    (a) $\chi^2_{14, 0.025} = 5.629$, $\chi^2_{14, 0.975} = 26.119$.

    CI for $\sigma^2$: $(14 \cdot 0.0025/26.119, 14 \cdot 0.0025/5.629) = (0.00134, 0.00622)$.

    (b) CI for $\sigma$ = $(\sqrt{0.00134}, \sqrt{0.00622}) = (0.0366, 0.0789)$ mm.

    Note: monotonic transformation of CI for $\sigma^2$ gives valid CI for $\sigma$ — same coverage probability.

---

**Exercise 2.**
**Pivot quantity.** Show $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$ is a pivot under normality.

??? success "Solution to Exercise 2"
    A **pivot** has a known distribution that doesn't depend on unknown parameters.

    $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$ — the chi-squared distribution depends only on $n$, not on $\sigma^2$ or $\mu$.

    So we can write $P(\chi^2_{n-1, \alpha/2} \le (n-1)S^2/\sigma^2 \le \chi^2_{n-1, 1-\alpha/2}) = 1 - \alpha$.

    Inverting for $\sigma^2$ gives the CI.

    Other pivots: $(\bar X - \mu)/(s/\sqrt n) \sim t_{n-1}$ for the mean CI under normality.

---

**Exercise 3.**
**Asymmetric CI.** Why is the CI for $\sigma^2$ asymmetric about $S^2$?

??? success "Solution to Exercise 3"
    Chi-squared is **right-skewed** — its quantiles are asymmetric about its mean.

    For $\chi^2_{14}$ (Exercise 1): lower 2.5%-tile is 5.629; upper 97.5%-tile is 26.119. Mean is 14.

    Distance from mean to lower: $14 - 5.6 \approx 8.4$. Distance from mean to upper: $26 - 14 = 12$. Upper tail is further out.

    When inverting to CI for $\sigma^2$: lower bound is closer to $S^2$ (uses larger denominator); upper bound is farther (uses smaller denominator).

    For large $n$, chi-squared approaches normal and the asymmetry shrinks. By $n = 100$ the CI is nearly symmetric.

---

**Exercise 4.**
**Sample size for variance CI.** What $n$ ensures CI for $\sigma$ has half-width $\le 10\%$ of $\sigma$?

??? success "Solution to Exercise 4"
    For large $n$, $\sqrt{(n-1) S^2/\sigma^2} \sim \sqrt{\chi^2_{n-1}}$ which is approximately $N(\sqrt{n-1}, 1/2)$ — uses Wilson-Hilferty approximation.

    So $S/\sigma \approx N(1, 1/(2(n-1)))$, meaning $\mathrm{SE}(\log S) \approx 1/\sqrt{2(n-1)}$.

    95% CI for $\sigma$ has relative half-width $\approx 1.96/\sqrt{2(n-1)}$. Setting = 0.10:

    $\sqrt{2(n-1)} = 19.6 \Rightarrow n - 1 \approx 192 \Rightarrow n \approx 193$.

    So roughly $n = 200$ for relative half-width 10%. Variance estimation needs surprisingly large samples — much larger than for mean estimation.

---

**Exercise 5.**
**Non-normal data and CI for $\sigma^2$.** Why is the chi-squared-based CI fragile under non-normality?

??? success "Solution to Exercise 5"
    The pivot $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$ relies on **normality** of the underlying data. For non-normal data, this fails — even for large $n$.

    Specifically: $S^2$'s sampling distribution depends on the **fourth moment** (kurtosis) of the population. Heavy-tailed populations make $S^2$ much more variable than the chi-squared formula suggests.

    **Consequence:** for non-normal data, the nominal 95% CI may have actual coverage 70-80%. Heavy-tailed data is particularly affected.

    **Robust alternatives:**

    - **Bootstrap CI for $\sigma^2$:** resamples capture the actual sampling distribution.
    - **Median absolute deviation (MAD):** robust scale estimator; CI via bootstrap.
    - **Trimmed standard deviation:** robust to extreme observations.

    Always check normality (Q-Q plot) before using chi-squared CI for variance. When in doubt, bootstrap.

---

**Exercise 6.**
**CI for $\sigma$ vs $\sigma^2$.** Why are they different intervals even though $\sigma = \sqrt{\sigma^2}$?

??? success "Solution to Exercise 6"
    Two answers:

    **(a) Monotonic transformation:** since $\sqrt{\cdot}$ is monotonic, applying it to endpoints of a CI for $\sigma^2$ gives a valid CI for $\sigma$ (with the same coverage). So they ARE the same in the sense that one is obtained from the other.

    **(b) Direct construction via $S$:** alternatively, you could base inference on $S$ directly. But $S$'s exact distribution is more complicated (Chi distribution); $S^2$ has the clean Chi-squared distribution. So the standard approach: build CI for $\sigma^2$, take square root for CI for $\sigma$.

    **Symmetry:** CI for $\sigma$ is *more* symmetric than CI for $\sigma^2$ (square root reduces asymmetry). But it remains asymmetric.

    **Bias:** $S$ is a biased estimator of $\sigma$ (Jensen). $S^2$ is unbiased for $\sigma^2$. Bias correction factors like $c_4$ exist for $\sigma$ but not commonly applied.
