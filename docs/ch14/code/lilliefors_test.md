# Lilliefors Test

## Overview

The Lilliefors test is a modification of the Kolmogorov-Smirnov test designed for the common situation where the null-hypothesis parameters (mean and variance) are estimated from the data rather than specified in advance. By using parametric bootstrap to calibrate the null distribution of the KS statistic, the Lilliefors test provides valid $p$-values when testing composite normality.

## The Problem with the Standard KS Test

Recall that the one-sample KS statistic is

$$
D_n = \sup_x |F_n(x) - F_0(x)|.
$$

When $F_0 = \mathcal{N}(\cdot\,; \mu, \sigma)$ with $\mu$ and $\sigma$ estimated as $\hat{\mu} = \bar{X}$ and $\hat{\sigma} = S$, the fitted CDF $\hat{F}_0$ is systematically closer to $F_n$ than a pre-specified $F_0$ would be. This reduces $D_n$, inflates the $p$-value, and makes the standard KS test conservative (under-rejecting).

## Parametric Bootstrap Algorithm

The Lilliefors test corrects this by simulating the null distribution of $D_n$ under parameter estimation:

1. **Compute the observed statistic.** Fit $\hat{\mu}, \hat{\sigma}$ from the data and compute

    $$
    D_{\text{obs}} = \sup_x |F_n(x) - \mathcal{N}(x;\, \hat{\mu}, \hat{\sigma})|.
    $$

2. **Bootstrap loop.** For $b = 1, \ldots, B$:
    - Simulate $X_1^*, \ldots, X_n^* \overset{\text{iid}}{\sim} \mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$.
    - Re-estimate $\hat{\mu}^* = \bar{X}^*$ and $\hat{\sigma}^* = S^*$.
    - Compute $D_b^* = \sup_x |F_n^*(x) - \mathcal{N}(x;\, \hat{\mu}^*, \hat{\sigma}^*)|$.

3. **Compute the $p$-value.**

    $$
    p \approx \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}(D_b^* \geq D_{\text{obs}}).
    $$

### Code

```python
import numpy as np
from scipy import stats

def ks_stat_fitted_normal(x):
    x = np.asarray(x, dtype=float)
    mu, sd = x.mean(), x.std(ddof=1)
    D, _ = stats.kstest(x, 'norm', args=(mu, sd))
    return float(D), float(mu), float(sd)

def lilliefors_normal_bootstrap(x, B=2000, seed=0):
    x = np.asarray(x, dtype=float)
    n = x.size
    D_obs, mu, sd = ks_stat_fitted_normal(x)

    rng = np.random.default_rng(seed)
    D_star = np.empty(B)
    for b in range(B):
        xb = rng.normal(mu, sd, size=n)
        D_star[b], _, _ = ks_stat_fitted_normal(xb)

    p_boot = float(np.mean(D_star >= D_obs))
    return D_obs, p_boot, mu, sd

# Example: skewed data that should be rejected
rng = np.random.default_rng(1)
x = rng.lognormal(0.0, 0.6, size=300)

D, p, mu, sd = lilliefors_normal_bootstrap(x, B=1500, seed=7)
print(f"Fitted Normal: mu = {mu:.4f}, sd = {sd:.4f}")
print(f"Lilliefors KS D = {D:.4f}, bootstrap p = {p:.4f}")
if p < 0.05:
    print("=> Reject normality at alpha = 0.05.")
else:
    print("=> Fail to reject normality at alpha = 0.05.")
```

## Why the Bootstrap Works

The bootstrap generates samples under the *same* procedure used on the original data: draw from a normal, re-estimate parameters, compute the KS statistic. This replicates the exact source of the bias (parameter estimation shrinks $D$) in the reference distribution, producing correctly calibrated critical values.

## Interpretation

For the lognormal example, the Lilliefors test should reject normality ($p \approx 0$), correctly identifying the right skew. In contrast, the naive KS test with estimated parameters would yield a larger $p$-value and might fail to reject, demonstrating why the Lilliefors correction is essential.

## Exercises

**Exercise 1.** Generate $n = 200$ standard normal observations. Run both the naive KS test (with estimated $\hat{\mu}, \hat{\sigma}$) and the Lilliefors bootstrap test. Compare the two $p$-values.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=200)

    mu_hat, sd_hat = x.mean(), x.std(ddof=1)
    D_naive, p_naive = stats.kstest(x, 'norm', args=(mu_hat, sd_hat))

    # Bootstrap Lilliefors
    B = 2000
    D_obs = D_naive
    D_star = np.empty(B)
    for b in range(B):
        xb = rng.normal(mu_hat, sd_hat, size=200)
        mu_b, sd_b = xb.mean(), xb.std(ddof=1)
        D_star[b], _ = stats.kstest(xb, 'norm', args=(mu_b, sd_b))
    p_boot = np.mean(D_star >= D_obs)

    print(f"Naive KS p-value:      {p_naive:.4f}")
    print(f"Lilliefors bootstrap p: {p_boot:.4f}")
    ```

    The naive $p$-value will be larger (more conservative) than the bootstrap $p$-value because the standard KS critical values do not account for parameter estimation. Both should be above 0.05, but the difference illustrates the bias. $\square$

---

**Exercise 2.** Run a Monte Carlo experiment with 5,000 replicates to estimate the empirical size of the Lilliefors bootstrap test ($B = 500$) at $\alpha = 0.05$ for $n = 100$. Compare with the naive KS test.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n, reps, alpha, B = 100, 5000, 0.05, 500

    rej_naive, rej_boot = 0, 0
    for _ in range(reps):
        x = rng.normal(0, 1, size=n)
        mu, sd = x.mean(), x.std(ddof=1)
        D, p_naive = stats.kstest(x, 'norm', args=(mu, sd))
        if p_naive < alpha:
            rej_naive += 1

        D_star = np.empty(B)
        for b in range(B):
            xb = rng.normal(mu, sd, size=n)
            mu_b, sd_b = xb.mean(), xb.std(ddof=1)
            D_star[b], _ = stats.kstest(xb, 'norm', args=(mu_b, sd_b))
        if np.mean(D_star >= D) < alpha:
            rej_boot += 1

    print(f"Naive KS size:  {rej_naive / reps:.4f}")
    print(f"Lilliefors size: {rej_boot / reps:.4f}")
    ```

    The naive KS test should show an empirical size well below 0.05 (around 0.01--0.02), confirming its conservatism. The Lilliefors bootstrap test should be close to 0.05, demonstrating correct calibration. $\square$

---

**Exercise 3.** Explain why the Lilliefors $p$-value has a resolution of $1/B$. How large should $B$ be to reliably distinguish $p = 0.04$ from $p = 0.06$?

??? success "Solution to Exercise 3"

    The bootstrap $p$-value is $\hat{p} = \frac{1}{B}\sum_{b=1}^B \mathbf{1}(D_b^* \geq D_{\text{obs}})$, which can only take values in $\{0, 1/B, 2/B, \ldots, 1\}$. Its standard error (under the null) is $\sqrt{\hat{p}(1-\hat{p})/B}$. To distinguish $p = 0.04$ from $p = 0.06$, we need the standard error to be much smaller than $0.01$. At $p \approx 0.05$, $\text{SE} = \sqrt{0.05 \times 0.95/B}$. Setting $\text{SE} < 0.005$ gives $B > 0.0475/0.000025 = 1900$. In practice, $B \geq 2000$ is a reasonable minimum for reliable inference near the 5% threshold. $\square$

---

**Exercise 4.** Modify the Lilliefors bootstrap to test for exponentiality instead of normality. Outline the required changes to the algorithm.

??? success "Solution to Exercise 4"

    The changes are:

    1. **Parameter estimation:** Replace $\hat{\mu}, \hat{\sigma}$ with the MLE of the exponential rate, $\hat{\lambda} = 1/\bar{X}$.
    2. **Observed statistic:** Compute $D_{\text{obs}} = \sup_x |F_n(x) - (1 - e^{-\hat{\lambda} x})|$.
    3. **Bootstrap loop:** Simulate $X_b^* \sim \text{Exp}(\hat{\lambda})$, re-estimate $\hat{\lambda}_b^* = 1/\bar{X}_b^*$, and compute $D_b^*$ against the re-fitted exponential CDF.
    4. **$p$-value:** Same as before, $\hat{p} = \frac{1}{B}\sum \mathbf{1}(D_b^* \geq D_{\text{obs}})$.

    The key principle is the same: the bootstrap replicates the parameter-estimation step so that the reference distribution of $D$ is correctly calibrated. $\square$

---

**Exercise 5.** Prove that the bootstrap $p$-value $\hat{p}_B$ converges to the true $p$-value as $B \to \infty$ (for fixed data).

??? success "Solution to Exercise 5"

    For fixed data, $D_{\text{obs}}$ is a constant. Each bootstrap draw produces $D_b^*$, and $\mathbf{1}(D_b^* \geq D_{\text{obs}})$ is a Bernoulli random variable with success probability $p^* = P^*(D^* \geq D_{\text{obs}})$, where $P^*$ denotes the bootstrap distribution (sampling from $\mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$). The bootstrap $p$-value is

    $$
    \hat{p}_B = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}(D_b^* \geq D_{\text{obs}}).
    $$

    By the strong law of large numbers (the $D_b^*$ are i.i.d. conditional on the data), $\hat{p}_B \xrightarrow{\text{a.s.}} p^*$ as $B \to \infty$. The quantity $p^*$ is the exact Lilliefors $p$-value under the fitted null. The convergence rate is $O(1/\sqrt{B})$ by the CLT, so the standard error of $\hat{p}_B$ is $\sqrt{p^*(1-p^*)/B}$. $\square$
