# Interpretation and Common Misconceptions

## The Repeated-Sampling Interpretation

A 95% confidence interval does **not** mean "there is a 95% probability that $\mu$ is in this interval." The parameter $\mu$ is a fixed (but unknown) number — it either is or is not in the interval.

The correct interpretation: if we were to repeat the sampling process many times, each time constructing a 95% CI, then **approximately 95% of those intervals would contain the true parameter**.

### Formal Statement

Let $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$ with $\sigma$ known. The interval

$$\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

satisfies:

$$P\left(\bar{X} - z_{\alpha/2}\frac{\sigma}{\sqrt{n}} \leq \mu \leq \bar{X} + z_{\alpha/2}\frac{\sigma}{\sqrt{n}}\right) = 1 - \alpha$$

The probability statement is about the **random endpoints** $\bar{X} \pm z_{\alpha/2}\sigma/\sqrt{n}$, not about $\mu$.

### Simulation Demonstration

```python
import numpy as np
np.random.seed(42)

mu, sigma, n = 50, 10, 30
alpha = 0.05
z = 1.96
n_simulations = 1000

covers = 0
for _ in range(n_simulations):
    sample = np.random.normal(mu, sigma, n)
    xbar = sample.mean()
    me = z * sigma / np.sqrt(n)
    lower, upper = xbar - me, xbar + me
    if lower <= mu <= upper:
        covers += 1

print(f"Coverage: {covers}/{n_simulations} = {covers/n_simulations:.3f}")
# ≈ 0.950
```

## Common Misconceptions

### Misconception 1: "95% probability that μ is in this interval"

After computing $[48.2, 51.8]$, the statement "there is a 95% probability that $\mu$ is between 48.2 and 51.8" is **wrong**. Either $\mu$ is in that interval or it is not — there is no randomness left.

The 95% refers to the **procedure**, not any single interval.

### Misconception 2: "95% of the data falls in the interval"

A CI estimates a **parameter** (like the population mean), not the range of individual observations. The interval $\bar{X} \pm z_{\alpha/2}\sigma/\sqrt{n}$ shrinks with $n$, while the range of data does not.

### Misconception 3: "If two CIs overlap, the difference is not significant"

Two 95% CIs can overlap even when the difference between parameters is statistically significant. The proper comparison uses a CI for the **difference** $\mu_1 - \mu_2$.

## Width, Confidence Level, and Sample Size

The margin of error for a z-interval is:

$$E = z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

Three relationships follow:

1. **Higher confidence → wider interval.** Increasing from 95% to 99% increases $z_{\alpha/2}$ from 1.96 to 2.576, widening the interval by 31%.

2. **Larger sample → narrower interval.** The margin of error decreases as $1/\sqrt{n}$. To halve the width, you need 4 times the sample size.

3. **Larger variance → wider interval.** More variability in the population makes estimation harder.

### Sample Size Determination

To achieve a desired margin of error $E$ at confidence level $1 - \alpha$:

$$n = \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2$$

**Example:** To estimate a population mean within $\pm 2$ units with 95% confidence, given $\sigma = 10$:

$$n = \left(\frac{1.96 \times 10}{2}\right)^2 = 96.04 \implies n = 97$$

### Common Confidence Levels

| Confidence Level | $\alpha$ | $z_{\alpha/2}$ |
|---|---|---|
| 90% | 0.10 | 1.645 |
| 95% | 0.05 | 1.960 |
| 99% | 0.01 | 2.576 |

## One-Sided Confidence Intervals (Confidence Bounds)

Sometimes we only need a bound in one direction:

- **Upper bound:** $\mu \leq \bar{X} + z_\alpha \cdot \sigma/\sqrt{n}$ (with confidence $1 - \alpha$)
- **Lower bound:** $\mu \geq \bar{X} - z_\alpha \cdot \sigma/\sqrt{n}$ (with confidence $1 - \alpha$)

Note that one-sided bounds use $z_\alpha$ (not $z_{\alpha/2}$). A 95% one-sided bound uses $z_{0.05} = 1.645$.

**Financial example:** A risk manager may want an upper bound on portfolio loss: "We are 95% confident that the expected loss does not exceed $X."
