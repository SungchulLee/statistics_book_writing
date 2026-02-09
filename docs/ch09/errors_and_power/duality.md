# Confidence Interval ↔ Hypothesis Test Duality

## The Duality Principle

There is a deep connection between confidence intervals and hypothesis tests. A $(1 - \alpha) \times 100\%$ confidence interval and a hypothesis test at significance level $\alpha$ are two sides of the same coin:

> **A two-sided hypothesis test at level $\alpha$ rejects $H_0: \theta = \theta_0$ if and only if $\theta_0$ falls outside the $(1-\alpha) \times 100\%$ confidence interval for $\theta$.**

This duality means that you can perform a hypothesis test by examining a confidence interval, and vice versa.

## How the Duality Works

### From Confidence Interval to Hypothesis Test

Given a $(1 - \alpha) \times 100\%$ confidence interval $(L, U)$ for a parameter $\theta$:

- If $\theta_0 \in (L, U)$: Fail to reject $H_0: \theta = \theta_0$ at significance level $\alpha$.
- If $\theta_0 \notin (L, U)$: Reject $H_0: \theta = \theta_0$ at significance level $\alpha$.

### From Hypothesis Test to Confidence Interval

A $(1 - \alpha) \times 100\%$ confidence interval is the set of all values $\theta_0$ for which the hypothesis test $H_0: \theta = \theta_0$ would **not** be rejected at significance level $\alpha$.

$$CI_{1-\alpha} = \{\theta_0 : \text{fail to reject } H_0: \theta = \theta_0 \text{ at level } \alpha\}$$

## Examples

### Example 1: One-Sample Mean

For a one-sample z-test of $H_0: \mu = \mu_0$ vs $H_a: \mu \neq \mu_0$:

- **Test**: Reject $H_0$ if $|z| > z_{\alpha/2}$, where $z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$.
- **CI**: $\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$

The test rejects $H_0$ if and only if $\mu_0$ lies outside the confidence interval.

**Proof of equivalence:**

$$|z| > z_{\alpha/2} \iff \left|\frac{\bar{x} - \mu_0}{\sigma/\sqrt{n}}\right| > z_{\alpha/2} \iff \mu_0 \notin \left(\bar{x} - z_{\alpha/2}\frac{\sigma}{\sqrt{n}},\ \bar{x} + z_{\alpha/2}\frac{\sigma}{\sqrt{n}}\right)$$

### Example 2: Two Varieties of Pears

Yuna compares caloric content of Bosc and Anjou pears. The 99% confidence interval for $\mu_{\text{Bosc}} - \mu_{\text{Anjou}}$ is $4 \pm 6.44 = (-2.44, 10.44)$.

Testing $H_0: \mu_{\text{Bosc}} = \mu_{\text{Anjou}}$ (i.e., $\mu_{\text{Bosc}} - \mu_{\text{Anjou}} = 0$) at $\alpha = 0.01$:

Since $0 \in (-2.44, 10.44)$, we **fail to reject** $H_0$. There is not enough evidence to conclude the caloric contents differ.

### Example 3: In-person vs Online Classes

A 95% confidence interval for $p_{\text{in\_person}} - p_{\text{online}}$ is $(-0.04, 0.14)$.

Testing $H_0: p_{\text{in\_person}} = p_{\text{online}}$ at $\alpha = 0.05$:

Since $0 \in (-0.04, 0.14)$, we **fail to reject** $H_0$. There is no significant difference in passing rates.

## One-Sided Tests and Confidence Intervals

The duality extends to one-sided tests using one-sided confidence intervals (confidence bounds):

- **Upper confidence bound**: $\theta < U$ at confidence level $1 - \alpha$ corresponds to the test $H_0: \theta \geq \theta_0$ vs $H_a: \theta < \theta_0$.
- **Lower confidence bound**: $\theta > L$ at confidence level $1 - \alpha$ corresponds to the test $H_0: \theta \leq \theta_0$ vs $H_a: \theta > \theta_0$.

## Python Illustration

```python
import numpy as np
from scipy import stats

# Sample data
x_bar = 52
mu_0 = 50
sigma = 10
n = 25
alpha = 0.05

# Hypothesis test approach
z = (x_bar - mu_0) / (sigma / np.sqrt(n))
p_value = 2 * stats.norm.sf(abs(z))
reject_test = p_value <= alpha

# Confidence interval approach
z_crit = stats.norm.ppf(1 - alpha / 2)
ci_lower = x_bar - z_crit * sigma / np.sqrt(n)
ci_upper = x_bar + z_crit * sigma / np.sqrt(n)
reject_ci = mu_0 < ci_lower or mu_0 > ci_upper

print(f"Test: z = {z:.4f}, p-value = {p_value:.4f}, Reject = {reject_test}")
print(f"CI: ({ci_lower:.4f}, {ci_upper:.4f}), mu_0 outside CI = {reject_ci}")
print(f"Both methods agree: {reject_test == reject_ci}")
```

## Key Takeaways

- Confidence intervals and hypothesis tests provide equivalent information for two-sided tests.
- Confidence intervals are often more informative because they show the range of plausible values, not just a binary reject/fail-to-reject decision.
- When reporting results, it is good practice to report both the p-value and the confidence interval.
- The duality holds exactly for two-sided tests; one-sided tests correspond to one-sided confidence bounds.
