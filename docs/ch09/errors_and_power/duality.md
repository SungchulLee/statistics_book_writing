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

## Exercises

**Exercise 1.**
A 95% confidence interval for $\mu$ is $(12.3, 18.7)$. Without computing a test statistic, determine the result of a two-sided test of $H_0: \mu = 10$ at $\alpha = 0.05$.

??? success "Solution to Exercise 1"
    Since $\mu_0 = 10$ falls **outside** the 95% confidence interval $(12.3, 18.7)$, we **reject** $H_0: \mu = 10$ at $\alpha = 0.05$. By the duality between confidence intervals and hypothesis tests, any value outside the $(1-\alpha)$ CI would be rejected at significance level $\alpha$.

---

**Exercise 2.**
A researcher conducts a two-sided test of $H_0: \mu = 50$ and obtains a p-value of 0.03. What can you conclude about whether 50 is inside or outside the 95% and 99% confidence intervals?

??? success "Solution to Exercise 2"
    Since $p = 0.03 < 0.05$, the test rejects $H_0$ at $\alpha = 0.05$. By duality, $\mu_0 = 50$ is **outside** the 95% confidence interval.

    Since $p = 0.03 > 0.01$, the test does not reject $H_0$ at $\alpha = 0.01$. By duality, $\mu_0 = 50$ is **inside** the 99% confidence interval.

---

**Exercise 3.**
Explain why a confidence interval provides more information than a hypothesis test, even though they are mathematically equivalent.

??? success "Solution to Exercise 3"
    A hypothesis test produces a binary decision: reject or fail to reject $H_0$ for a single hypothesized value $\mu_0$. A confidence interval simultaneously shows which values of $\mu_0$ would be rejected and which would not. It provides:

    1. The **direction** of the effect (is the estimate above or below $\mu_0$?).
    2. The **magnitude** of the effect (how far is the estimate from $\mu_0$?).
    3. The **precision** of the estimate (how wide is the interval?).

    For example, a CI of $(0.1, 15.2)$ and a CI of $(7.5, 7.8)$ both reject $\mu_0 = 0$ at the 5% level, but the first suggests a highly uncertain estimate, while the second indicates a precise estimate near 7.65.

---

**Exercise 4.**
Does the duality between confidence intervals and hypothesis tests hold for one-sided tests? If so, what is the corresponding confidence bound?

??? success "Solution to Exercise 4"
    The duality extends to one-sided tests, but the corresponding confidence construct is a **one-sided confidence bound** rather than a two-sided interval.

    For a one-sided test $H_0: \mu \leq \mu_0$ vs $H_1: \mu > \mu_0$ at level $\alpha$, the corresponding construct is a lower confidence bound: $(\bar{x} - z_\alpha \cdot \text{SE},\; \infty)$. We reject $H_0$ if and only if $\mu_0$ falls below this lower bound.

    Similarly, for $H_0: \mu \geq \mu_0$ vs $H_1: \mu < \mu_0$, the corresponding construct is an upper confidence bound: $(-\infty,\; \bar{x} + z_\alpha \cdot \text{SE})$. The one-sided bound uses $z_\alpha$ rather than $z_{\alpha/2}$, reflecting the one-tailed nature of the test.
