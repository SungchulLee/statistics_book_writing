# One-Sample Proportion Test

## Overview

The one-sample proportion test assesses whether a population proportion $p$ equals a hypothesized value $p_0$. The **Wald z-test** uses a normal approximation to the binomial distribution and is valid when $np_0$ and $n(1-p_0)$ are both at least 5. For small samples, the **exact binomial test** provides an alternative that does not rely on asymptotic approximations.

## Test Formulation

**Hypotheses:**

- Two-sided: $H_0\colon p = p_0$ vs $H_1\colon p \neq p_0$
- One-sided: $H_0\colon p = p_0$ vs $H_1\colon p > p_0$ (or $H_1\colon p < p_0$)

**Wald z-test:** Given $k$ successes in $n$ trials with $\hat{p} = k/n$, the test statistic is

$$
Z = \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}} \;\dot\sim\; N(0,1).
$$

Note that the standard error uses $p_0$ (not $\hat{p}$) because we compute under the null hypothesis.

**Exact binomial test:** Compute the p-value directly from the binomial distribution $\text{Bin}(n, p_0)$ without normal approximation.

## Code

```python
from scipy.stats import norm, binomtest
import math

def test_prop_one_sample(k, n, p0=0.5, method="wald",
                         alt="two-sided", alpha=0.05):
    """
    method='wald' (normal approx) or 'exact' (binomial).
    Returns (stat_or_None, pvalue, reject_bool, label).
    """
    phat = k / n
    if method == "exact":
        p = binomtest(k, n, p0, alternative=alt).pvalue
        return None, p, (p < alpha), "exact binomial"

    se0 = math.sqrt(p0 * (1 - p0) / n)
    z = (phat - p0) / se0
    if alt == "two-sided":
        p = 2 * min(norm.cdf(z), 1 - norm.cdf(z))
    elif alt == "less":
        p = norm.cdf(z)
    else:
        p = 1 - norm.cdf(z)
    return z, p, (p < alpha), "wald z-test"
```

### Example

```python
stat, p, reject, label = test_prop_one_sample(
    k=12, n=50, p0=0.2, method="wald", alt="two-sided"
)
print(label, "stat:", stat, "p:", p, "reject:", reject)
```

### Interpretation

We test $H_0\colon p = 0.2$ with $k = 12$ successes out of $n = 50$, giving $\hat{p} = 0.24$. The test statistic is

$$
Z = \frac{0.24 - 0.20}{\sqrt{0.20 \times 0.80 / 50}} = \frac{0.04}{0.0566} \approx 0.707.
$$

The two-sided p-value is approximately 0.48, so we fail to reject $H_0$. The observed proportion is consistent with $p = 0.20$.

## Exercises

**Exercise 1.** In a sample of $n = 200$ manufactured items, 18 are defective. Test $H_0\colon p = 0.05$ vs $H_1\colon p > 0.05$ at $\alpha = 0.01$.

??? success "Solution to Exercise 1"

    We have $\hat{p} = 18/200 = 0.09$. The test statistic is

    $$
    Z = \frac{0.09 - 0.05}{\sqrt{0.05 \times 0.95 / 200}} = \frac{0.04}{\sqrt{0.0002375}} = \frac{0.04}{0.01541} \approx 2.596.
    $$

    The one-sided p-value is $P(Z \geq 2.596) \approx 0.0047$. Since $0.0047 < 0.01$, we reject $H_0$. There is significant evidence that the defect rate exceeds 5%. $\square$

---

**Exercise 2.** A coin is flipped 100 times and comes up heads 60 times. Test whether the coin is fair at $\alpha = 0.05$ using both the Wald test and the exact binomial test. Compare the p-values.

??? success "Solution to Exercise 2"

    **Wald test:** $\hat{p} = 0.60$, $p_0 = 0.50$.

    $$
    Z = \frac{0.60 - 0.50}{\sqrt{0.50 \times 0.50/100}} = \frac{0.10}{0.05} = 2.0.
    $$

    Two-sided p-value: $2 \times P(Z \geq 2.0) = 2(0.0228) = 0.0456$. Reject $H_0$.

    **Exact binomial test:** $p = 2 \times P(X \geq 60 \mid X \sim \text{Bin}(100, 0.5)) \approx 0.0569$. Fail to reject $H_0$.

    The Wald test rejects while the exact test does not, illustrating that the normal approximation can be slightly anti-conservative. For borderline cases, the exact test is more reliable. $\square$

---

**Exercise 3.** Derive the Wald test statistic from the Central Limit Theorem applied to the binomial distribution.

??? success "Solution to Exercise 3"

    Let $X_1, \dots, X_n \overset{\text{iid}}{\sim} \text{Bernoulli}(p)$. The sample proportion is $\hat{p} = \bar{X} = \sum X_i / n$. Under $H_0\colon p = p_0$, $E[\hat{p}] = p_0$ and $\text{Var}(\hat{p}) = p_0(1-p_0)/n$. By the CLT,

    $$
    \frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}} \xrightarrow{d} N(0,1)
    $$

    as $n \to \infty$. This standardized quantity is exactly the Wald test statistic $Z$. The approximation is accurate when $np_0 \geq 5$ and $n(1-p_0) \geq 5$. $\square$

---

**Exercise 4.** A political pollster surveys $n = 1000$ voters and finds that 540 support a candidate. Construct a 95% confidence interval for $p$ and use CI--test duality to test $H_0\colon p = 0.50$.

??? success "Solution to Exercise 4"

    The Wald 95% CI for $p$ is

    $$
    \hat{p} \pm z_{0.025}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}} = 0.54 \pm 1.96\sqrt{\frac{0.54 \times 0.46}{1000}} = 0.54 \pm 1.96(0.01577) = 0.54 \pm 0.0309.
    $$

    The interval is $(0.509, 0.571)$. Since $p_0 = 0.50$ is not in this interval, we reject $H_0\colon p = 0.50$ at $\alpha = 0.05$. $\square$

---

**Exercise 5.** Show that the exact binomial test at level $\alpha$ for $H_0\colon p = p_0$ vs $H_1\colon p > p_0$ rejects when $k \geq c$, where $c$ is the smallest integer satisfying $P(X \geq c \mid X \sim \text{Bin}(n, p_0)) \leq \alpha$.

??? success "Solution to Exercise 5"

    The p-value of the one-sided test is

    $$
    p\text{-value} = P(X \geq k \mid p = p_0) = \sum_{j=k}^{n} \binom{n}{j} p_0^j (1-p_0)^{n-j}.
    $$

    We reject $H_0$ when this p-value is at most $\alpha$. The critical value $c$ is the smallest integer such that

    $$
    P(X \geq c \mid p = p_0) = \sum_{j=c}^{n} \binom{n}{j} p_0^j (1-p_0)^{n-j} \leq \alpha.
    $$

    Since $P(X \geq k)$ is a decreasing step function of $k$, any observed $k \geq c$ yields a p-value no larger than $\alpha$, and any $k < c$ yields a p-value exceeding $\alpha$. The test is discrete, so the achieved significance level may be strictly less than $\alpha$. $\square$
