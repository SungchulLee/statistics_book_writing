# One-Sample Mean Test

## Overview

The one-sample mean test determines whether a population mean $\mu$ equals a hypothesized value $\mu_0$. When the population variance is known, the **z-test** is used; when it is unknown and estimated from the sample, the **t-test** is appropriate. Both tests rely on the assumption that the data are normally distributed or that the sample size is large enough for the Central Limit Theorem to apply.

## Test Formulation

**Hypotheses:**

- Two-sided: $H_0\colon \mu = \mu_0$ vs $H_1\colon \mu \neq \mu_0$
- One-sided: $H_0\colon \mu = \mu_0$ vs $H_1\colon \mu > \mu_0$ (or $H_1\colon \mu < \mu_0$)

**z-test** (known $\sigma$): the test statistic is

$$
Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} \sim N(0,1).
$$

**t-test** (unknown $\sigma$, estimated by $S$): the test statistic is

$$
T = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n-1}.
$$

## Code

```python
import math
from scipy.stats import t as tdist, norm

def test_mean_one_sample(xbar, n, mu0=0.0, sd=None, known_sigma=None,
                         alt="two-sided", alpha=0.05):
    """
    If known_sigma is given -> z-test, else t-test using sample sd.
    Returns (stat, pvalue, reject_bool, label).
    """
    if known_sigma is not None:
        se = known_sigma / math.sqrt(n)
        z = (xbar - mu0) / se
        if alt == "two-sided":
            p = 2 * min(norm.cdf(z), 1 - norm.cdf(z))
        elif alt == "less":
            p = norm.cdf(z)
        else:
            p = 1 - norm.cdf(z)
        return z, p, (p < alpha), "z-test"

    if sd is None:
        raise ValueError("Provide sd for t-test or known_sigma for z-test.")
    se = sd / math.sqrt(n)
    df = n - 1
    t = (xbar - mu0) / se
    if alt == "two-sided":
        p = 2 * min(tdist.cdf(t, df), 1 - tdist.cdf(t, df))
    elif alt == "less":
        p = tdist.cdf(t, df)
    else:
        p = 1 - tdist.cdf(t, df)
    return t, p, (p < alpha), f"t-test (df={df})"
```

### Example

```python
stat, p, reject, label = test_mean_one_sample(
    xbar=3.2, n=25, mu0=3.0, sd=1.1, alt="greater"
)
print(label, "stat:", stat, "p:", p, "reject:", reject)
```

### Interpretation

In this example, we test $H_0\colon \mu = 3.0$ against $H_1\colon \mu > 3.0$ with $\bar{x} = 3.2$, $s = 1.1$, and $n = 25$. The test statistic is

$$
T = \frac{3.2 - 3.0}{1.1/\sqrt{25}} = \frac{0.2}{0.22} \approx 0.909.
$$

With 24 degrees of freedom, the one-sided p-value is approximately 0.186. Since this exceeds the typical $\alpha = 0.05$, we fail to reject $H_0$.

## Exercises

**Exercise 1.** A sample of $n = 36$ observations has $\bar{x} = 52$ and the population standard deviation is known to be $\sigma = 6$. Test $H_0\colon \mu = 50$ vs $H_1\colon \mu \neq 50$ at $\alpha = 0.05$.

??? success "Solution to Exercise 1"

    The z-test statistic is

    $$
    Z = \frac{52 - 50}{6/\sqrt{36}} = \frac{2}{1} = 2.0.
    $$

    The two-sided p-value is $2\,P(Z \geq 2.0) = 2(0.0228) = 0.0456$. Since $0.0456 < 0.05$, we reject $H_0$. There is significant evidence that $\mu \neq 50$. $\square$

---

**Exercise 2.** With $n = 10$, $\bar{x} = 15.3$, and $s = 2.5$, test $H_0\colon \mu = 14$ vs $H_1\colon \mu > 14$ at $\alpha = 0.01$.

??? success "Solution to Exercise 2"

    The t-test statistic is

    $$
    T = \frac{15.3 - 14}{2.5/\sqrt{10}} = \frac{1.3}{0.7906} \approx 1.644.
    $$

    With $\text{df} = 9$, the one-sided p-value is $P(T_9 \geq 1.644) \approx 0.068$. Since $0.068 > 0.01$, we fail to reject $H_0$ at the 1% level. $\square$

---

**Exercise 3.** Explain why the t-test is used instead of the z-test when $\sigma$ is unknown. What happens to the t-distribution as $n \to \infty$?

??? success "Solution to Exercise 3"

    When $\sigma$ is unknown, we estimate it with $S$. The ratio $(\bar{X}-\mu_0)/(S/\sqrt{n})$ has heavier tails than the standard normal because $S$ is itself a random variable. The $t_{n-1}$ distribution accounts for this extra uncertainty. As $n \to \infty$, by the law of large numbers $S \to \sigma$ almost surely, so $S/\sqrt{n}$ behaves like $\sigma/\sqrt{n}$, and $t_{n-1} \to N(0,1)$. Formally, $t_\nu \xrightarrow{d} N(0,1)$ as $\nu \to \infty$. $\square$

---

**Exercise 4.** Derive the rejection region for the one-sided t-test $H_0\colon \mu = \mu_0$ vs $H_1\colon \mu > \mu_0$ at significance level $\alpha$.

??? success "Solution to Exercise 4"

    Under $H_0$, $T = (\bar{X} - \mu_0)/(S/\sqrt{n}) \sim t_{n-1}$. We reject $H_0$ in favor of $H_1\colon \mu > \mu_0$ when $T$ is large. The rejection region is

    $$
    T > t_{\alpha,\,n-1},
    $$

    where $t_{\alpha,\,n-1}$ is the $(1-\alpha)$-quantile of the $t_{n-1}$ distribution, i.e., $P(T_{n-1} > t_{\alpha,\,n-1}) = \alpha$. Equivalently, we reject when the p-value $P(T_{n-1} \geq t_{\text{obs}}) < \alpha$. $\square$

---

**Exercise 5.** A manufacturer claims that the mean tensile strength of a steel rod is at least 5000 psi. A sample of $n = 20$ rods gives $\bar{x} = 4917$ and $s = 200$. Test at $\alpha = 0.05$ whether the claim is supported.

??? success "Solution to Exercise 5"

    We test $H_0\colon \mu \geq 5000$ vs $H_1\colon \mu < 5000$. The test statistic is

    $$
    T = \frac{4917 - 5000}{200/\sqrt{20}} = \frac{-83}{44.72} \approx -1.856.
    $$

    With $\text{df} = 19$, the one-sided p-value is $P(T_{19} \leq -1.856) \approx 0.039$. Since $0.039 < 0.05$, we reject $H_0$. There is significant evidence that the mean tensile strength is less than 5000 psi, contradicting the manufacturer's claim. $\square$
