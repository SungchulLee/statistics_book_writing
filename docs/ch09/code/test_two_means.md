# Two-Sample Mean Test

## Overview

The two-sample t-test compares the means of two independent populations. The **pooled t-test** assumes equal variances and combines the two sample variances into a single estimate, while **Welch's t-test** allows unequal variances by adjusting the degrees of freedom. Welch's test is generally recommended as the default because it performs well even when variances are equal.

## Test Formulation

**Hypotheses:**

- Two-sided: $H_0\colon \mu_1 - \mu_2 = \delta_0$ vs $H_1\colon \mu_1 - \mu_2 \neq \delta_0$
- One-sided: $H_0\colon \mu_1 - \mu_2 = \delta_0$ vs $H_1\colon \mu_1 - \mu_2 > \delta_0$

### Welch's t-Test

$$
T = \frac{(\bar{X}_1 - \bar{X}_2) - \delta_0}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}
$$

with Welch--Satterthwaite degrees of freedom:

$$
\nu = \frac{\left(\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}\right)^2}{\frac{(S_1^2/n_1)^2}{n_1-1} + \frac{(S_2^2/n_2)^2}{n_2-1}}.
$$

### Pooled t-Test

When $\sigma_1^2 = \sigma_2^2 = \sigma^2$, pool the variances:

$$
S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}, \qquad T = \frac{(\bar{X}_1 - \bar{X}_2) - \delta_0}{S_p\sqrt{1/n_1 + 1/n_2}} \sim t_{n_1+n_2-2}.
$$

## Code

```python
import math
from scipy.stats import t as tdist

def test_diff_two_means(n1, m1, s1, n2, m2, s2, method="welch",
                        delta0=0.0, alt="two-sided", alpha=0.05):
    """
    H0: mu1 - mu2 = delta0.
    method='welch' (default) or 'pooled'.
    Returns (t, df, p, reject).
    """
    diff_hat = m1 - m2
    if method == "welch":
        se = math.sqrt(s1**2 / n1 + s2**2 / n2)
        num = (s1**2 / n1 + s2**2 / n2) ** 2
        den = (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)
        df = num / den
    else:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df
        se = math.sqrt(sp2 * (1 / n1 + 1 / n2))

    t = (diff_hat - delta0) / se
    if alt == "two-sided":
        p = 2 * min(tdist.cdf(t, df), 1 - tdist.cdf(t, df))
    elif alt == "less":
        p = tdist.cdf(t, df)
    else:
        p = 1 - tdist.cdf(t, df)
    return t, df, p, (p < alpha)
```

### Example

```python
t, df, p, reject = test_diff_two_means(
    n1=12, m1=0.0, s1=1.0, n2=10, m2=0.5, s2=1.5,
    method="welch", alt="greater"
)
print("t:", t, "df:", df, "p:", p, "reject:", reject)
```

### Interpretation

We test $H_0\colon \mu_1 - \mu_2 = 0$ vs $H_1\colon \mu_1 - \mu_2 > 0$ with $\bar{x}_1 = 0.0$, $s_1 = 1.0$, $n_1 = 12$ and $\bar{x}_2 = 0.5$, $s_2 = 1.5$, $n_2 = 10$. Since $\bar{x}_1 - \bar{x}_2 = -0.5 < 0$, the test statistic is negative. For a right-tailed test, the p-value will be close to 1, and we will fail to reject $H_0$.

## Exercises

**Exercise 1.** Group A ($n_1=15$, $\bar{x}_1=82$, $s_1=6$) and Group B ($n_2=12$, $\bar{x}_2=76$, $s_2=8$). Conduct Welch's t-test for $H_0\colon \mu_1 = \mu_2$ at $\alpha=0.05$.

??? success "Solution to Exercise 1"

    The standard error is

    $$
    SE = \sqrt{\frac{6^2}{15} + \frac{8^2}{12}} = \sqrt{2.4 + 5.333} = \sqrt{7.733} \approx 2.781.
    $$

    The test statistic is

    $$
    T = \frac{82 - 76}{2.781} = \frac{6}{2.781} \approx 2.157.
    $$

    The Welch--Satterthwaite degrees of freedom:

    $$
    \nu = \frac{7.733^2}{\frac{2.4^2}{14} + \frac{5.333^2}{11}} = \frac{59.80}{0.411 + 2.587} = \frac{59.80}{2.998} \approx 19.95.
    $$

    With $\nu \approx 20$, the two-sided p-value is approximately 0.044. Since $0.044 < 0.05$, we reject $H_0$. The means differ significantly. $\square$

---

**Exercise 2.** Show that when $n_1 = n_2 = n$ and $s_1 = s_2 = s$, the pooled and Welch t-tests give identical test statistics and degrees of freedom.

??? success "Solution to Exercise 2"

    **Pooled:** $S_p^2 = \frac{(n-1)s^2 + (n-1)s^2}{2n-2} = s^2$. The SE is $s\sqrt{2/n}$ and $\text{df} = 2n-2$.

    **Welch:** $SE = \sqrt{s^2/n + s^2/n} = s\sqrt{2/n}$. The test statistics are identical. For the degrees of freedom:

    $$
    \nu = \frac{(s^2/n + s^2/n)^2}{\frac{(s^2/n)^2}{n-1} + \frac{(s^2/n)^2}{n-1}} = \frac{(2s^2/n)^2}{2(s^2/n)^2/(n-1)} = \frac{4s^4/n^2}{2s^4/(n^2(n-1))} = 2(n-1).
    $$

    This equals $2n-2$, the pooled degrees of freedom. $\square$

---

**Exercise 3.** Under what conditions is the pooled t-test preferred over Welch's test? When can it lead to incorrect conclusions?

??? success "Solution to Exercise 3"

    The pooled t-test is preferred when there is strong prior evidence that $\sigma_1^2 = \sigma_2^2$ (e.g., from an F-test or domain knowledge). Under equal variances, the pooled test has slightly more degrees of freedom ($n_1+n_2-2$ vs. $\nu_{\text{Welch}}$), giving it marginally more power.

    However, when variances are unequal, the pooled test can have an inflated Type I error rate (liberal test) or a deflated rate (conservative), depending on the relationship between sample sizes and variances. Specifically, if the group with the smaller sample has the larger variance, the pooled test rejects too often. Welch's test is robust to unequal variances with only minimal power loss under equality, making it the safer default. $\square$

---

**Exercise 4.** Two machines produce bolts. Machine 1: $n_1=20$, $\bar{x}_1=10.02$ mm, $s_1=0.05$. Machine 2: $n_2=25$, $\bar{x}_2=10.00$ mm, $s_2=0.04$. Test $H_0\colon \mu_1 - \mu_2 = 0$ vs $H_1\colon \mu_1 - \mu_2 \neq 0$ using the pooled t-test at $\alpha = 0.01$.

??? success "Solution to Exercise 4"

    The pooled variance is

    $$
    S_p^2 = \frac{19(0.05)^2 + 24(0.04)^2}{43} = \frac{19(0.0025) + 24(0.0016)}{43} = \frac{0.0475 + 0.0384}{43} = \frac{0.0859}{43} \approx 0.001998.
    $$

    The SE is

    $$
    SE = \sqrt{0.001998 \times (1/20 + 1/25)} = \sqrt{0.001998 \times 0.09} = \sqrt{0.0001798} \approx 0.01341.
    $$

    The test statistic is

    $$
    T = \frac{10.02 - 10.00}{0.01341} = \frac{0.02}{0.01341} \approx 1.491.
    $$

    With $\text{df} = 43$, the two-sided p-value is approximately 0.143. Since $0.143 > 0.01$, we fail to reject $H_0$. $\square$

---

**Exercise 5.** Derive the pooled variance estimator $S_p^2$ as the maximum likelihood estimator of $\sigma^2$ under the assumption $\sigma_1^2 = \sigma_2^2 = \sigma^2$ (up to a bias correction).

??? success "Solution to Exercise 5"

    Under the equal-variance model, $X_{1j} \overset{\text{iid}}{\sim} N(\mu_1, \sigma^2)$ for $j=1,\dots,n_1$ and $X_{2j} \overset{\text{iid}}{\sim} N(\mu_2, \sigma^2)$ for $j=1,\dots,n_2$, independently. The log-likelihood is

    $$
    \ell(\mu_1, \mu_2, \sigma^2) = -\frac{n_1+n_2}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\left[\sum_{j=1}^{n_1}(X_{1j}-\mu_1)^2 + \sum_{j=1}^{n_2}(X_{2j}-\mu_2)^2\right].
    $$

    Maximizing over $\mu_1, \mu_2$ gives $\hat{\mu}_i = \bar{X}_i$. Substituting and maximizing over $\sigma^2$:

    $$
    \hat{\sigma}^2_{\text{MLE}} = \frac{\sum(X_{1j}-\bar{X}_1)^2 + \sum(X_{2j}-\bar{X}_2)^2}{n_1+n_2}.
    $$

    Applying the bias correction (replacing $n_1+n_2$ with $n_1+n_2-2$) gives the unbiased estimator:

    $$
    S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}. \quad \square
    $$
