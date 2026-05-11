# Two-Sample Proportion Test

## Overview

The two-sample proportion test compares proportions from two independent populations. It is widely used in A/B testing, clinical trials, and social science research to determine whether a treatment or intervention changes the rate of a binary outcome. The test uses a z-statistic based on the normal approximation to the binomial distribution.

## Test Formulation

**Hypotheses:**

- Two-sided: $H_0\colon p_1 - p_2 = \delta_0$ vs $H_1\colon p_1 - p_2 \neq \delta_0$

When $\delta_0 = 0$ (testing equality), a **pooled** standard error is used:

$$
\hat{p}_{\text{pool}} = \frac{k_1 + k_2}{n_1 + n_2}, \qquad SE = \sqrt{\hat{p}_{\text{pool}}(1-\hat{p}_{\text{pool}})\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}.
$$

When $\delta_0 \neq 0$, the **Wald** standard error is used:

$$
SE = \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}.
$$

The test statistic is

$$
Z = \frac{(\hat{p}_1 - \hat{p}_2) - \delta_0}{SE} \;\dot\sim\; N(0,1).
$$

## Code

```python
import math
from scipy.stats import norm

def test_diff_two_props(k1, n1, k2, n2, delta0=0.0,
                        method="pooled", alt="two-sided", alpha=0.05):
    """
    H0: p1 - p2 = delta0.
    When delta0=0 and method='pooled', uses pooled SE.
    Returns (z, p, reject, label).
    """
    p1, p2 = k1 / n1, k2 / n2
    d_hat = p1 - p2
    if delta0 == 0.0 and method == "pooled":
        p_pool = (k1 + k2) / (n1 + n2)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        label = "pooled z-test"
    else:
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        label = "wald z-test"

    z = (d_hat - delta0) / se
    if alt == "two-sided":
        p = 2 * min(norm.cdf(z), 1 - norm.cdf(z))
    elif alt == "less":
        p = norm.cdf(z)
    else:
        p = 1 - norm.cdf(z)
    return z, p, (p < alpha), label
```

### Example

```python
z, p, reject, label = test_diff_two_props(
    k1=30, n1=80, k2=18, n2=60, delta0=0.0, method="pooled"
)
print(label, "z:", z, "p:", p, "reject:", reject)
```

### Interpretation

We test $H_0\colon p_1 = p_2$ with $\hat{p}_1 = 30/80 = 0.375$ and $\hat{p}_2 = 18/60 = 0.300$. The pooled proportion is $\hat{p}_{\text{pool}} = 48/140 \approx 0.343$. The test statistic is

$$
Z = \frac{0.375 - 0.300}{\sqrt{0.343 \times 0.657 \times (1/80 + 1/60)}} = \frac{0.075}{\sqrt{0.343 \times 0.657 \times 0.0292}} \approx \frac{0.075}{0.0805} \approx 0.932.
$$

The two-sided p-value is approximately 0.35, so we fail to reject $H_0$.

## Exercises

**Exercise 1.** In a clinical trial, 45 out of 200 patients receiving a drug recover, while 30 out of 200 receiving a placebo recover. Test $H_0\colon p_1 = p_2$ vs $H_1\colon p_1 > p_2$ at $\alpha = 0.05$.

??? success "Solution to Exercise 1"

    $\hat{p}_1 = 0.225$, $\hat{p}_2 = 0.150$, $\hat{p}_{\text{pool}} = 75/400 = 0.1875$.

    $$
    SE = \sqrt{0.1875 \times 0.8125 \times (1/200 + 1/200)} = \sqrt{0.1875 \times 0.8125 \times 0.01} = \sqrt{0.001523} \approx 0.03903.
    $$

    $$
    Z = \frac{0.225 - 0.150}{0.03903} = \frac{0.075}{0.03903} \approx 1.921.
    $$

    The one-sided p-value is $P(Z \geq 1.921) \approx 0.0274$. Since $0.0274 < 0.05$, we reject $H_0$. The drug has a significantly higher recovery rate. $\square$

---

**Exercise 2.** Explain when the pooled standard error is used versus the Wald standard error.

??? success "Solution to Exercise 2"

    The **pooled SE** is used when $H_0$ specifies $p_1 = p_2$ (i.e., $\delta_0 = 0$). Under this null, the best estimate of the common proportion is the pooled proportion $\hat{p}_{\text{pool}}$, and using it in the SE gives a more accurate test.

    The **Wald SE** is used when $\delta_0 \neq 0$ (testing whether the difference equals some nonzero value). In this case, there is no common proportion to estimate, so each group's proportion is used separately. The Wald SE is also used for constructing confidence intervals for $p_1 - p_2$, regardless of the null hypothesis value. $\square$

---

**Exercise 3.** An A/B test on a website shows 120 conversions out of 5000 visitors (variant A) and 95 conversions out of 5000 visitors (variant B). Test for a difference at $\alpha = 0.05$ and compute a 95% confidence interval for $p_A - p_B$.

??? success "Solution to Exercise 3"

    $\hat{p}_A = 0.024$, $\hat{p}_B = 0.019$, $\hat{p}_{\text{pool}} = 215/10000 = 0.0215$.

    **Test:**

    $$
    SE = \sqrt{0.0215 \times 0.9785 \times 2/5000} = \sqrt{0.0215 \times 0.9785 \times 0.0004} \approx 0.002898.
    $$

    $$
    Z = \frac{0.024 - 0.019}{0.002898} \approx 1.725.
    $$

    Two-sided p-value: $2 \times P(Z \geq 1.725) \approx 0.0845$. Fail to reject at $\alpha = 0.05$.

    **95% CI** (using Wald SE):

    $$
    SE_{\text{Wald}} = \sqrt{\frac{0.024 \times 0.976}{5000} + \frac{0.019 \times 0.981}{5000}} \approx \sqrt{0.000004685 + 0.000003728} \approx 0.002901.
    $$

    $$
    0.005 \pm 1.96 \times 0.002901 = 0.005 \pm 0.00569 = (-0.00069,\; 0.01069).
    $$

    The CI contains 0, consistent with the test result. $\square$

---

**Exercise 4.** Derive the pooled z-test statistic from the likelihood ratio test for $H_0\colon p_1 = p_2$.

??? success "Solution to Exercise 4"

    Under $H_0\colon p_1 = p_2 = p$, the MLE is $\hat{p} = (k_1+k_2)/(n_1+n_2)$. Under the unrestricted model, the MLEs are $\hat{p}_1 = k_1/n_1$ and $\hat{p}_2 = k_2/n_2$. The log-likelihood ratio is

    $$
    \Lambda = 2\left[\ell(\hat{p}_1, \hat{p}_2) - \ell(\hat{p}, \hat{p})\right].
    $$

    By asymptotic theory, $\Lambda \xrightarrow{d} \chi^2_1$ under $H_0$. The Rao score test (equivalent to first order) gives the test statistic

    $$
    Z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})(1/n_1 + 1/n_2)}},
    $$

    and $Z^2 \approx \Lambda$. This is exactly the pooled z-statistic, confirming that the pooled test arises naturally from the score/LRT framework. $\square$

---

**Exercise 5.** Show that the normal approximation to the binomial requires both $n_1\hat{p}_{\text{pool}} \geq 5$ and $n_1(1-\hat{p}_{\text{pool}}) \geq 5$ (and similarly for $n_2$). What alternative can be used when these conditions fail?

??? success "Solution to Exercise 5"

    The z-test relies on the CLT approximation $\hat{p}_i \approx N(p_i, p_i(1-p_i)/n_i)$. This approximation is poor when $p$ is near 0 or 1 (the binomial is highly skewed) or when $n$ is small. The rule of thumb $np \geq 5$ and $n(1-p) \geq 5$ ensures the binomial is sufficiently symmetric for the normal approximation. Under the pooled test, we check using $\hat{p}_{\text{pool}}$.

    When these conditions fail, alternatives include:

    - **Fisher's exact test**: computes the exact p-value using the hypergeometric distribution under $H_0$. No asymptotic approximation is needed.
    - **Barnard's exact test**: an unconditional exact test that can be more powerful than Fisher's test.
    - **Bayesian methods**: use Beta-Binomial models with posterior inference. $\square$
