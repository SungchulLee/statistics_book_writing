# Height/Weight Hypothesis Tests

## Overview

This page walks through three foundational hypothesis-testing workflows using height and proportion data: a one-sample $t$-test for a population mean, a two-sample $z$-test (known variances) comparing two population means, and a two-proportion $z$-test for comparing rejection rates. Each test follows the same four-step framework: state the hypotheses, compute the test statistic, find the p-value, and make a decision.

## One-Sample t-Test

We test whether the average male height equals a claimed value $\mu_0$.

**Hypotheses:**

$$
H_0\colon \mu = \mu_0, \qquad H_1\colon \mu \neq \mu_0.
$$

**Test statistic:** Given a sample of size $n$ with mean $\bar{x}$ and standard deviation $s$,

$$
t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}},
$$

which follows a $t$-distribution with $n - 1$ degrees of freedom under $H_0$.

**Decision rule:** Reject $H_0$ if $|t| > t_{\alpha/2,\, n-1}$.

### Code

```python
import numpy as np
from scipy import stats

np.random.seed(42)
data = stats.norm.rvs(loc=170, scale=8, size=250)

mu0 = 172
n = len(data)
xbar = data.mean()
s = data.std(ddof=1)
se = s / np.sqrt(n)
df = n - 1

t_stat = (xbar - mu0) / se
p_val = 2 * stats.t.cdf(-abs(t_stat), df)
t_crit = stats.t.ppf(1 - 0.05 / 2, df)

print(f"x-bar = {xbar:.2f}, s = {s:.2f}, SE = {se:.2f}")
print(f"t = {t_stat:.4f}, p = {p_val:.4f}, t_crit = {t_crit:.4f}")
print("Reject H0" if abs(t_stat) > t_crit else "Fail to reject H0")
```

## Two-Sample z-Test (Known Variances)

When the population standard deviations $\sigma_x$ and $\sigma_y$ are known, we compare two means with

$$
z = \frac{\bar{x} - \bar{y} - D_0}{\sqrt{\dfrac{\sigma_x^2}{n_1} + \dfrac{\sigma_y^2}{n_2}}},
$$

where $D_0$ is the hypothesized difference (typically 0). Under $H_0\colon \mu_x - \mu_y = D_0$, $z \sim N(0,1)$.

### Code

```python
male = stats.norm.rvs(loc=170, scale=8, size=250)
female = stats.norm.rvs(loc=165, scale=7, size=250)

sigma_x, sigma_y = 8, 7
n1, n2 = len(male), len(female)
se = np.sqrt(sigma_x**2 / n1 + sigma_y**2 / n2)
z = (male.mean() - female.mean()) / se
p = 2 * stats.norm.cdf(-abs(z))

print(f"z = {z:.4f}, p = {p:.6f}")
```

When the population variances are unknown, the Welch $t$-test replaces the known $\sigma$ values with sample estimates and adjusts the degrees of freedom:

```python
t_w, p_w = stats.ttest_ind(male, female, equal_var=False)
print(f"Welch t = {t_w:.4f}, p = {p_w:.6f}")
```

## Two-Proportion z-Test

To test whether two population proportions differ, we use the pooled proportion under $H_0\colon p_1 = p_2$:

$$
\hat{p} = \frac{k_1 + k_2}{n_1 + n_2}, \qquad z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}.
$$

### Code

```python
k1, n1 = 59, 649    # women: 59 rejected out of 649
k2, n2 = 128, 2490   # men: 128 rejected out of 2490

p1, p2 = k1 / n1, k2 / n2
p_pool = (k1 + k2) / (n1 + n2)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
z = (p1 - p2) / se
p = 1 - stats.norm.cdf(z)  # one-sided: H1: p1 > p2

print(f"p_women = {p1:.4f}, p_men = {p2:.4f}")
print(f"z = {z:.4f}, p (one-sided) = {p:.4f}")
```

## Interpretation

- The **one-sample $t$-test** checks whether 250 male heights are consistent with a population mean of 172 cm. If the sample mean is noticeably lower (around 170), the $t$-statistic may fall in the rejection region.
- The **two-sample $z$-test** detects a roughly 5 cm gap between male and female average heights. With $n = 250$ per group and known variances, even moderate differences produce large $z$ values and tiny p-values.
- The **two-proportion test** examines whether women face a higher bank-loan rejection rate (9.1% vs. 5.1%). The one-sided test asks specifically if the female rate exceeds the male rate, which is appropriate given the research question about discrimination.

## Exercises

**Exercise 1.** Derive the standard error formula for the one-sample $t$-test. Why do we divide by $\sqrt{n}$ rather than $n$?

??? success "Solution to Exercise 1"

    The sample mean $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ has variance

    $$
    \text{Var}(\bar{X}) = \text{Var}\!\left(\frac{1}{n}\sum X_i\right) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}.
    $$

    The standard error is $\text{SE} = \sigma/\sqrt{n}$, estimated by $s/\sqrt{n}$. We divide by $\sqrt{n}$ because the variance of the mean scales as $1/n$, and the standard deviation (square root of variance) scales as $1/\sqrt{n}$. $\square$

---

**Exercise 2.** In the two-sample $z$-test, suppose the population variances are unknown but believed equal. Write down the pooled $t$-test statistic and explain how it differs from Welch's $t$-test.

??? success "Solution to Exercise 2"

    The pooled two-sample $t$-statistic is

    $$
    t = \frac{\bar{x} - \bar{y}}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}, \qquad s_p^2 = \frac{(n_1-1)s_x^2 + (n_2-1)s_y^2}{n_1 + n_2 - 2}.
    $$

    This uses a single pooled variance estimate $s_p^2$ and has $n_1 + n_2 - 2$ degrees of freedom. Welch's test does not assume equal variances; instead it uses separate variance estimates $s_x^2/n_1 + s_y^2/n_2$ and approximates the degrees of freedom via the Welch-Satterthwaite equation:

    $$
    \text{df} = \frac{\left(\frac{s_x^2}{n_1} + \frac{s_y^2}{n_2}\right)^2}{\frac{(s_x^2/n_1)^2}{n_1-1} + \frac{(s_y^2/n_2)^2}{n_2-1}}.
    $$

    Welch's test is more robust when the variances are unequal. $\square$

---

**Exercise 3.** For the two-proportion test, verify that the pooled proportion \$\hat{p} = (59 + 128)/(649 + 2490)\$ gives the correct value and compute the 95% confidence interval for $p_1 - p_2$.

??? success "Solution to Exercise 3"

    The pooled proportion is

    $$
    \hat{p} = \frac{59 + 128}{649 + 2490} = \frac{187}{3139} \approx 0.0596.
    $$

    For the confidence interval, use the unpooled standard error:

    $$
    SE = \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}} = \sqrt{\frac{0.0909 \times 0.9091}{649} + \frac{0.0514 \times 0.9486}{2490}} \approx 0.0121.
    $$

    The 95% CI for $p_1 - p_2$ is

    $$
    (0.0909 - 0.0514) \pm 1.96 \times 0.0121 = 0.0395 \pm 0.0237 = [0.0158,\; 0.0632].
    $$

    Since this interval does not contain 0, we conclude the difference is statistically significant. $\square$

---

**Exercise 4.** Explain the difference between a one-sided and two-sided test. Under what circumstances is a one-sided test appropriate in the bank discrimination example?

??? success "Solution to Exercise 4"

    A **two-sided** test has $H_1\colon p_1 \neq p_2$ and rejects for large values of $|z|$. A **one-sided** test has $H_1\colon p_1 > p_2$ (or $p_1 < p_2$) and rejects for large values of $z$ in one direction only.

    In the bank discrimination example, the research question is specifically whether women face a *higher* rejection rate. If there is no prior reason to test the opposite direction, a one-sided test ($H_1\colon p_{\text{women}} > p_{\text{men}}$) is appropriate. The one-sided p-value is half the two-sided p-value, making it easier to reject $H_0$ -- but this is only legitimate if the direction was specified before seeing the data. $\square$

---

**Exercise 5.** If the one-sample $t$-test for male heights yields $p = 0.04$, what does this mean in plain language? What does it *not* mean?

??? success "Solution to Exercise 5"

    A p-value of 0.04 means: if the true population mean were $\mu_0 = 172$ cm, there is a 4% probability of observing a sample mean at least as extreme as the one obtained. Since $0.04 < 0.05$, we reject $H_0$ at the 5% significance level.

    What it does **not** mean:

    - It does not mean there is a 4% probability that $H_0$ is true. The p-value is $P(\text{data} \mid H_0)$, not $P(H_0 \mid \text{data})$.
    - It does not measure the size of the effect. A small p-value can arise from a trivially small difference with a large sample.
    - It does not prove $H_1$ is true. Rejecting $H_0$ only says the data are unlikely under $H_0$, not that any specific alternative is correct.

    $\square$
