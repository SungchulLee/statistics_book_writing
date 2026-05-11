# Hypothesis Testing Demonstrations

## Overview

Hypothesis testing is a formal framework for making decisions about population parameters based on sample data. The procedure sets up a null hypothesis $H_0$ (a default claim) against an alternative hypothesis $H_1$, computes a test statistic, and uses its sampling distribution to obtain a p-value. This page demonstrates one-sample, two-sample, and paired tests for means and proportions, along with power analysis and the duality between confidence intervals and hypothesis tests.

## One-Sample Tests for the Mean

When the population standard deviation $\sigma$ is unknown, we use the **t-test**. Given a random sample $X_1, \dots, X_n$ with sample mean $\bar{X}$ and sample standard deviation $S$, the test statistic for $H_0\colon \mu = \mu_0$ is

$$
T = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n-1}.
$$

### Code

```python
import numpy as np
from scipy import stats

# Factory claims mean weight is 500g
data = np.array([498, 495, 502, 497, 501, 499, 496, 503, 494, 500,
                 497, 502, 496, 501, 498, 499, 495, 503, 497, 500])
mu_0 = 500
alpha = 0.05

# t-test (sigma unknown)
t_stat, p_value = stats.ttest_1samp(data, mu_0)
print(f"t = {t_stat:.4f}, p-value = {p_value:.4f}")
print(f"Decision: {'Reject H0' if p_value < alpha else 'Fail to reject H0'}")
```

For a **one-sided** test $H_1\colon \mu < \mu_0$, divide the two-sided p-value by 2 when the test statistic falls in the direction of the alternative:

```python
p_one_sided = p_value / 2 if t_stat < 0 else 1 - p_value / 2
```

### Manual Computation

```python
n = len(data)
xbar = data.mean()
s = data.std(ddof=1)
t_manual = (xbar - mu_0) / (s / np.sqrt(n))
p_manual = 2 * stats.t.cdf(-abs(t_manual), df=n - 1)
```

## One-Sample Test for a Proportion

For testing $H_0\colon p = p_0$ with $\hat{p} = x/n$, the Wald test statistic is

$$
Z = \frac{\hat{p} - p_0}{\sqrt{p_0(1 - p_0)/n}} \;\dot\sim\; N(0,1).
$$

```python
from statsmodels.stats.proportion import proportions_ztest

x, n = 12, 200  # 12 defectives in 200
p_0 = 0.05
p_hat = x / n

z_stat = (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / n)
p_value = 2 * stats.norm.sf(abs(z_stat))

# Or using statsmodels
z_sm, p_sm = proportions_ztest(x, n, value=p_0)
```

## Two-Sample t-Tests

Given independent samples from two populations, the null hypothesis is $H_0\colon \mu_1 = \mu_2$. **Welch's t-test** (unequal variances) uses

$$
T = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}
$$

with degrees of freedom computed by the Welch--Satterthwaite approximation.

```python
drug_a = np.array([5.2, 4.8, 6.1, 5.5, 4.9, 5.7, 5.3, 6.0, 5.1, 5.4])
drug_b = np.array([4.1, 3.8, 4.5, 4.2, 3.9, 4.6, 4.0, 4.3, 3.7, 4.4])

# Welch's t-test (default: unequal variances)
t_welch, p_welch = stats.ttest_ind(drug_a, drug_b, equal_var=False)

# Pooled t-test (equal variances assumed)
t_pooled, p_pooled = stats.ttest_ind(drug_a, drug_b, equal_var=True)
```

## Paired t-Test

When observations come in natural pairs (e.g., before/after measurements), we compute the differences $D_i = X_i^{(\text{after})} - X_i^{(\text{before})}$ and test $H_0\colon \mu_D = 0$:

$$
T = \frac{\bar{D}}{S_D / \sqrt{n}} \sim t_{n-1}.
$$

```python
before = np.array([145, 150, 138, 155, 142, 148, 136, 152, 140, 146])
after  = np.array([138, 142, 130, 148, 135, 140, 132, 145, 134, 139])

t_stat, p_value = stats.ttest_rel(after, before)

# Equivalent to one-sample t-test on the differences
diff = after - before
t_stat2, p_value2 = stats.ttest_1samp(diff, 0)
```

## Power Analysis

The **power** of a test is the probability of correctly rejecting $H_0$ when $H_1$ is true:

$$
\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_1 \text{ true}).
$$

```python
from statsmodels.stats.power import TTestPower, TTestIndPower

# One-sample: detect a 5-point difference with sigma=15
analysis = TTestPower()
effect_size = 5 / 15  # Cohen's d
n_needed = analysis.solve_power(effect_size=effect_size, alpha=0.05,
                                 power=0.80, alternative='two-sided')

# Two-sample: medium effect size
analysis2 = TTestIndPower()
n_each = analysis2.solve_power(effect_size=0.5, alpha=0.05, power=0.80,
                                ratio=1.0, alternative='two-sided')
```

## CI--Test Duality

A value $\mu_0$ lies inside the $100(1-\alpha)\%$ confidence interval if and only if the two-sided test at level $\alpha$ fails to reject $H_0\colon \mu = \mu_0$.

```python
data = np.array([52, 48, 55, 50, 47, 53, 49, 51, 54, 46])
n = len(data)
xbar = data.mean()
s = data.std(ddof=1)
alpha = 0.05

t_c = stats.t.ppf(1 - alpha / 2, df=n - 1)
me = t_c * s / np.sqrt(n)
ci = (xbar - me, xbar + me)

# Test various mu_0 values
for mu0 in [48, 49, 50, 51, 52, 53]:
    t_stat, p_val = stats.ttest_1samp(data, mu0)
    in_ci = ci[0] <= mu0 <= ci[1]
    reject = p_val < alpha
    # reject <=> mu0 NOT in CI
```

### Interpretation

The duality principle states:

$$
\mu_0 \notin \left(\bar{X} \pm t_{\alpha/2,\,n-1}\,\frac{S}{\sqrt{n}}\right) \iff \text{reject } H_0\colon \mu = \mu_0 \text{ at level } \alpha.
$$

This provides a unified view: constructing a confidence interval is equivalent to inverting a family of hypothesis tests.

## Exercises

**Exercise 1.** A sample of $n=16$ light bulbs has $\bar{x}=1020$ hours and $s=80$ hours. Test $H_0\colon \mu=1000$ vs $H_1\colon \mu>1000$ at $\alpha=0.05$. State the test statistic, p-value, and decision.

??? success "Solution to Exercise 1"

    The test statistic is

    $$
    T = \frac{1020 - 1000}{80/\sqrt{16}} = \frac{20}{20} = 1.0.
    $$

    Under $H_0$, $T \sim t_{15}$. The one-sided p-value is

    $$
    p = P(T_{15} \geq 1.0) \approx 0.1667.
    $$

    Since $p = 0.1667 > 0.05$, we fail to reject $H_0$. There is insufficient evidence that the mean lifetime exceeds 1000 hours. $\square$

---

**Exercise 2.** In a poll of 400 voters, 228 support a ballot measure. Test whether the true proportion exceeds 0.50 at the 1% significance level.

??? success "Solution to Exercise 2"

    We have $\hat{p} = 228/400 = 0.57$ and test $H_0\colon p = 0.50$ vs $H_1\colon p > 0.50$. The test statistic is

    $$
    Z = \frac{0.57 - 0.50}{\sqrt{0.50 \times 0.50 / 400}} = \frac{0.07}{0.025} = 2.80.
    $$

    The one-sided p-value is $P(Z \geq 2.80) \approx 0.0026 < 0.01$, so we reject $H_0$. There is strong evidence that more than half the voters support the measure. $\square$

---

**Exercise 3.** Two independent groups yield $\bar{x}_1=75$, $s_1=10$, $n_1=20$ and $\bar{x}_2=70$, $s_2=12$, $n_2=25$. Conduct Welch's t-test for $H_0\colon \mu_1 = \mu_2$ at $\alpha=0.05$. Write out the formula for the Welch--Satterthwaite degrees of freedom.

??? success "Solution to Exercise 3"

    The test statistic is

    $$
    T = \frac{75 - 70}{\sqrt{10^2/20 + 12^2/25}} = \frac{5}{\sqrt{5 + 5.76}} = \frac{5}{\sqrt{10.76}} \approx \frac{5}{3.280} \approx 1.524.
    $$

    The Welch--Satterthwaite degrees of freedom are

    $$
    \nu = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}} = \frac{(5 + 5.76)^2}{\frac{25}{19} + \frac{33.18}{24}} \approx \frac{115.78}{1.316 + 1.382} \approx 42.9.
    $$

    Using $\nu \approx 42$, the two-sided p-value is approximately 0.135. Since $p > 0.05$, we fail to reject $H_0$. $\square$

---

**Exercise 4.** Prove the CI--test duality: show that for a two-sided one-sample t-test at level $\alpha$, rejecting $H_0\colon \mu = \mu_0$ is equivalent to $\mu_0$ falling outside the $100(1-\alpha)\%$ confidence interval.

??? success "Solution to Exercise 4"

    The test rejects $H_0$ when $|T| > t_{\alpha/2,\,n-1}$, i.e.,

    $$
    \left|\frac{\bar{X} - \mu_0}{S/\sqrt{n}}\right| > t_{\alpha/2,\,n-1}.
    $$

    This is equivalent to

    $$
    |\bar{X} - \mu_0| > t_{\alpha/2,\,n-1}\,\frac{S}{\sqrt{n}},
    $$

    which means $\mu_0 < \bar{X} - t_{\alpha/2,\,n-1}\,S/\sqrt{n}$ or $\mu_0 > \bar{X} + t_{\alpha/2,\,n-1}\,S/\sqrt{n}$. In other words,

    $$
    \mu_0 \notin \left(\bar{X} - t_{\alpha/2,\,n-1}\,\frac{S}{\sqrt{n}},\; \bar{X} + t_{\alpha/2,\,n-1}\,\frac{S}{\sqrt{n}}\right).
    $$

    The interval on the right is exactly the $100(1-\alpha)\%$ confidence interval for $\mu$. Therefore, the test rejects $H_0\colon \mu=\mu_0$ at level $\alpha$ if and only if $\mu_0$ is not contained in the confidence interval. $\square$

---

**Exercise 5.** Using the power formula for a one-sample z-test, show that the required sample size to achieve power $1-\beta$ at significance level $\alpha$ for detecting a shift of $\delta$ (with known $\sigma$) in a two-sided test is

$$
n = \left(\frac{(z_{\alpha/2} + z_\beta)\,\sigma}{\delta}\right)^2.
$$

??? success "Solution to Exercise 5"

    Under $H_0$, $Z = (\bar{X}-\mu_0)/(\sigma/\sqrt{n}) \sim N(0,1)$. We reject when $|Z|>z_{\alpha/2}$. Under $H_1\colon \mu = \mu_0 + \delta$, the statistic has distribution $Z \sim N(\delta\sqrt{n}/\sigma,\,1)$. The power (considering the right tail, which dominates for $\delta>0$) is

    $$
    1 - \beta = P\!\left(Z > z_{\alpha/2}\right) = P\!\left(N(0,1) > z_{\alpha/2} - \frac{\delta\sqrt{n}}{\sigma}\right).
    $$

    Setting this equal to $1-\beta$ gives

    $$
    z_{\alpha/2} - \frac{\delta\sqrt{n}}{\sigma} = -z_\beta,
    $$

    so

    $$
    \frac{\delta\sqrt{n}}{\sigma} = z_{\alpha/2} + z_\beta \implies \sqrt{n} = \frac{(z_{\alpha/2}+z_\beta)\,\sigma}{\delta}.
    $$

    Squaring both sides yields the desired formula. $\square$
