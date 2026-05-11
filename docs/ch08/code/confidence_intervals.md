# Confidence Interval Demonstrations

## Overview

Confidence intervals provide a range of plausible values for an unknown population parameter, constructed so that the interval captures the true parameter at a specified confidence level across repeated sampling. This page demonstrates how to build confidence intervals for the population mean $\mu$, a proportion $p$, the variance $\sigma^2$, and the difference of two means $\mu_1 - \mu_2$, along with coverage simulations and sample-size calculations.

## CI for the Population Mean

### z-Interval (Known Variance)

When the population standard deviation $\sigma$ is known, the $(1-\alpha)100\%$ confidence interval for $\mu$ is

$$
\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

where $z_{\alpha/2}$ is the upper $\alpha/2$ quantile of the standard normal distribution.

### t-Interval (Unknown Variance)

When $\sigma$ is unknown and replaced by the sample standard deviation $s$, the interval uses the $t$-distribution with $n - 1$ degrees of freedom:

$$
\bar{X} \pm t_{\alpha/2,\, n-1} \cdot \frac{s}{\sqrt{n}}
$$

### Python Code

```python
import numpy as np
from scipy import stats

data = np.array([120, 125, 118, 130, 122, 128, 115, 135, 121, 126,
                 119, 132, 124, 117, 129, 123, 131, 116, 127, 120])
n = len(data)
xbar = data.mean()
s = data.std(ddof=1)
alpha = 0.05

# z-interval (sigma known)
sigma_known = 6
z_crit = stats.norm.ppf(1 - alpha / 2)
me_z = z_crit * sigma_known / np.sqrt(n)
print(f"z-interval: ({xbar - me_z:.2f}, {xbar + me_z:.2f})")

# t-interval (sigma unknown)
t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
me_t = t_crit * s / np.sqrt(n)
print(f"t-interval: ({xbar - me_t:.2f}, {xbar + me_t:.2f})")

# Using scipy directly
ci = stats.t.interval(1 - alpha, df=n - 1, loc=xbar, scale=s / np.sqrt(n))
print(f"scipy t.interval: ({ci[0]:.2f}, {ci[1]:.2f})")
```

## CI for a Proportion

For a proportion $\hat{p} = x / n$ the **Wald interval** is

$$
\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}
$$

The **Wilson score interval** adjusts the center and width to improve coverage:

$$
\frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}
\;\pm\;
\frac{z}{1 + \frac{z^2}{n}}
\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}
$$

The **Agresti--Coull interval** adds $z^2/2$ pseudo-successes and pseudo-failures, then applies the Wald formula to the adjusted counts $\tilde{n} = n + z^2$ and $\tilde{p} = (x + z^2/2) / \tilde{n}$.

### Python Code

```python
x, n = 84, 200
p_hat = x / n
z = stats.norm.ppf(1 - alpha / 2)

# Wald interval
me_wald = z * np.sqrt(p_hat * (1 - p_hat) / n)
print(f"Wald: ({p_hat - me_wald:.4f}, {p_hat + me_wald:.4f})")

# Wilson interval
denom = 1 + z**2 / n
center = (p_hat + z**2 / (2 * n)) / denom
me_wilson = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
print(f"Wilson: ({center - me_wilson:.4f}, {center + me_wilson:.4f})")

# Agresti-Coull interval
n_tilde = n + z**2
p_tilde = (x + z**2 / 2) / n_tilde
me_ac = z * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
print(f"Agresti-Coull: ({p_tilde - me_ac:.4f}, {p_tilde + me_ac:.4f})")
```

## CI for the Variance

Under the assumption that the data come from a normal population, the pivotal quantity $(n-1)S^2 / \sigma^2$ follows a chi-squared distribution with $n - 1$ degrees of freedom. The resulting $(1-\alpha)100\%$ confidence interval for $\sigma^2$ is

$$
\left(\frac{(n-1)s^2}{\chi^2_{1-\alpha/2,\,n-1}},\;\;
      \frac{(n-1)s^2}{\chi^2_{\alpha/2,\,n-1}}\right)
$$

### Python Code

```python
data = np.array([120, 125, 118, 130, 122, 128, 115, 135, 121, 126])
n = len(data)
s2 = data.var(ddof=1)

chi2_lower = stats.chi2.ppf(alpha / 2, df=n - 1)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=n - 1)

ci_var = ((n - 1) * s2 / chi2_upper, (n - 1) * s2 / chi2_lower)
ci_sd = (np.sqrt(ci_var[0]), np.sqrt(ci_var[1]))

print(f"95% CI for sigma^2: ({ci_var[0]:.2f}, {ci_var[1]:.2f})")
print(f"95% CI for sigma:   ({ci_sd[0]:.2f}, {ci_sd[1]:.2f})")
```

## Two-Sample CI for the Difference of Means

### Welch's t-Interval (Unequal Variances)

For independent samples of sizes $n_1$ and $n_2$, the confidence interval for $\mu_1 - \mu_2$ is

$$
(\bar{X}_1 - \bar{X}_2) \;\pm\; t_{\alpha/2,\,\nu} \cdot \sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}
$$

where the Satterthwaite degrees of freedom are

$$
\nu = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}
$$

### Pooled t-Interval (Equal Variances)

When equal variances are assumed, the pooled variance is $s_p^2 = [(n_1-1)s_1^2 + (n_2-1)s_2^2] / (n_1+n_2-2)$ and the interval becomes

$$
(\bar{X}_1 - \bar{X}_2) \;\pm\; t_{\alpha/2,\,n_1+n_2-2} \cdot s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}
$$

### Python Code

```python
group_a = np.array([12, 15, 11, 14, 13, 16, 10, 15, 12, 14])
group_b = np.array([18, 20, 17, 19, 16, 21, 15, 20, 18, 17])

n1, n2 = len(group_a), len(group_b)
x1, x2 = group_a.mean(), group_b.mean()
s1, s2_val = group_a.std(ddof=1), group_b.std(ddof=1)

# Welch's t-interval
se = np.sqrt(s1**2 / n1 + s2_val**2 / n2)
df_welch = (s1**2 / n1 + s2_val**2 / n2)**2 / (
    (s1**2 / n1)**2 / (n1 - 1) + (s2_val**2 / n2)**2 / (n2 - 1)
)
t_crit = stats.t.ppf(1 - alpha / 2, df=df_welch)
diff = x1 - x2
ci_welch = (diff - t_crit * se, diff + t_crit * se)
print(f"Welch CI: ({ci_welch[0]:.2f}, {ci_welch[1]:.2f})")
```

## Coverage Simulation

A coverage simulation draws many samples, builds a CI from each, and records the fraction that contain the true parameter. The empirical coverage should be close to the nominal confidence level.

```python
np.random.seed(42)
mu_true, sigma_true = 100, 15
n_sim = 10_000

for n in [5, 10, 30, 100]:
    z_covers = 0
    t_covers = 0
    for _ in range(n_sim):
        sample = np.random.normal(mu_true, sigma_true, n)
        xbar = sample.mean()
        s = sample.std(ddof=1)

        # z-interval using s (common but incorrect)
        me_z = 1.96 * s / np.sqrt(n)
        if xbar - me_z <= mu_true <= xbar + me_z:
            z_covers += 1

        # t-interval (correct)
        t_c = stats.t.ppf(0.975, df=n - 1)
        me_t = t_c * s / np.sqrt(n)
        if xbar - me_t <= mu_true <= xbar + me_t:
            t_covers += 1

    print(f"n={n:>3}: z-coverage={z_covers/n_sim:.3f}  t-coverage={t_covers/n_sim:.3f}")
```

## Interpretation

- For small $n$, the z-interval that substitutes $s$ for $\sigma$ **under-covers**: its empirical coverage falls below 95 %. The $t$-interval corrects this by using wider critical values from the $t$-distribution.
- As $n$ grows, the $t$ and $z$ critical values converge, so both intervals perform similarly.
- The Wilson and Agresti--Coull proportion intervals outperform the Wald interval when $n$ is small or $p$ is near 0 or 1.
- The chi-squared variance interval is exact only under normality; for non-normal data a bootstrap CI is preferable.

## Sample Size Determination

To estimate $\mu$ within a margin of error $E$ at confidence level $1 - \alpha$, the required sample size is

$$
n = \left\lceil \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2 \right\rceil
$$

For a proportion with the conservative choice $p = 0.5$,

$$
n = \left\lceil \left(\frac{z_{\alpha/2}}{2E}\right)^2 \right\rceil
$$

```python
sigma_est = 15
for E in [1, 2, 3, 5]:
    for conf in [0.90, 0.95, 0.99]:
        z = stats.norm.ppf(1 - (1 - conf) / 2)
        n_needed = int(np.ceil((z * sigma_est / E)**2))
        print(f"  E=+/-{E}, {conf*100:.0f}% conf -> n = {n_needed}")
```

## Exercises

**Exercise 1.** A random sample of 36 light bulbs has a mean lifetime of 1200 hours with a known population standard deviation of $\sigma = 120$ hours. Construct a 95 % confidence interval for $\mu$ using the $z$-interval. Interpret the result.

??? success "Solution to Exercise 1"

    The standard error is $\text{SE} = 120 / \sqrt{36} = 20$. The critical value is $z_{0.025} = 1.96$. The margin of error is $1.96 \times 20 = 39.2$. Hence the 95 % CI is

    $$
    1200 \pm 39.2 = (1160.8,\; 1239.2)
    $$

    We are 95 % confident that the true mean lifetime lies between 1160.8 and 1239.2 hours. $\square$

---

**Exercise 2.** In a survey of 400 voters, 220 support a ballot measure. Compute the Wald and Wilson 95 % confidence intervals for the true proportion $p$. Which do you prefer and why?

??? success "Solution to Exercise 2"

    Here $\hat{p} = 220/400 = 0.55$ and $z_{0.025} = 1.96$.

    **Wald interval:**

    $$
    \text{SE} = \sqrt{\frac{0.55 \times 0.45}{400}} = 0.02487
    $$

    $$
    0.55 \pm 1.96 \times 0.02487 = (0.5013,\; 0.5987)
    $$

    **Wilson interval:** With $\tilde{n} = 1 + z^2/n = 1 + 3.8416/400 = 1.009604$,

    $$
    \tilde{p} = \frac{0.55 + 3.8416/800}{1.009604} = \frac{0.554802}{1.009604} \approx 0.5495
    $$

    $$
    \text{half-width} = \frac{1.96\sqrt{0.55 \times 0.45/400 + 3.8416/640000}}{1.009604} \approx 0.04872
    $$

    $$
    \text{Wilson CI} \approx (0.5008,\; 0.5983)
    $$

    Both intervals are close because $n$ is large and $\hat{p}$ is not extreme. The Wilson interval is preferred in general because it has better coverage properties for small $n$ or extreme $\hat{p}$. $\square$

---

**Exercise 3.** A sample of 15 measurements from a normal population yields $s^2 = 25$. Construct a 90 % confidence interval for $\sigma^2$. Then derive the corresponding interval for $\sigma$.

??? success "Solution to Exercise 3"

    With $n = 15$, $\text{df} = 14$, $\alpha = 0.10$:

    $$
    \chi^2_{0.05,\,14} = 6.571, \quad \chi^2_{0.95,\,14} = 23.685
    $$

    The 90 % CI for $\sigma^2$ is

    $$
    \left(\frac{14 \times 25}{23.685},\; \frac{14 \times 25}{6.571}\right) = (14.78,\; 53.26)
    $$

    Taking square roots, the 90 % CI for $\sigma$ is

    $$
    (\sqrt{14.78},\; \sqrt{53.26}) = (3.84,\; 7.30)
    $$

    $\square$

---

**Exercise 4.** Prove that as $n \to \infty$, the $t$-interval $\bar{X} \pm t_{\alpha/2,\,n-1} \cdot s/\sqrt{n}$ converges to the $z$-interval $\bar{X} \pm z_{\alpha/2} \cdot \sigma/\sqrt{n}$.

??? success "Solution to Exercise 4"

    Two convergences combine:

    1. **Critical values.** As $\text{df} = n - 1 \to \infty$, the $t_{n-1}$ distribution converges to $N(0,1)$. Therefore $t_{\alpha/2,\,n-1} \to z_{\alpha/2}$.

    2. **Sample standard deviation.** By the Law of Large Numbers, $s^2 \xrightarrow{P} \sigma^2$, so $s \xrightarrow{P} \sigma$ by the continuous mapping theorem.

    Combining, the margin of error satisfies

    $$
    t_{\alpha/2,\,n-1} \cdot \frac{s}{\sqrt{n}} \;\xrightarrow{P}\; z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
    $$

    and the two intervals become indistinguishable for large $n$. $\square$

---

**Exercise 5.** How large a sample is needed to estimate a population mean to within $\pm 2$ units at the 99 % confidence level, assuming $\sigma = 10$?

??? success "Solution to Exercise 5"

    The critical value is $z_{0.005} = 2.576$. The required sample size is

    $$
    n = \left\lceil \left(\frac{2.576 \times 10}{2}\right)^2 \right\rceil = \left\lceil 12.88^2 \right\rceil = \left\lceil 165.87 \right\rceil = 166
    $$

    A sample of at least 166 observations is required. $\square$
