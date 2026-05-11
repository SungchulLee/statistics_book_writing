# Paired Mean Confidence Interval Coverage Simulation

## Overview

When two measurements are taken on the same subject (e.g., before and after a treatment), the observations within each pair are correlated. The paired-sample confidence interval for the mean difference $\mu_D = \mu_X - \mu_Y$ reduces the problem to a one-sample interval on the differences $D_i = X_i - Y_i$. This page simulates coverage for three methods applied to paired data and shows how within-pair correlation affects the interval width.

## Paired Confidence Interval

Given paired observations $(X_1, Y_1), \ldots, (X_n, Y_n)$, define the differences $D_i = X_i - Y_i$. The sample mean and standard deviation of the differences are

$$
\bar{D} = \frac{1}{n}\sum_{i=1}^n D_i, \quad S_D = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (D_i - \bar{D})^2}
$$

### t-Interval (Default)

$$
\bar{D} \pm t_{\alpha/2,\,n-1} \cdot \frac{S_D}{\sqrt{n}}
$$

### z-Interval with Known Variance of D

When $\sigma_D$ is known (rare in practice):

$$
\bar{D} \pm z_{\alpha/2} \cdot \frac{\sigma_D}{\sqrt{n}}
$$

The true variance of the difference is $\sigma_D^2 = \sigma_X^2 + \sigma_Y^2 - 2\rho\,\sigma_X \sigma_Y$, where $\rho$ is the within-pair correlation.

### z-Interval with Plug-in s

$$
\bar{D} \pm z_{\alpha/2} \cdot \frac{S_D}{\sqrt{n}}
$$

This is a large-$n$ approximation that under-covers for small samples.

## Role of Correlation

Notice the variance of the differences:

$$
\operatorname{Var}(D) = \sigma_X^2 + \sigma_Y^2 - 2\rho\,\sigma_X\sigma_Y
$$

When $\rho > 0$ (positive within-pair correlation), the variance of $D$ is **reduced** compared to $\sigma_X^2 + \sigma_Y^2$. This is the statistical advantage of pairing: the confidence interval is narrower because much of the between-subject variability cancels out.

## Python Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

n_simulations = 100
n = 12
mu_x, mu_y = 0.5, 0.0
sigma_x, sigma_y = 1.0, 1.2
rho = 0.6
alpha = 0.05
method = "t"  # 't' | 'z_known' | 'z_plugin'

rng = np.random.default_rng(None)

delta_true = mu_x - mu_y
var_d_true = sigma_x**2 + sigma_y**2 - 2 * rho * sigma_x * sigma_y
sigma_d_true = np.sqrt(var_d_true)

# Build covariance matrix and Cholesky factor
cov = rho * sigma_x * sigma_y
Sigma = np.array([[sigma_x**2, cov], [cov, sigma_y**2]])
L = np.linalg.cholesky(Sigma)

df = n - 1
t_star = t.ppf(1 - alpha / 2, df=df)
z_star = norm.ppf(1 - alpha / 2)

lowers = np.empty(n_simulations)
uppers = np.empty(n_simulations)
centers = np.empty(n_simulations)

for i in range(n_simulations):
    z_vals = rng.standard_normal(size=(2, n))
    xy = (L @ z_vals).T
    x = xy[:, 0] + mu_x
    y = xy[:, 1] + mu_y
    d = x - y
    dbar = d.mean()
    s_d = d.std(ddof=1)

    if method == "t":
        se, crit = s_d / np.sqrt(n), t_star
    elif method == "z_known":
        se, crit = sigma_d_true / np.sqrt(n), z_star
    else:
        se, crit = s_d / np.sqrt(n), z_star

    lowers[i] = dbar - crit * se
    uppers[i] = dbar + crit * se
    centers[i] = dbar

covered = (lowers <= delta_true) & (delta_true <= uppers)
coverage_pct = 100.0 * covered.mean()
print(f"Paired {method} coverage: {coverage_pct:.1f}%")
```

### Plotting the Intervals

```python
fig, ax = plt.subplots(figsize=(12, 12))
for i in range(n_simulations):
    color = "k" if covered[i] else "r"
    ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
    ax.plot(centers[i], i, marker="o", ms=3, color=color)

ax.axvline(delta_true, linestyle="--", linewidth=1.5, color="r")
n_fail = int((~covered).sum())
ax.set_title(f"{n_simulations} Paired {method} CIs | n={n}, rho={rho}, CL=95%")
ax.set_yticks([])
ax.set_xlabel("Mean difference")
plt.tight_layout()
plt.show()
```

## Interpretation

- The paired $t$-interval achieves the nominal 95 % coverage because it correctly accounts for the estimation of $\sigma_D$.
- Higher within-pair correlation $\rho$ reduces $\sigma_D$, producing narrower confidence intervals.
- The plug-in $z$-interval under-covers for small $n$ because the normal critical value is too small relative to the $t$ critical value.
- Pairing is beneficial when $\rho > 0$. If $\rho \le 0$, pairing can actually **widen** the interval relative to an independent two-sample design, and the design should be reconsidered.

## Exercises

**Exercise 1.** Ten patients have their blood pressure measured before and after a medication. The differences $D_i$ (before minus after) are: 5, 3, 8, 2, 6, 4, 7, 1, 5, 3. Construct a 95 % $t$-interval for $\mu_D$.

??? success "Solution to Exercise 1"

    Computing summary statistics:

    $$
    \bar{D} = \frac{5+3+8+2+6+4+7+1+5+3}{10} = \frac{44}{10} = 4.4
    $$

    $$
    S_D = \sqrt{\frac{1}{9}\sum_{i=1}^{10}(D_i - 4.4)^2} = \sqrt{\frac{1}{9}(0.36+1.96+12.96+5.76+2.56+0.16+6.76+11.56+0.36+1.96)} = \sqrt{\frac{44.4}{9}} = \sqrt{4.933} \approx 2.221
    $$

    With $\text{df} = 9$ and $t_{0.025,9} = 2.262$:

    $$
    4.4 \pm 2.262 \times \frac{2.221}{\sqrt{10}} = 4.4 \pm 2.262 \times 0.7024 = 4.4 \pm 1.589
    $$

    The 95 % CI for $\mu_D$ is $(2.81, 5.99)$. Since the interval is entirely positive, the medication appears to reduce blood pressure. $\square$

---

**Exercise 2.** Derive the formula $\sigma_D^2 = \sigma_X^2 + \sigma_Y^2 - 2\rho\,\sigma_X\sigma_Y$ for paired observations.

??? success "Solution to Exercise 2"

    Let $D = X - Y$. By the properties of variance:

    $$
    \operatorname{Var}(D) = \operatorname{Var}(X - Y) = \operatorname{Var}(X) + \operatorname{Var}(Y) - 2\operatorname{Cov}(X,Y)
    $$

    Since $\operatorname{Cov}(X,Y) = \rho\,\sigma_X\sigma_Y$ by definition of the correlation coefficient, we obtain

    $$
    \sigma_D^2 = \sigma_X^2 + \sigma_Y^2 - 2\rho\,\sigma_X\sigma_Y
    $$

    When $\rho > 0$, the subtracted term $2\rho\,\sigma_X\sigma_Y > 0$ reduces the variance of $D$ below $\sigma_X^2 + \sigma_Y^2$, which is the variance one would get if $X$ and $Y$ were independent. $\square$

---

**Exercise 3.** Suppose $\sigma_X = \sigma_Y = \sigma$ and $\rho = 0.8$. Compare the standard error of $\bar{D}$ from a paired design with $n$ pairs to the standard error of $\bar{X} - \bar{Y}$ from an independent two-sample design with $n$ observations per group.

??? success "Solution to Exercise 3"

    **Paired design:** $\sigma_D^2 = \sigma^2 + \sigma^2 - 2(0.8)\sigma^2 = 2\sigma^2(1 - 0.8) = 0.4\sigma^2$. The standard error is

    $$
    \text{SE}_{\text{paired}} = \frac{\sigma_D}{\sqrt{n}} = \frac{\sigma\sqrt{0.4}}{\sqrt{n}} = \frac{0.632\,\sigma}{\sqrt{n}}
    $$

    **Independent design:** $\operatorname{Var}(\bar{X} - \bar{Y}) = \sigma^2/n + \sigma^2/n = 2\sigma^2/n$. The standard error is

    $$
    \text{SE}_{\text{indep}} = \sqrt{\frac{2\sigma^2}{n}} = \frac{\sigma\sqrt{2}}{\sqrt{n}} = \frac{1.414\,\sigma}{\sqrt{n}}
    $$

    The ratio is $\text{SE}_{\text{paired}}/\text{SE}_{\text{indep}} = \sqrt{0.4}/\sqrt{2} = \sqrt{0.2} \approx 0.447$. The paired design cuts the standard error by more than half, yielding a much narrower confidence interval. $\square$

---

**Exercise 4.** For what value of $\rho$ does the paired design offer no advantage over an independent design? What happens when $\rho < 0$?

??? success "Solution to Exercise 4"

    With $\sigma_X = \sigma_Y = \sigma$, the paired variance is $\sigma_D^2 = 2\sigma^2(1-\rho)$ and the independent variance of $\bar{X}-\bar{Y}$ is $2\sigma^2/n$ (with $n$ per group). Comparing standard errors:

    $$
    \text{SE}_{\text{paired}} = \frac{\sigma\sqrt{2(1-\rho)}}{\sqrt{n}}, \quad \text{SE}_{\text{indep}} = \frac{\sigma\sqrt{2}}{\sqrt{n}}
    $$

    These are equal when $\sqrt{2(1-\rho)} = \sqrt{2}$, i.e., $\rho = 0$. When $\rho = 0$ the measurements within a pair are uncorrelated and pairing provides no reduction in variance.

    When $\rho < 0$, we have $2(1-\rho) > 2$, so $\text{SE}_{\text{paired}} > \text{SE}_{\text{indep}}$. Negative within-pair correlation actually **increases** the variance of the differences, making the paired design **worse** than the independent design. This situation is unusual in practice but could arise if, for example, paired subjects tend to respond in opposite directions. $\square$

---

**Exercise 5.** A study uses $n = 15$ paired observations. The sample mean difference is $\bar{D} = 2.3$ and $S_D = 4.1$. Test whether $\mu_D = 0$ at the 5 % level by checking if 0 lies inside the 95 % confidence interval.

??? success "Solution to Exercise 5"

    With $\text{df} = 14$ and $t_{0.025,14} = 2.145$:

    $$
    \text{SE} = \frac{4.1}{\sqrt{15}} = \frac{4.1}{3.873} = 1.059
    $$

    $$
    \bar{D} \pm t_{0.025,14} \times \text{SE} = 2.3 \pm 2.145 \times 1.059 = 2.3 \pm 2.272
    $$

    The 95 % CI is $(0.028, 4.572)$. Since $0$ falls outside this interval (just barely), we reject $H_0: \mu_D = 0$ at the 5 % significance level. The data provide evidence that the true mean difference is positive. $\square$
