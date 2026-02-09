# Sampling Distribution of the Difference of Two Sample Means

## Overview

When comparing two populations, we often examine the difference $\bar{X}_1 - \bar{X}_2$. The sampling distribution of this difference determines the appropriate test statistic, confidence interval formula, and distributional reference — which vary depending on what is known about the population variances and sample sizes.

## Setup

Let $X_1^{(1)}, \dots, X_{n_1}^{(1)}$ be i.i.d. from population 1 with mean $\mu_1$ and variance $\sigma_1^2$, and $X_1^{(2)}, \dots, X_{n_2}^{(2)}$ be i.i.d. from population 2 with mean $\mu_2$ and variance $\sigma_2^2$. Assume the two samples are **independent**.

### Common Properties (All Cases)

$$
E[\bar{X}_1 - \bar{X}_2] = \mu_1 - \mu_2
$$

$$
\text{Var}(\bar{X}_1 - \bar{X}_2) = \frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}
$$

## Case A: Population Variances Known

When $\sigma_1^2$ and $\sigma_2^2$ are known:

$$
Z = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}} \sim N(0, 1)
$$

**Confidence interval:**

$$
(\bar{X}_1 - \bar{X}_2) \pm z_{\alpha/2} \sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}
$$

## Case B: Large Sample Sizes

When $n_1$ and $n_2$ are both large (CLT applies), replace $\sigma_i^2$ with $S_i^2$:

$$
Z = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}} \approx N(0, 1)
$$

**Confidence interval:**

$$
(\bar{X}_1 - \bar{X}_2) \pm z_{\alpha/2} \sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}
$$

## Case C: Normal Populations, Equal Variances (Pooled $t$)

When both populations are normal and $\sigma_1^2 = \sigma_2^2 = \sigma^2$:

$$
T = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{S_p^2\!\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}} \sim t_{n_1 + n_2 - 2}
$$

where the **pooled variance** is:

$$
S_p^2 = \frac{(n_1 - 1)S_1^2 + (n_2 - 1)S_2^2}{n_1 + n_2 - 2}
= \frac{\sum_{i=1}^{n_1}(X_i^{(1)} - \bar{X}_1)^2 + \sum_{i=1}^{n_2}(X_i^{(2)} - \bar{X}_2)^2}{n_1 + n_2 - 2}
$$

**Confidence interval:**

$$
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, \, n_1+n_2-2} \sqrt{S_p^2\!\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}
$$

## Case D: Normal Populations, Unequal Variances (Welch's $t$)

When both populations are normal but $\sigma_1^2 \neq \sigma_2^2$:

$$
T = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}} \sim t_\nu
$$

where the **Welch–Satterthwaite degrees of freedom** are:

$$
\nu = \frac{\left(\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}\right)^2}{\frac{\left(\frac{S_1^2}{n_1}\right)^2}{n_1} + \frac{\left(\frac{S_2^2}{n_2}\right)^2}{n_2}}
$$

**Confidence interval:**

$$
(\bar{X}_1 - \bar{X}_2) \pm t_{\alpha/2, \, \nu} \sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}
$$

## Case E: Conservative Degrees of Freedom

When the Welch formula is inconvenient, a conservative (safe) alternative uses:

$$
\text{df} = \min(n_1 - 1, \; n_2 - 1)
$$

This always underestimates the true degrees of freedom, producing wider confidence intervals.

$$
T = \frac{(\bar{X}_1 - \bar{X}_2) - (\mu_1 - \mu_2)}{\sqrt{\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}}} \sim t_{\min(n_1-1, \, n_2-1)}
$$

## Decision Guide

| Conditions | Statistic | Reference Distribution |
|-----------|-----------|----------------------|
| $\sigma_1^2, \sigma_2^2$ known | $Z$ | $N(0,1)$ |
| Large $n_1, n_2$ | $Z$ | $N(0,1)$ (approx.) |
| Normal, $\sigma_1^2 = \sigma_2^2$ | Pooled $t$ | $t_{n_1+n_2-2}$ |
| Normal, $\sigma_1^2 \neq \sigma_2^2$ | Welch's $t$ | $t_\nu$ (Satterthwaite) |
| Normal, quick approximation | Conservative $t$ | $t_{\min(n_1-1, n_2-1)}$ |

## Example: Two Cupcake Shifts

**Problem.** A bakery has two shifts. Shift A: $\mu_A = 130$g, $\sigma_A = 4$g. Shift B: $\mu_B = 125$g, $\sigma_B = 3$g. With $n_A = n_B = 40$, find $P(|\bar{X}_A - \bar{X}_B| > 6)$.

**Solution.** Since $\sigma_A, \sigma_B$ are known (Case A):

$$
\text{SE} = \sqrt{\frac{4^2}{40} + \frac{3^2}{40}} = \sqrt{\frac{16 + 9}{40}} = \sqrt{0.625} \approx 0.7906
$$

**Upper tail:**

$$
Z = \frac{6 - (130 - 125)}{0.7906} = \frac{1}{0.7906} \approx 1.265
$$

$$
P(\bar{X}_A - \bar{X}_B > 6) = P(Z > 1.265) \approx 0.1030
$$

**Lower tail:**

$$
Z = \frac{-6 - (130 - 125)}{0.7906} = \frac{-11}{0.7906} \approx -13.91
$$

$$
P(\bar{X}_A - \bar{X}_B < -6) \approx 0.0000
$$

**Answer:** $P(|\bar{X}_A - \bar{X}_B| > 6) \approx 0.1030$.

```python
import numpy as np
from scipy import stats

se = np.sqrt(16/40 + 9/40)
z_upper = (6 - 5) / se
z_lower = (-6 - 5) / se
prob = stats.norm.sf(z_upper) + stats.norm.cdf(z_lower)
print(f"P(|X_bar_A - X_bar_B| > 6) = {prob:.4f}")
```

## Example: Standard Error of the Difference

**Problem.** Population A: $\mu_A = 100$, $\sigma_A = 15$, $n_A = 36$. Population B: $\mu_B = 110$, $\sigma_B = 20$, $n_B = 49$. Find $\text{SE}(\bar{X}_A - \bar{X}_B)$.

**Solution.**

$$
\text{SE} = \sqrt{\frac{15^2}{36} + \frac{20^2}{49}} = \sqrt{6.25 + 8.16} = \sqrt{14.41} \approx 3.80
$$

## Summary

| Case | Key Condition | Distribution | df |
|------|--------------|-------------|-----|
| A | $\sigma$'s known | $Z$ | — |
| B | Large $n$ | $Z$ (approx.) | — |
| C | Normal, equal $\sigma$ | $t$ (pooled) | $n_1 + n_2 - 2$ |
| D | Normal, unequal $\sigma$ | $t$ (Welch) | Satterthwaite |
| E | Normal, quick approx. | $t$ (conservative) | $\min(n_1-1, n_2-1)$ |

In all cases, the confidence interval takes the form:

$$
(\bar{X}_1 - \bar{X}_2) \pm (\text{critical value}) \times \text{SE}
$$

The choice of critical value ($z^*$ or $t^*$) and SE formula depend on the case.
