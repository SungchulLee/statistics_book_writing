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

## Case C: Normal Populations, Equal Variances (Pooled t)
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

## Case D: Normal Populations, Unequal Variances (Welch's t)
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

## Exercises

**Exercise 1.**
Population A: $\sigma_A = 15$, $n_A = 36$. Population B: $\sigma_B = 20$, $n_B = 49$. Compute $\mathrm{SE}(\bar X_A - \bar X_B)$.

??? success "Solution to Exercise 1"
    By independence:

    $$
    \mathrm{Var}(\bar X_A - \bar X_B) = \sigma_A^2/n_A + \sigma_B^2/n_B = 225/36 + 400/49 = 6.25 + 8.16 = 14.41
    $$

    $\mathrm{SE} \approx 3.80$.

---

**Exercise 2.**
**Distribution of the difference.** If $\bar X_A, \bar X_B$ are independent and (approximately) normal, derive the distribution of $\bar X_A - \bar X_B$.

??? success "Solution to Exercise 2"
    Linear combinations of independent normals are normal. So:

    $$
    \bar X_A - \bar X_B \sim N(\mu_A - \mu_B, \sigma_A^2/n_A + \sigma_B^2/n_B)
    $$

    Under $H_0: \mu_A = \mu_B$, the centered statistic $Z = (\bar X_A - \bar X_B)/\sqrt{\sigma_A^2/n_A + \sigma_B^2/n_B} \sim N(0, 1)$.

    With $\sigma_A, \sigma_B$ unknown, substitute $s_A, s_B$ and use a $t$ distribution (Welch's with adjusted df).

---

**Exercise 3.**
**Hypothesis test.** Sample A: $\bar x_A = 45$, $s_A = 10$, $n_A = 50$. Sample B: $\bar x_B = 40$, $s_B = 12$, $n_B = 50$. Test $H_0: \mu_A = \mu_B$ vs. $H_1: \mu_A \ne \mu_B$ at $\alpha = 0.05$.

??? success "Solution to Exercise 3"
    $\mathrm{SE} = \sqrt{100/50 + 144/50} = \sqrt{4.88} \approx 2.21$.

    $t = (45 - 40)/2.21 \approx 2.26$.

    Welch's df: $\nu = (\mathrm{SE}^2)^2/[(s_A^2/n_A)^2/(n_A - 1) + (s_B^2/n_B)^2/(n_B - 1)] \approx (4.88)^2/[(2)^2/49 + (2.88)^2/49] = 23.8/0.250 \approx 95.5$. Round to 95.

    Critical value: $t_{0.975, 95} \approx 1.985$.

    Since $|t| = 2.26 > 1.985$, **reject $H_0$**. Evidence of a difference in means at the 5% level.

    $p$-value: $2 \cdot P(T_{95} > 2.26) \approx 2 \cdot 0.013 = 0.026$.

---

**Exercise 4.**
**Confidence interval.** Continuing Exercise 3, construct a 95% CI for $\mu_A - \mu_B$.

??? success "Solution to Exercise 4"
    $\mathrm{CI} = (\bar x_A - \bar x_B) \pm t_{0.975, 95} \cdot \mathrm{SE} = 5 \pm 1.985 \cdot 2.21 = 5 \pm 4.39$.

    95% CI: $(0.61, 9.39)$.

    Does *not* contain 0 — consistent with rejecting $H_0$. The data suggest A's mean exceeds B's by 0.6 to 9.4 units.

    The CI is generally more informative than the $p$-value: it gives the magnitude and direction of the difference, not just a binary "significant or not."

---

**Exercise 5.**
**Paired vs. independent.** Two-sample $t$ assumes independent samples. When the data is **paired** (e.g., before/after measurements on the same subjects), describe the appropriate test and why paired analysis is more powerful.

??? success "Solution to Exercise 5"
    **Paired $t$-test:** compute differences $D_i = X_{1i} - X_{2i}$ for each pair $i = 1, \ldots, n$. Test $H_0: \mathbb{E}[D] = 0$ using a one-sample $t$-test on $D_i$'s.

    Test statistic: $t = \bar D/(s_D/\sqrt n) \sim t_{n-1}$ under $H_0$ assuming pairs are i.i.d.

    **Why more powerful:** pairing removes between-subject variability. If $\mathrm{Cov}(X_1, X_2) > 0$ (typical for repeated measures on the same subject), then

    $$
    \mathrm{Var}(D) = \mathrm{Var}(X_1) + \mathrm{Var}(X_2) - 2\mathrm{Cov}(X_1, X_2) < \mathrm{Var}(X_1) + \mathrm{Var}(X_2)
    $$

    Smaller variance ⟹ smaller SE ⟹ more powerful test.

    Always use paired analysis when the design is paired. Ignoring pairing wastes information and reduces power.

---

**Exercise 6.**
**Equivalence testing.** Test whether two means are practically **equivalent** within a tolerance $\delta = 2$. State the null and alternative for the **TOST procedure** (Two One-Sided Tests).

??? success "Solution to Exercise 6"
    Traditional test: $H_0: \mu_A = \mu_B$ vs. $H_1: \mu_A \ne \mu_B$. Failing to reject doesn't prove equivalence — it could mean inadequate power.

    **Equivalence test (TOST):** swap null and alternative.

    $H_0: |\mu_A - \mu_B| \ge \delta$ (not equivalent) vs. $H_1: |\mu_A - \mu_B| < \delta$ (equivalent).

    Implementation: two one-sided tests at level $\alpha$.

    - Test $H_{01}: \mu_A - \mu_B \le -\delta$ vs. $H_{11}: \mu_A - \mu_B > -\delta$.
    - Test $H_{02}: \mu_A - \mu_B \ge \delta$ vs. $H_{12}: \mu_A - \mu_B < \delta$.
    - Reject $H_0$ (declare equivalence) iff both one-sided tests reject.

    Equivalent to: confidence interval $(\bar X_A - \bar X_B) \pm t_{1-\alpha} \cdot \mathrm{SE}$ entirely within $(-\delta, +\delta)$.

    Used in bioequivalence trials (FDA requires showing two formulations are equivalent within $\pm 20\%$), A/B test "no harm" verifications, and any setting where "no meaningful difference" is the desired conclusion.
