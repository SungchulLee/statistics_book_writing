# Two-Sample t-Test (Pooled and Welch)

## Overview

The two-sample t-test compares the means of two independent samples to determine if they differ significantly. It is widely used in A/B testing, clinical trials, and experimental design.

## Hypotheses

- **Null Hypothesis** ($H_0$): $\mu_1 = \mu_2$ (the means are equal)
- **Alternative Hypothesis** ($H_a$): $\mu_1 \neq \mu_2$ (the means differ)

## Test Statistics

### Pooled t-Test (Assumes Equal Variances)

When both groups are assumed to have the same population variance:

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{S_p\sqrt{1/n_1 + 1/n_2}}
$$

where the pooled standard deviation is:

$$S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}$$

**Degrees of freedom**: $df = n_1 + n_2 - 2$

### Welch's t-Test (Unequal Variances Assumed)

When variances may differ, Welch's test is more robust:

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}
$$

**Degrees of freedom** (Satterthwaite approximation):

$$df = \frac{\left(\frac{S_1^2}{n_1} + \frac{S_2^2}{n_2}\right)^2}{\frac{(S_1^2/n_1)^2}{n_1-1} + \frac{(S_2^2/n_2)^2}{n_2-1}}$$

**Note**: Welch's test is generally preferred as it does not assume equal variances and maintains Type I error control.

## Practical Considerations

### Equal Variance Assumption

Before choosing between pooled and Welch's tests, one might be tempted to perform a pre-test for equality of variances (like Levene's test). However, modern statistical practice recommends using Welch's test as the default because:

1. It is robust to violations of the equal variance assumption
2. It has nearly identical power to the pooled test when variances are actually equal
3. It provides better protection when variances differ

### Effect Size

For two-sample comparisons, **Cohen's d** measures practical significance:

$$d = \frac{\bar{X}_1 - \bar{X}_2}{S_p}$$

Interpretation:

- $|d| < 0.2$: Small effect
- $0.2 \leq |d| < 0.5$: Small to medium effect
- $0.5 \leq |d| < 0.8$: Medium effect
- $|d| \geq 0.8$: Large effect

## Example: Web Page A/B Test

Suppose we test whether users spend more time on a redesigned web page (Page B) than the current version (Page A):

```python
import numpy as np
from scipy import stats

# Session times in seconds
page_a = np.array([185, 188, 142, 160, 161, 157, 182, 181, 159, 167])
page_b = np.array([173, 181, 182, 170, 169, 177, 168, 183, 169, 164])

# Welch's t-test (default: equal_var=False)
t_stat, p_value = stats.ttest_ind(page_a, page_b, equal_var=False)

print(f"Page A: mean = {page_a.mean():.2f}, std = {page_a.std(ddof=1):.2f}")
print(f"Page B: mean = {page_b.mean():.2f}, std = {page_b.std(ddof=1):.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value (two-sided): {p_value:.4f}")

# One-sided test: H_a: μ_B > μ_A
p_one_sided = p_value / 2 if page_b.mean() > page_a.mean() else 1 - p_value / 2
print(f"p-value (one-sided): {p_one_sided:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(page_a) - 1) * page_a.std(ddof=1)**2 +
                       (len(page_b) - 1) * page_b.std(ddof=1)**2) /
                      (len(page_a) + len(page_b) - 2))
cohens_d = (page_b.mean() - page_a.mean()) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")
```

## Implementation in Python

### Using scipy.stats

```python
from scipy import stats

# Welch's t-test (recommended)
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False)

# Pooled t-test
t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)

# One-sided tests
if t_stat > 0:
    p_one_sided = p_value / 2  # Upper tail
else:
    p_one_sided = 1 - p_value / 2  # Lower tail
```

### Using statsmodels

```python
import statsmodels.api as sm

# Welch's t-test with more details
t_stat, p_value, df = sm.stats.ttest_ind(group1, group2,
                                         usevar='unequal',
                                         alternative='two-sided')
```

## Assumptions

1. **Independence**: Observations within each group are independent
2. **Normality**: Data in each group are approximately normally distributed (less critical with n > 30)
3. **Random Sampling**: Samples are randomly drawn from their respective populations

## When to Use Each Test

| Scenario | Test to Use |
|----------|------------|
| Small samples, variances appear equal | Pooled t-test |
| Any sample size or unclear variances | **Welch's t-test** |
| Non-normal data, small samples | Permutation test or Mann-Whitney U test |
| Large samples (n > 30) | Either test (both work well) |

## Related Tests

- **Paired t-test**: For dependent samples (matched pairs)
- **Mann-Whitney U test**: Non-parametric alternative for non-normal data
- **Permutation test**: Assumption-free resampling approach
- **Bootstrap confidence interval**: For confidence intervals without distributional assumptions

## Exercises

**Exercise 1.**
Program A: $\bar X_1 = 55\,000, s_1 = 7\,500, n_1 = 14$. Program B: $\bar X_2 = 60\,000, s_2 = 8\,000, n_2 = 16$. Pooled $t$-test at $\alpha = 0.05$.

??? success "Solution to Exercise 1"
    $S_p^2 = (13 \cdot 56\,250\,000 + 15 \cdot 64\,000\,000)/28 \approx 60\,402\,679$.

    $t = -5000/\sqrt{60\,402\,679 \cdot (1/14 + 1/16)} = -5000/\sqrt{8\,089\,358} \approx -1.76$.

    Critical: $t_{0.025, 28} = \pm 2.048$. $|t| = 1.76 < 2.048$. **Fail to reject.**

---

**Exercise 2.**
Norway ($\bar X = 64.3, s = 18.2, n = 65$) vs US ($\bar X = 53.4, s = 23.9, n = 75$) income. Welch test.

??? success "Solution to Exercise 2"
    $\mathrm{SE} = \sqrt{18.2^2/65 + 23.9^2/75} = \sqrt{5.10 + 7.62} = \sqrt{12.72} \approx 3.57$.

    $t = (64.3 - 53.4)/3.57 \approx 3.06$. Strong significance.

    Welch df: $\nu \approx (12.72)^2/[(5.10)^2/64 + (7.62)^2/74] \approx 137$. Effectively $z$-test for this large df.

    P-value $\approx 0.003$. Reject $H_0$. Norwegian income significantly higher.

---

**Exercise 3.**
US ($\bar X = 25.5, s = 3.8, n = 108$) vs Canada ($\bar X = 26.3, s = 3.2, n = 102$) marriage age. Pooled $t$-test.

??? success "Solution to Exercise 3"
    $S_p^2 = (107 \cdot 14.44 + 101 \cdot 10.24)/208 \approx 12.40$.

    $t = -0.8/\sqrt{12.40 \cdot (1/108 + 1/102)} = -0.8/\sqrt{0.2362} \approx -1.65$.

    Critical: $t_{0.025, 208} \approx \pm 1.97$. $|t| < 1.97$. Fail to reject.

    P-value $\approx 0.10$. Borderline; not significant at 5% but close. Larger samples might detect a real difference.

---

**Exercise 4.**
Electric cars Model A ($\bar X = 168, s = 5.4, n = 5$) vs B ($\bar X = 172, s = 7.5, n = 5$). Welch.

??? success "Solution to Exercise 4"
    $\mathrm{SE} = \sqrt{29.16/5 + 56.25/5} = \sqrt{17.08} \approx 4.13$.

    $t = -4/4.13 \approx -0.97$. Welch df: $\nu \approx (17.08)^2/[5.83^2/4 + 11.25^2/4] \approx 7.3$.

    Critical: $t_{0.025, 7} \approx 2.36$. $|t| < 2.36$. Fail to reject.

    Tiny sample sizes — low power. Cannot conclude difference even if it exists.

---

**Exercise 5.**
**Welch vs pooled.** When does pooled $t$-test fail (give wrong $\alpha$) under unequal variances?

??? success "Solution to Exercise 5"
    Pooled $t$ assumes $\sigma_1 = \sigma_2$. Under unequal variances, the pooled SE estimator is biased, and the test statistic doesn't have an exact $t$ distribution.

    **Failure modes:**

    - **Unequal $n$ and unequal $\sigma$:** can inflate Type I error dramatically. If the smaller sample has larger variance, $\alpha$ can exceed nominal level by factor 2-3.
    - **Equal $n$:** pooled test is robust to unequal variances. $\alpha$ stays near nominal.

    **Welch's test:** doesn't assume equal variances. Slightly less powerful when variances actually are equal (small efficiency loss).

    **Modern default:** Welch (R's `t.test`, scipy's `ttest_ind(equal_var=False)`). Avoids the risk of inflated $\alpha$.

---

**Exercise 6.**
**Effect size and sample-size planning** for two-sample $t$-test.

??? success "Solution to Exercise 6"
    Cohen's $d = (\mu_1 - \mu_2)/\sigma_{\text{pooled}}$. For Exercise 1: $d = -5000/7766 \approx -0.64$.

    Sample size (per group) for two-sample $t$-test:

    $$
    n = \frac{2(z_{\alpha/2} + z_\beta)^2}{d^2}
    $$

    For 80% power at $\alpha = 0.05$: $n \approx 16/d^2$. For $d = 0.5$ (medium): $n \approx 64$. For $d = 0.8$ (large): $n \approx 25$.

    Small effects require large samples — common scaling in social science and medical research. Conduct power analysis *before* the study.
