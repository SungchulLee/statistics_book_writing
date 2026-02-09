# Two-Sample Tests

## 1. Two Sample z Test

The two sample z-test determines whether the means of two independent samples differ significantly, given that the population variances are known. It applies when comparing two groups under different conditions and assumes normally distributed, independent samples.

### A. Hypothesis

- **Null**: $H_0: \mu_1 = \mu_2$
- **Alternative**:
    - Two-tailed: $H_a: \mu_1 \neq \mu_2$
    - One-tailed (greater): $H_a: \mu_1 > \mu_2$
    - One-tailed (less): $H_a: \mu_1 < \mu_2$

### B. Test Statistic

$$ z = \frac{(\bar{x}_1 - \bar{x}_2) - (\mu_1 - \mu_2)}{\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}} $$

Under $H_0$ (where $\mu_1 - \mu_2 = 0$), and when using sample standard deviations for large samples:

$$ z = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}} $$

### C. Decision Rule

- **Two-tailed**: Reject $H_0$ if $|z| > z_{\alpha/2}$.
- **One-tailed (greater)**: Reject $H_0$ if $z > z_{\alpha}$.
- **One-tailed (less)**: Reject $H_0$ if $z < -z_{\alpha}$.

### D. P-value

- Two-tailed: $p\text{-value} = 2P(Z \geq |z|)$
- One-tailed (greater): $p\text{-value} = P(Z \geq z)$
- One-tailed (less): $p\text{-value} = P(Z \leq z)$

### E. Example

```python
import numpy as np
from scipy import stats

n_f, n_s = 100, 100
x_bar_f, x_bar_s = 1.85, 1.65
s_f, s_s = 1.3, 1.2

statistic = (x_bar_f - x_bar_s) / np.sqrt(s_f**2/n_f + s_s**2/n_s)
p_value = stats.norm().sf(abs(statistic)) * 2

print(f"statistic : {statistic:.4f}")
print(f"p value   : {p_value:.4f}")
```

---

## 2. Two Sample t Test

The two sample t-test (independent samples t-test) determines whether the means of two independent groups differ significantly. It is used when the population variances are unknown and assumed to be equal.

### A. Hypothesis

- **Null**: $H_0: \mu_1 = \mu_2$
- **Alternative**:
    - Two-tailed: $H_a: \mu_1 \neq \mu_2$
    - One-tailed (greater): $H_a: \mu_1 > \mu_2$
    - One-tailed (less): $H_a: \mu_1 < \mu_2$

### B. Test Statistic (Pooled Variance)

$$ t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \cdot \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} $$

where the pooled standard deviation is:

$$ s_p = \sqrt{\frac{(n_1 - 1) s_1^2 + (n_2 - 1) s_2^2}{n_1 + n_2 - 2}} $$

This statistic follows a t-distribution with $n_1 + n_2 - 2$ degrees of freedom.

### C. Decision Rule

- **Two-tailed**: Reject $H_0$ if $|t| > t_{\alpha/2, n_1+n_2-2}$.
- **One-tailed (greater)**: Reject $H_0$ if $t > t_{\alpha, n_1+n_2-2}$.
- **One-tailed (less)**: Reject $H_0$ if $t < -t_{\alpha, n_1+n_2-2}$.

### D. Examples

#### Example: Gender Bias in Salary

Market researchers compare average salaries for male vs female managers.

$$H_0 : \mu_{\text{men}} = \mu_{\text{women}} \quad\text{vs}\quad H_1: \mu_{\text{men}} > \mu_{\text{women}}$$

#### Example: Tomatoes from Two Different Fields

| | Field A | Field B |
|:---:|:---:|:---:|
| Mean | 1.3m | 1.6m |
| Std Dev | 0.5m | 0.3m |
| n | 22 | 24 |

$$H_0 : \mu_A = \mu_B \quad\text{vs}\quad H_1: \mu_A \neq \mu_B$$

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

X_1_bar, X_2_bar = 1.3, 1.6
s_1, s_2 = 0.5, 0.3
n_1, n_2 = 22, 24

# Welch's approach (unequal variances)
statistic = (X_1_bar - X_2_bar) / np.sqrt(s_1**2 / n_1 + s_2**2 / n_2)

# Welch-Satterthwaite degrees of freedom
top = (s_1**2 / n_1 + s_2**2 / n_2)**2
bottom = (s_1**2 / n_1)**2 / (n_1 - 1) + (s_2**2 / n_2)**2 / (n_2 - 1)
df = top / bottom

p_value = 2 * stats.t(df).cdf(-abs(statistic))
print(f"{df = :.4f}")
print(f"{statistic = :.4f}")
print(f"{p_value   = :.4f}")

alpha = 0.05
if p_value <= alpha:
    print("Reject H_0")
else:
    print("Fail to reject H_0")
```

#### Example: Number of Babies (France vs Switzerland)

| | France | Switzerland |
|:---:|:---:|:---:|
| Mean | 1.85 | 1.65 |
| Std Dev | 1.3 | 1.2 |
| n | 100 | 100 |

Using pooled variance:

```python
X_1_bar, X_2_bar = 1.3, 1.6
s_1, s_2 = 0.5, 0.3
n_1, n_2 = 22, 24

s_p_square = ((n_1 - 1) * s_1**2 + (n_2 - 1) * s_2**2) / (n_1 + n_2 - 2)
statistic = (X_1_bar - X_2_bar) / np.sqrt(s_p_square / n_1 + s_p_square / n_2)
df = n_1 + n_2 - 2
p_value = 2 * stats.t(df).cdf(-abs(statistic))

print(f"{df = :.4f}")
print(f"{statistic = :.4f}")
print(f"{p_value   = :.4f}")
```

#### Example: Two Varieties of Pears (Bosc and Anjou)

| | Bosc | Anjou |
|:---:|:---:|:---:|
| Mean | 120 | 116 |
| Std Dev | 15 | 13 |
| n | 65 | 65 |

The 99% confidence interval for $\mu_{\text{Bosc}} - \mu_{\text{Anjou}}$ is $4 \pm 6.44$, i.e., $(-2.44, 10.44)$. Since the confidence interval contains 0, we fail to reject $H_0$ at $\alpha = 0.01$.

---

## 3. Welch's t Test

**Welch's t-test** is a robust adaptation of the standard two-sample t-test that accounts for unequal variances and potentially unequal sample sizes.

### Formula

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

The degrees of freedom are approximated using the **Welch-Satterthwaite equation**:

$$df = \frac{\left( \frac{s_1^2}{n_1} + \frac{s_2^2}{n_2} \right)^2}{\frac{\left( \frac{s_1^2}{n_1} \right)^2}{n_1 - 1} + \frac{\left( \frac{s_2^2}{n_2} \right)^2}{n_2 - 1}}$$

### When to Use

- When the two groups have noticeably different variances.
- When the sample sizes between the two groups differ significantly.
- When population variances are unknown.

### Python Implementation

```python
import numpy as np
from scipy.stats import ttest_ind

team_a = [120, 118, 125, 130, 115, 122, 121, 119, 117, 123, 124, 126, 127, 118, 116]
team_b = [135, 132, 137, 140, 136, 130, 134, 138, 139, 133, 131, 142, 141,
           129, 128, 135, 137, 136, 134, 132]

stat, p_value = ttest_ind(team_a, team_b, equal_var=False)
print(f"Test Statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Reject H0: The means are significantly different.")
else:
    print("Fail to reject H0.")
```

### Comparison to Standard Two-Sample t-Test

| Feature | Standard t-Test | Welch's t-Test |
|---|---|---|
| Variance assumption | Equal variances | No equal variance assumption |
| Sample sizes | Similar sizes assumed | Handles unequal sizes |
| Degrees of freedom | Fixed: $n_1 + n_2 - 2$ | Approximated via Welch-Satterthwaite |

---

## 4. Two Sample Proportion Test

The two sample proportion test determines if there is a significant difference between the proportions of two independent groups based on a binary outcome.

### A. Hypothesis

- **Null**: $H_0: p_1 = p_2$
- **Alternative**:
    - Two-tailed: $H_a: p_1 \neq p_2$
    - One-tailed (greater): $H_a: p_1 > p_2$
    - One-tailed (less): $H_a: p_1 < p_2$

### B. Test Statistic

The pooled proportion:

$$\hat{p}_{\text{pool}} = \frac{x_1 + x_2}{n_1 + n_2}$$

The test statistic:

$$z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}_{\text{pool}} (1 - \hat{p}_{\text{pool}}) \left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}$$

### C. Examples

#### Example: Support for New Law

| | District A | District B | Total |
|:---:|:---:|:---:|:---:|
| Yes | 58 | 52 | 110 |
| No | 42 | 48 | 90 |

$$H_0 : p_A = p_B \quad\text{vs}\quad H_1: p_A \neq p_B$$

```python
import numpy as np
from scipy import stats

positive_A, positive_B = 58, 52
n_A, n_B = 100, 100
p_hat_A, p_hat_B = positive_A / n_A, positive_B / n_B

p_pooled = (positive_A + positive_B) / (n_A + n_B)
statistic = (p_hat_A - p_hat_B) / (np.sqrt(p_pooled * (1 - p_pooled)) * np.sqrt(1/n_A + 1/n_B))
p_value = stats.norm().sf(abs(statistic)) * 2

print(f"{statistic = :.4f}")
print(f"{p_value = :.4f}")

alpha = 0.05
if p_value <= alpha:
    print("Reject H_0")
else:
    print("Fail to reject H_0")
```

#### Example: Derrick's Approval Rate

Derrick tests whether the prime minister's approval is lower in December vs November.

$$H_0 : p_{\text{Nov}} = p_{\text{Dec}} \quad\text{vs}\quad H_1: p_{\text{Nov}} > p_{\text{Dec}}$$

#### Example: Dime and Nickel

Kiley tests if a dime and nickel have the same likelihood of showing heads.

$$H_0 : p_{\text{Dime}} = p_{\text{Nickel}} \quad\text{vs}\quad H_1: p_{\text{Dime}} \neq p_{\text{Nickel}}$$

#### Example: Myopia

Researchers test whether myopia prevalence increased from 2000 to 2015. In 2000: 132/400 positive. In 2015: 228/600 positive.

$$H_0 : p_{2000} = p_{2015} \quad\text{vs}\quad H_1: p_{2000} < p_{2015}$$

```python
n_2000, n_2015 = 400, 600
positive_2000, positive_2015 = 132, 228
p_hat_2000, p_hat_2015 = positive_2000 / n_2000, positive_2015 / n_2015

p_pooled = (positive_2000 + positive_2015) / (n_2000 + n_2015)
statistic = (p_hat_2000 - p_hat_2015) / (np.sqrt(p_pooled * (1 - p_pooled)) * np.sqrt(1/n_2000 + 1/n_2015))
p_value = stats.norm().cdf(statistic)

print(f"{statistic = :.4f}")
print(f"{p_value = :.4f}")

alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: significant increase in myopia")
else:
    print("Fail to reject H_0")
```

#### Example: Cat Disease

Veterinarians test $H_0: p_{\text{male}} = p_{\text{female}}$ vs $H_1: p_{\text{male}} > p_{\text{female}}$ with 24/259 male cats and 14/241 female cats affected.

```python
positive_male, positive_female = 24, 14
n_male, n_female = 259, 241
p_hat_male, p_hat_female = positive_male / n_male, positive_female / n_female

p_pooled = (positive_male + positive_female) / (n_male + n_female)
statistic = (p_hat_male - p_hat_female) / (np.sqrt(p_pooled * (1 - p_pooled)) * np.sqrt(1/n_male + 1/n_female))
p_value = stats.norm().sf(statistic)

print(f"{statistic = :.4f}")
print(f"{p_value = :.4f}")
```

#### Example: In-person vs Online Classes

A 95% confidence interval for $p_{\text{in\_person}} - p_{\text{online}}$ is $(-0.04, 0.14)$. Since the interval contains 0, we fail to reject $H_0: p_{\text{in\_person}} = p_{\text{online}}$.

---

## 5. Mann-Whitney U Test (Wilcoxon Rank-Sum Test)

The Mann-Whitney U test is a non-parametric test used to compare the distributions of two independent groups. It is useful when the assumptions of a parametric test are not met (e.g., non-normality or ordinal data).

### Key Features

- Tests whether the distributions of two independent groups are the same.
- Assumptions: independent groups, ordinal/interval/ratio data, random samples.
- **Null**: The two groups have the same distribution.
- **Alternative**: The distributions differ, or one group tends to have higher values.

### How the Test Works

1. Combine all data and assign **ranks** (average tied ranks).
2. Calculate rank sums $R_1$ and $R_2$.
3. Compute U-statistics:
    - $U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1$
    - $U_2 = n_1 n_2 + \frac{n_2(n_2+1)}{2} - R_2$
4. Test statistic: $U = \min(U_1, U_2)$.
5. For large samples ($n_1, n_2 > 20$), use normal approximation with Z-score.

### Interpretation

- If $p < 0.05$: Reject $H_0$; the two groups have significantly different distributions.
- If $p$ is large: Fail to reject $H_0$.

### Which Group is Larger?

If $H_0$ is rejected, compare **mean ranks** of the two groups. A higher mean rank indicates that group tends to have larger values.

### Note

The Mann-Whitney U test and Wilcoxon rank-sum test are statistically equivalent. The terminology varies by software (e.g., "Mann-Whitney U" in SPSS, "Wilcoxon rank-sum" in R).
