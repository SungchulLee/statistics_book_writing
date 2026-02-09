# One-Sample Tests

## 1. One Sample z Test

The one sample z test is a statistical method used to determine whether the mean of a single sample of data differs significantly from a known or hypothesized population mean, given that the population standard deviation is known.

Suppose we have only a handful of samples. In that case, by the property of the normal distribution, we can use this test when the data are known to follow a normal distribution, and we know the population standard deviation. In this case, the sampling distribution satisfies:

$$\frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}}\sim Z$$

When sample size $n$ is large ($n \geq 30$), by the central limit theorem and weak law of large numbers, we can still use this test without knowing the population standard deviation. In this case, the sampling distribution satisfies:

$$\frac{\bar{x}-\mu_0}{s/\sqrt{n}}\approx Z$$

### A. Hypothesis

In the one sample z test, two hypotheses are formulated:

- **Null Hypothesis ($H_0$)**: States that the mean of the sample ($\bar{x}$) is equal to the hypothesized population mean ($\mu_0$). It is formulated as: $H_0: \mu = \mu_0$

- **Alternative Hypothesis ($H_a$)**: States that the mean of the sample is different from the hypothesized population mean. Depending on the research question, this can be:
    - Two-tailed: $H_a: \mu \neq \mu_0$
    - One-tailed (greater): $H_a: \mu > \mu_0$
    - One-tailed (less): $H_a: \mu < \mu_0$

### B. Test Statistic

The test statistic for a one sample z test is calculated using the formula:

$$ z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}} $$

where $\bar{x}$ is the sample mean, $\mu_0$ is the population mean under the null hypothesis, $\sigma$ is the known population standard deviation, and $n$ is the sample size. This statistic follows a standard normal distribution (Z-distribution) under the null hypothesis.

When sample size $n$ is large ($n \geq 30$), we can substitute the sample standard deviation $s$ for $\sigma$:

$$ z = \frac{\bar{x} - \mu_0}{s / \sqrt{n}} $$

### C. Decision Rule

The decision to reject or retain the null hypothesis is based on the calculated z-value and the critical z-values associated with the desired significance level ($\alpha$). Commonly used significance levels are 0.05, 0.01, and 0.10.

- **Two-tailed test**: Reject $H_0$ if $|z| > z_{\alpha/2}$.
- **One-tailed test (greater)**: Reject $H_0$ if $z > z_{\alpha}$.
- **One-tailed test (less)**: Reject $H_0$ if $z < -z_{\alpha}$.

### D. P-value

The p-value provides a measure of the evidence against the null hypothesis:

- For a two-tailed test: $p\text{-value} = 2P(Z \geq |z|)$
- For a one-tailed test (greater): $p\text{-value} = P(Z \geq z)$
- For a one-tailed test (less): $p\text{-value} = P(Z \leq z)$

### E. Interpretation

- If the p-value $\leq \alpha$, there is significant evidence to reject the null hypothesis, indicating that the sample mean is statistically significantly different from the population mean.
- If the p-value $> \alpha$, there is not enough evidence to reject the null hypothesis.

### F. Examples

#### Example: One Sample z Test — Two Sided

$$H_0: \mu=50 \quad \text{vs} \quad H_1: \mu\neq50$$

Given: $n = 500$, $\bar{x} = 48$, $s = 20.3$.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def plot_z_statistic(statistic, ax, alternative='two-sided'):
    x = np.linspace(-4, 4, 100)
    y = stats.norm().pdf(x)
    ax.plot(x, y, '-k')

    if alternative == 'less':
        x_fill = np.linspace(-4, statistic, 100)
        y_fill = stats.norm().pdf(x_fill)
        ax.fill_between(x_fill, y_fill, color='r', alpha=0.2)
    elif alternative == 'greater':
        x_fill = np.linspace(statistic, 4, 100)
        y_fill = stats.norm().pdf(x_fill)
        ax.fill_between(x_fill, y_fill, color='r', alpha=0.2)
    elif alternative == 'two-sided':
        x_fill_left = np.linspace(-4, -abs(statistic), 100)
        y_fill_left = stats.norm().pdf(x_fill_left)
        ax.fill_between(x_fill_left, y_fill_left, color='r', alpha=0.2)
        x_fill_right = np.linspace(abs(statistic), 4, 100)
        y_fill_right = stats.norm().pdf(x_fill_right)
        ax.fill_between(x_fill_right, y_fill_right, color='r', alpha=0.2)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position("zero")
    ax.set_yticks([])

mu = 50
n = 500
x_bar = 48
s = 20.3

statistic = (x_bar - mu) / (s / np.sqrt(n))
p_value = stats.norm().sf(abs(statistic)) * 2

print(f"Statistic: {statistic:.4f}")
print(f"P-value  : {p_value:.4f}")

alpha = 0.05
if p_value <= alpha:
    print("Reject H0 (Choose H1)")
else:
    print("Fail to reject H0")

fig, ax = plt.subplots(figsize=(12, 3))
plot_z_statistic(statistic, ax=ax, alternative='two-sided')
plt.show()
```

#### Example: One Sample z Test — Less

$$H_0: \mu=50 \quad \text{vs} \quad H_1: \mu<50$$

Given: $n = 500$, $\bar{x} = 48$, $s = 20.3$.

```python
mu = 50
n = 500
x_bar = 48
s = 20.3

statistic = (x_bar - mu) / (s / np.sqrt(n))
p_value = stats.norm().cdf(statistic)

print(f"Statistic : {statistic:.4f}")
print(f"P-value   : {p_value:.4f}")

alpha = 0.05
if p_value <= alpha:
    print("We choose H1, or using statistician's jargon, reject H0")
else:
    print("We choose H0, or using statistician's jargon, fail to reject H0")

fig, ax = plt.subplots(figsize=(12, 3))
plot_z_statistic(statistic, ax=ax, alternative='less')
plt.show()
```

#### Example: One Sample z Test — Greater

$$H_0: \mu=50 \quad \text{vs} \quad H_1: \mu>50$$

Given: $n = 500$, $\bar{x} = 52$, $s = 20.3$.

```python
mu = 50
n = 500
x_bar = 52
s = 20.3

statistic = (x_bar - mu) / (s / np.sqrt(n))
p_value = stats.norm().sf(statistic)

print(f"Statistic : {statistic:.4f}")
print(f"P-value   : {p_value:.4f}")

alpha = 0.05
if p_value <= alpha:
    print("We choose H1, or using statistician's jargon, reject H0")
else:
    print("We choose H0, or using statistician's jargon, fail to reject H0")

fig, ax = plt.subplots(figsize=(12, 3))
plot_z_statistic(statistic, ax=ax, alternative='greater')
plt.show()
```

---

## 2. One Sample t Test

The one sample t test is a parametric statistical technique used to determine whether the mean of a single sample differs significantly from a known or hypothesized population mean when the population standard deviation is unknown and the sample size is relatively small. This test assumes that the population distribution is approximately normal.

### A. Hypothesis

- **Null Hypothesis ($H_0$)**: $H_0: \mu = \mu_0$
- **Alternative Hypothesis ($H_a$)**:
    - Two-tailed: $H_a: \mu \neq \mu_0$
    - One-tailed (greater): $H_a: \mu > \mu_0$
    - One-tailed (less): $H_a: \mu < \mu_0$

### B. Test Statistic

$$ t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}} $$

where $\bar{x}$ is the sample mean, $\mu_0$ is the population mean under the null hypothesis, $s$ is the sample standard deviation, and $n$ is the sample size. This statistic follows a t-distribution with $n - 1$ degrees of freedom.

### C. Decision Rule

- **Two-tailed test**: Reject $H_0$ if $|t| > t_{\alpha/2, n-1}$.
- **One-tailed test (greater)**: Reject $H_0$ if $t > t_{\alpha, n-1}$.
- **One-tailed test (less)**: Reject $H_0$ if $t < -t_{\alpha, n-1}$.

### D. P-value

- For a two-tailed test: $p\text{-value} = 2P(T \geq |t|)$
- For a one-tailed test (greater): $p\text{-value} = P(T \geq t)$
- For a one-tailed test (less): $p\text{-value} = P(T \leq t)$

### E. Interpretation

- If the p-value $\leq \alpha$, there is statistically significant evidence to reject the null hypothesis.
- If the p-value $> \alpha$, the evidence is insufficient to reject the null hypothesis.

### F. Examples

#### Example: t Statistic for Teacher's Experience

Rory suspects that teachers in his school district, on average, have less than five years of experience. He tests $H_0: \mu = 5$ vs $H_1: \mu < 5$. He collects a sample of 25 teachers and finds $\bar{x} = 4$ years, $s = 2$ years.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_t_statistic(statistic, df, ax, alternative='two-sided'):
    x = np.linspace(-4, 4, 100)
    y = stats.t(df).pdf(x)
    ax.plot(x, y, '-k')

    if alternative == 'less':
        x_fill = np.linspace(-4, statistic, 100)
        y_fill = stats.t(df).pdf(x_fill)
        ax.fill_between(x_fill, y_fill, color='k', alpha=0.2)
    elif alternative == 'greater':
        x_fill = np.linspace(statistic, 4, 100)
        y_fill = stats.t(df).pdf(x_fill)
        ax.fill_between(x_fill, y_fill, color='k', alpha=0.2)
    elif alternative == 'two-sided':
        x_fill_left = np.linspace(-4, -abs(statistic), 100)
        y_fill_left = stats.t(df).pdf(x_fill_left)
        ax.fill_between(x_fill_left, y_fill_left, color='k', alpha=0.2)
        x_fill_right = np.linspace(abs(statistic), 4, 100)
        y_fill_right = stats.t(df).pdf(x_fill_right)
        ax.fill_between(x_fill_right, y_fill_right, color='k', alpha=0.2)

    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position("zero")
    ax.set_yticks(())

mu_0 = 5
x_bar = 4
s = 2
n = 25

statistic = (x_bar - mu_0) / (s / np.sqrt(n))
df = n - 1
p_value = stats.t(df).cdf(statistic)

print(f"T-statistic: {statistic:.4f}")
print(f"P-value    : {p_value:.4f}")

fig, ax = plt.subplots(figsize=(12, 3))
plot_t_statistic(statistic, df=df, ax=ax, alternative='less')
plt.show()
```

#### Example: p-value of Miriam's Test

Miriam conducted a test with $H_0: \mu = 18$ vs $H_1: \mu < 18$. She used $n = 7$ observations and obtained $t = -1.9$.

```python
n = 7
df = n - 1
statistic = -1.9
p_value = stats.t(df).cdf(statistic)
print(f"{statistic = :.4f}")
print(f"{p_value = :.4f}")

fig, ax = plt.subplots(figsize=(12, 3))
plot_t_statistic(statistic, df=df, ax=ax, alternative='less')
plt.show()
```

#### Example: p-value of Caterina's Test

Caterina conducted a test with $H_0: \mu = 0$ vs $H_1: \mu \neq 0$. She used $n = 6$ observations and obtained $t = 2.75$.

```python
n = 6
df = n - 1
statistic = 2.75
p_value = stats.t(df).sf(statistic) * 2
print(f"{statistic = :.4f}")
print(f"{p_value = :.4f}")

fig, ax = plt.subplots(figsize=(12, 3))
plot_t_statistic(statistic, df=df, ax=ax, alternative='two-sided')
plt.show()
```

#### Example: Jude's Automated Drink-Filling Machine

Jude tested $H_0: \mu = 530$ vs $H_1: \mu \neq 530$ with $n = 20$ drinks. He found $\bar{x} = 528$ mL, $s = 4$ mL, yielding $t = -2.236$ and $p \approx 0.038$. At $\alpha = 0.05$: since the p-value is smaller than the significance level, we reject $H_0$ and choose $H_1$.

#### Example: One Sample t Test — Toy Example

$$H_0 : \mu = 70 \quad\text{vs}\quad H_1: \mu > 70$$

```python
samples = np.array([78, 83, 68, 72, 88])

n = samples.shape[0]
df = n - 1
x_bar = samples.mean()
s = samples.std(ddof=1)
mu = 70

confidence_level = 0.95
alpha = 1 - confidence_level
t_score = (x_bar - mu) / (s / np.sqrt(n))
p_value = stats.t(df=df).sf(abs(t_score)) * 2

print(f"Test statistic (t-score): {t_score:.4f}")
print(f"p-value                 : {p_value:.4f}")

if p_value <= alpha:
    print("Reject H_0: Sufficient evidence to support the alternative hypothesis.")
else:
    print("Fail to reject H_0: Insufficient evidence to support the alternative hypothesis.")

fig, ax = plt.subplots(figsize=(12, 3))
plot_t_statistic(t_score, df=df, ax=ax, alternative='greater')
ax.legend(["t-distribution", f"t statistic = {t_score:.4f}"])
plt.show()
```

#### Example: Milk

A plant's milk containers are labeled 128 ounces. A sample of 12 containers has $\bar{x} = 127.2$ oz, $s = 2.1$ oz. Test $H_0: \mu = 128$ vs $H_1: \mu < 128$.

```python
mu_0 = 128
x_bar = 127.2
s = 2.1
n = 12
statistic = (x_bar - mu_0) / (s / np.sqrt(n))
df = n - 1
p_value = stats.t(df).cdf(statistic)

print(f"{statistic = :.4f}")
print(f"{p_value = :.4f}")

alpha = 0.05
if p_value <= alpha:
    print("Reject H_0 in favor of H_1")
else:
    print("Fail to reject H_0")

fig, ax = plt.subplots(figsize=(12, 3))
plot_t_statistic(statistic, df=df, ax=ax, alternative='less')
plt.show()
```

---

## 3. One Sample Proportion Test

The one sample proportion test (z test for a single proportion) is a statistical method used to determine whether the proportion of a particular characteristic in a sample represents a statistically significant difference from a hypothesized proportion in the population. We use this test when the variable of interest is categorical (e.g., success/failure, yes/no).

### A. Hypothesis

- **Null Hypothesis ($H_0$)**: $H_0: p = p_0$
- **Alternative Hypothesis ($H_a$)**:
    - Two-tailed: $H_a: p \neq p_0$
    - One-tailed (greater): $H_a: p > p_0$
    - One-tailed (less): $H_a: p < p_0$

### B. Test Statistic

$$ z = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0 (1 - p_0)}{n}}} $$

where $\hat{p}$ is the sample proportion, $p_0$ is the hypothesized population proportion, and $n$ is the total number of observations. This z-statistic follows a standard normal distribution if $np_0 \geq 5$ and $n(1 - p_0) \geq 5$.

### C. Decision Rule

- **Two-tailed test**: Reject $H_0$ if $|z| > z_{\alpha/2}$.
- **One-tailed test (greater)**: Reject $H_0$ if $z > z_{\alpha}$.
- **One-tailed test (less)**: Reject $H_0$ if $z < -z_{\alpha}$.

### D. P-value

- For a two-tailed test: $p\text{-value} = 2P(Z \geq |z|)$
- For a one-tailed test (greater): $p\text{-value} = P(Z \geq z)$
- For a one-tailed test (less): $p\text{-value} = P(Z \leq z)$

### E. Interpretation

- If the p-value $\leq \alpha$, the evidence is strong enough to reject the null hypothesis.
- If the p-value $> \alpha$, there is insufficient evidence to reject the null hypothesis.

### F. Examples

#### Example: Proportion of Labor Union

Ariel wants to test whether 49% of teachers in her state are union members.

$$H_0: p = 0.49 \quad \text{vs} \quad H_1: p \neq 0.49$$

#### Example: Proportion of California Homes with Internet

About 90% of California homes have internet access. Market researchers test if that proportion is now higher from a sample of 1,000 homes where 920 (92%) have access.

$$H_0: p = 0.90 \quad \text{vs} \quad H_1: p > 0.90$$

#### Example: Unemployment Rate — Test Statistic

The mayor tests $H_0: p = 0.08$ vs $H_1: p \neq 0.08$ with a sample of 200 residents, 22 unemployed.

```python
p_hat = 22 / 200
p = 0.08
n = 200

statistic = (p_hat - p) / np.sqrt(p * (1 - p) / n)
p_value = stats.norm().sf(abs(statistic)) * 2

print(f"Statistic: {statistic:.4f}")
print(f"P-value : {p_value:.4f}")
```

#### Example: Multilingual People

Fay tests $H_0: p = 0.26$ vs $H_1: p > 0.26$. She found 40 of 120 people could speak more than one language.

```python
p_hat = 40 / 120
p = 0.26
n = 120

statistic = (p_hat - p) / np.sqrt(p * (1 - p) / n)
p_value = stats.norm().sf(statistic)

print(f"Statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")
```

#### Example: Tax Increase for Public School Funding

Researchers test $H_0: p = 0.50$ vs $H_1: p > 0.50$ with 113 out of 200 supporting.

```python
k = 113
n = 200
p_0 = 0.5

# Exact test (Binomial)
result = stats.binomtest(k, n, p=p_0, alternative="greater")
print(f"Exact P-value: {result.pvalue:.4f}")

# Approximate test (Normal)
p_hat = k / n
approx_statistic = (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / n)
approx_p_value = stats.norm().sf(approx_statistic)
print(f"Approximate Statistic: {approx_statistic:.4f}")
print(f"Approximate P-value: {approx_p_value:.4f}")
```

#### Example: Free Video Rental Vouchers

Students test $H_0: p = 0.20$ vs $H_1: p < 0.20$. Found 11 vouchers in 65 boxes.

```python
k = 11
n = 65
p_0 = 0.2

# Exact test
result = stats.binomtest(k, n, p=p_0, alternative="less")
print(f"Exact P-value: {result.pvalue:.4f}")

# Approximate test
p_hat = k / n
approx_statistic = (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / n)
approx_p_value = stats.norm().cdf(approx_statistic)
print(f"Approximate Statistic: {approx_statistic:.4f}")
print(f"Approximate P-value: {approx_p_value:.4f}")
```

---

## 4. Alternatives to the One Sample t Test

There are non-parametric alternatives to the one-sample t-test when the assumptions of normality or other parametric conditions are violated.

| **Test** | **Assumption** | **When to Use** | **Strengths** |
|---|---|---|---|
| **Wilcoxon Signed-Rank** | Symmetry of differences around median | Most common non-parametric alternative | Uses ranks, more powerful than Sign Test |
| **Sign Test** | None (only signs matter) | Data is ordinal or skewed | Simple and robust, but less powerful |
| **Bootstrap** | None | Small sample, need confidence intervals | Flexible, computationally intensive |
| **Permutation Test** | None | No distribution assumptions | Robust and versatile, needs computation |
| **Mood's Median Test** | None | Median comparisons in non-normal data | Robust for outliers |

### A. Wilcoxon Signed-Rank Test

The Single-Sample Wilcoxon Signed-Rank Test is a non-parametric alternative to the one-sample t-test, used to assess whether the median of a single sample differs from a hypothesized value.

**Test Procedure:**

1. Compute differences: $d_i = X_i - m_0$. Ignore $d_i = 0$.
2. Rank the absolute differences $|d_i|$ in ascending order.
3. Assign the sign of each difference to its corresponding rank.
4. Calculate $W^+$ (sum of positive ranks) and $W^-$ (sum of negative ranks).
5. Test statistic: $W = \min(W^+, W^-)$.
6. For larger samples, use the normal approximation:

$$Z = \frac{W - \frac{n(n+1)}{4}}{\sqrt{\frac{n(n+1)(2n+1)}{24}}}$$

```python
import numpy as np
from scipy.stats import wilcoxon

task_times = np.array([16, 14, 15, 17, 13, 18, 14, 16, 15, 19])
hypothetical_median = 15

differences = task_times - hypothetical_median
stat, p_value = wilcoxon(differences)

print(f"Test Statistic: {stat}")
print(f"P-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The median is significantly different.")
else:
    print("Fail to reject the null hypothesis: No significant difference.")
```

### B. Sign Test

The Sign Test evaluates whether the median of a single sample is equal to a specified value. It considers only the direction (positive or negative) of the differences, ignoring their magnitude.

```python
from scipy.stats import binom
import numpy as np

def sign_test(data, median_hypothesis):
    differences = np.array(data) - median_hypothesis
    n_plus = np.sum(differences > 0)
    n_minus = np.sum(differences < 0)
    ties = np.sum(differences == 0)
    W = min(n_plus, n_minus)
    n = n_plus + n_minus
    p_value = 2 * binom.cdf(W, n, 0.5)
    return p_value, n_plus, n_minus, ties

data = [9.8, 10.1, 9.9, 10.2, 10.4, 10.3, 10.0, 9.7, 10.5, 9.6, 10.0, 9.9, 10.2, 9.8, 10.1]
p_value, n_plus, n_minus, ties = sign_test(data, 10)
print(f"P-value: {p_value}, n+: {n_plus}, n-: {n_minus}, Ties: {ties}")
```

### C. Bootstrap Method

The Bootstrap Method estimates the distribution of a statistic by repeatedly resampling with replacement from the observed data. Unlike parametric methods, it makes no assumptions about the underlying distribution.

```python
import numpy as np

def bootstrap_confidence_interval(data, statistic=np.mean, n_resamples=10000, ci=95):
    bootstrap_distribution = np.array([
        statistic(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_resamples)
    ])
    lower_bound = np.percentile(bootstrap_distribution, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrap_distribution, 100 - (100 - ci) / 2)
    return lower_bound, upper_bound, bootstrap_distribution

data = [5, 7, 9, 12, 15]
lower, upper, _ = bootstrap_confidence_interval(data)
print(f"95% CI for the Mean: ({lower:.2f}, {upper:.2f})")
```

### D. Permutation Test

The Permutation Test evaluates whether an observed test statistic is consistent with the null hypothesis by comparing it to a distribution generated by all possible rearrangements of the data.

```python
import numpy as np

def permutation_test(group_a, group_b, n_permutations=10000):
    combined = np.concatenate([group_a, group_b])
    observed_diff = np.mean(group_a) - np.mean(group_b)
    perm_differences = []
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_diff = np.mean(combined[:len(group_a)]) - np.mean(combined[len(group_a):])
        perm_differences.append(perm_diff)
    perm_differences = np.array(perm_differences)
    p_value = np.mean(np.abs(perm_differences) >= np.abs(observed_diff))
    return p_value, observed_diff, perm_differences

group_a = [8, 7, 9, 10, 6]
group_b = [5, 6, 4, 3, 7]
p_value, observed_diff, _ = permutation_test(group_a, group_b)
print(f"Observed Difference: {observed_diff:.2f}, P-value: {p_value:.4f}")
```

#### Comparison of Bootstrap and Permutation Tests

| Feature | **Bootstrap** | **Permutation Test** |
|---|---|---|
| **Primary Purpose** | Confidence intervals, variability estimation | Hypothesis testing |
| **Resampling** | With replacement | Without replacement |
| **Key Output** | Confidence intervals | p-value for hypothesis testing |
| **Assumptions** | Data is representative of population | Exchangeability under null hypothesis |
| **Flexibility** | Very flexible for complex statistics | Focuses on simpler tests |

### E. Mood's Median Test

Mood's Median Test is a non-parametric test used to compare the medians of two or more groups, particularly useful when data contains outliers.

```python
import numpy as np
from scipy.stats import chi2_contingency

def moods_median_test(*groups):
    combined_data = np.concatenate(groups)
    overall_median = np.median(combined_data)
    contingency_table = []
    for group in groups:
        above = np.sum(group > overall_median)
        below = np.sum(group < overall_median)
        contingency_table.append([above, below])
    contingency_table = np.array(contingency_table).T
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
    return chi2_stat, p_value, contingency_table

group_a = np.array([50, 55, 60, 65, 70])
group_b = np.array([45, 50, 55, 60, 65])
chi2_stat, p_value, table = moods_median_test(group_a, group_b)
print(f"Chi-Square: {chi2_stat:.4f}, P-value: {p_value:.4f}")
```
