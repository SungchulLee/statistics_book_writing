# CI for μ

## One-Sample z Confidence Interval

In numerous real-world applications, such as in business, healthcare, and education, estimating the population mean $\mu$ from a random sample is often essential. When the population variance is known, we can utilize the standard normal distribution to construct a confidence interval.

### Formula

If we know the population variance $\sigma^2$, the confidence interval for the population mean $\mu$ based on a sample of size $n$ is

$$
\bar{X} \pm z_{\alpha/2} \times \frac{\sigma}{\sqrt{n}}
$$

where

- $\bar{X}$ is the sample mean,
- $\alpha$ is the significance level ($\text{significance level} = 1 - \text{confidence level}$),
- $z_{\alpha/2}$ is the critical value from the standard normal distribution, satisfying $P(Z > z_{\alpha/2}) = \alpha/2$,
- $\sigma$ is the known population standard deviation,
- $n$ is the sample size.

The quantity $\sigma / \sqrt{n}$ is the standard error of the sample mean.

### Conditions for Validity

$$
\bar{x}\pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}
\quad\text{if}\quad
\begin{cases}
n \text{ is large, e.g., } n \ge 30, \text{ so that CLT approximation works} \\
n \text{ is small relative to } N, \text{ e.g., } n \le 0.1N, \text{ so that IID approximation works}
\end{cases}
$$

When $\sigma$ is unknown and $n$ is large, using the sample standard deviation $s$ in place of $\sigma$ is justified by the Law of Large Numbers:

$$
\bar{x}\pm z_{\alpha/2}\frac{s}{\sqrt{n}}
\quad\text{if}\quad
\begin{cases}
n \ge 30 \text{ (CLT)} \\
n \ge 30 \text{ (LLN for } s \approx \sigma\text{)} \\
n \le 0.1N \text{ (IID)}
\end{cases}
$$

### Python Code

```python
import scipy.stats as stats
import numpy as np

# Given data
n = 40
sample_mean = 85
sigma = 12  # known population standard deviation
confidence_level = 0.95

# Critical value
z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)

# Standard error and margin of error
standard_error = sigma / np.sqrt(n)
margin_of_error = z_critical * standard_error

# Confidence interval
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print(f"{confidence_interval = }")
```

---

## One-Sample t Confidence Interval

In practice, the population variance $\sigma^2$ is often unknown. When this is the case, we estimate the variance using the sample variance $s^2$, which introduces additional uncertainty. To account for this, we use the $t$-distribution instead of the normal distribution.

### Formula

The confidence interval for the population mean $\mu$ when the population variance is unknown is given by

$$
\bar{X} \pm t_{\alpha/2, \, n-1} \times \frac{s}{\sqrt{n}}
$$

where

- $\bar{X}$ is the sample mean,
- $\alpha$ is the significance level ($\text{significance level} = 1 - \text{confidence level}$),
- $n - 1$ is the degree of freedom of the $t$-distribution,
- $t_{\alpha/2, \, n-1}$ is the critical value from the $t$-distribution, satisfying $P(T > t_{\alpha/2, \, n-1}) = \alpha/2$,
- $s$ is the sample standard deviation,
- $n$ is the sample size.

### Conditions for Validity

$$
\bar{x}\pm t_{\alpha/2,n-1}\frac{s}{\sqrt{n}}
\quad\text{if}\quad
\begin{cases}
n \text{ is small, e.g., } n < 30, \text{ so that CLT approximation does not work} \\
\text{population distribution is normal, so that sampling distribution is known exactly} \\
n \le 0.1N \text{ (IID)}
\end{cases}
$$

### Python Code

```python
import scipy.stats as stats
import numpy as np

# Given data
n = 25
sample_mean = 50
sample_std = 8  # sample standard deviation
confidence_level = 0.95

# Critical value from the t-distribution
degrees_of_freedom = n - 1
t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, degrees_of_freedom)

# Standard error and margin of error
standard_error = sample_std / np.sqrt(n)
margin_of_error = t_critical * standard_error

# Confidence interval
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print(f"{confidence_interval = }")
```

---

## Examples

### Example 1: 95% CI for Population Mean (Known Variance)

Suppose we collect a random sample of size $n = 40$ from a population. The sample mean is $\bar{X} = 85$, and the known population standard deviation is $\sigma = 12$. Construct a 95% confidence interval for the population mean $\mu$.

**Solution.** For a 95% confidence level, the critical value $z_{\alpha/2}$ is approximately 1.96. Substituting:

$$
85 \pm 1.96 \times \frac{12}{\sqrt{40}}
$$

Standard error: $\text{SE} = 12 / \sqrt{40} \approx 1.8974$. Margin of error: $1.96 \times 1.8974 \approx 3.717$.

$$
\boxed{(81.283,\ 88.717)}
$$

We are 95% confident that the true population mean $\mu$ lies within $(81.283, 88.717)$.

### Example 2: 95% CI for Population Mean (Large Sample)

A random sample of 100 adult males yields a mean height of 175 cm and a standard deviation of 6 cm. Calculate the 95% confidence interval for the population mean height.

**Solution.** Since $n = 100 \ge 30$, we use the standard normal distribution with the sample standard deviation.

$$
\text{SE} = \frac{6}{\sqrt{100}} = 0.6, \qquad \text{ME} = 1.96 \times 0.6 = 1.176
$$

$$
\boxed{(173.82,\ 176.18) \text{ cm}}
$$

```python
import numpy as np
import scipy.stats as stats

n = 100
x_bar = 175
s = 6
confidence_level = 0.95
alpha = 1 - confidence_level

z_star = stats.norm().ppf(1 - alpha / 2)
standard_error = s / np.sqrt(n)
margin_of_error = z_star * standard_error

print(f"{confidence_level:.0%} confidence interval: {x_bar} ± {margin_of_error:.2f}")
```

### Example 3: Sample Size Determination (Astronomer)

An astronomer measures the distance to a distant star. Measurements are iid with mean $d$ (the actual distance) and variance 4 light-years. How many measurements should he take so that his estimate is accurate within $\pm 0.5$ light-year with 95% confidence?

**Solution.** We need

$$
1.96 \sqrt{\frac{4}{n}} \leq 0.5
$$

Solving for $n$:

$$
n \geq \frac{4 \times 1.96^2}{0.5^2} = 61.4656
$$

Since $n$ must be an integer, at least **62 measurements** are required.

### Example 4: 95% CI for Population Mean (Unknown Variance, Small Sample)

A random sample of size $n = 25$ yields $\bar{X} = 50$ and $s = 8$. Construct a 95% confidence interval for $\mu$.

**Solution.** With $df = 24$ and 95% confidence, $t_{\alpha/2, 24} \approx 2.064$.

$$
\text{SE} = \frac{8}{\sqrt{25}} = 1.6, \qquad \text{ME} = 2.064 \times 1.6 \approx 3.302
$$

$$
\boxed{(46.698,\ 53.302)}
$$

### Example 5: Computation of $t_*$

What is the critical value $t_*$ for a 98% confidence interval with $n = 15$ observations?

**Solution.**

```python
import scipy.stats as stats

confidence_level = 0.98
alpha = 1 - confidence_level
n = 15
df = n - 1

t_star = stats.t(df=df).ppf(1 - alpha / 2)
print(f"{t_star = :.4f}")
```

### Example 6: Painting Thickness

Felix randomly selected 50 points on a car part and measured coating thickness. The sample yielded $\bar{x} = 148$ microns and $s = 3.3$ microns. He constructed a 95% confidence interval of $(147.1, 148.9)$ microns. Is it plausible for the average thickness to agree with the target value of 150 microns?

**Solution.** No, since the confidence interval $(147.1, 148.9)$ does not include the target thickness of 150 microns. The data provide evidence that the mean thickness significantly deviates from the target.

---

## Exercises

### Exercise: 95% CI for Mean Amount of Liquid

Quality control specialists sample 50 bottles from a batch. The sample yields a mean of 503 mL and a standard deviation of 5 mL. Construct a 95% confidence interval for the mean amount of liquid.

**Solution.**

$$
\text{SE} = \frac{5}{\sqrt{50}} \approx 0.707, \qquad \text{ME} = 1.96 \times 0.707 \approx 1.386
$$

$$
\boxed{(501.61,\ 504.39) \text{ mL}}
$$

Since the target of 500 mL is not in the interval, there is evidence that the batch mean deviates from the target.

### Exercise: Smallest Sample Size with Given Margin of Error

Nadia must determine the minimum sample size for a margin of error of 10 km at 90% confidence. The estimated standard deviation is 15 km.

**Solution.** For 90% confidence, $z_* \approx 1.645$.

$$
n = \left\lceil\left(\frac{1.645 \times 15}{10}\right)^2\right\rceil = \left\lceil 6.09\right\rceil = 7
$$

```python
from math import ceil
from scipy import stats

confidence_level = 0.90
alpha = 1 - confidence_level
z_star = stats.norm().ppf(1 - alpha / 2)
sigma = 15
max_margin_of_error = 10

min_sample_size = ceil((z_star * sigma / max_margin_of_error) ** 2)
print(f"{min_sample_size = }")
```

### Exercise: Confidence Interval for Population Mean (t-interval)

A sample of 25 bottles of soda has a mean volume of 500 ml and a sample standard deviation of 10 ml. Construct a 95% confidence interval.

**Solution.** With $df = 24$, $t_{0.025, 24} \approx 2.064$.

$$
\text{ME} = 2.064 \times \frac{10}{\sqrt{25}} = 2.064 \times 2 = 4.128
$$

$$
\boxed{(495.872,\ 504.128)}
$$

### Exercise: One-Sample z CI for Exam Scores

A random sample of 36 students' scores yields a mean of 78 with known $\sigma = 12$. Construct a 95% CI.

**Solution.** $\text{ME} = 1.96 \times 12/\sqrt{36} = 1.96 \times 2 = 3.92$.

$$
\boxed{(74.08,\ 81.92)}
$$

### Exercise: Sample Size Determination

Estimate a population mean with margin of error 5 at 95% confidence, given $\sigma = 20$.

**Solution.**

$$
n = \left(\frac{1.96 \times 20}{5}\right)^2 = (7.84)^2 = 61.47 \implies \boxed{n = 62}
$$

### Exercise: Interpretation

A 90% confidence interval for the mean weight is (72, 78). Does this mean the probability of the true mean being between 72 and 78 is 90%?

**Solution.** No. The 90% confidence interval means that if we were to take many samples and compute a 90% CI for each, approximately 90% of those intervals would contain the true population mean. However, the true population mean is a fixed (unknown) value — it either lies within (72, 78) or it does not. The 90% refers to the confidence in the **procedure**, not the probability for this specific interval.

### Exercise: Wider Confidence Interval

Will a 99% confidence interval be wider or narrower than a 95% CI? Why?

**Solution.** A 99% CI will always be **wider** than a 95% CI. To be 99% confident that the true population mean is captured by the interval, we need to account for more variability by expanding the range. A higher confidence level requires a larger critical value, which increases the margin of error.
