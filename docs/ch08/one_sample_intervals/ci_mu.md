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

### Example 5: Computation of t*
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

**Exercise 1.**
50 bottles, $\bar X = 503$ mL, $s = 5$ mL. 95% CI for $\mu$.

??? success "Solution to Exercise 1"
    Large $n$, use $z$: $\mathrm{SE} = 5/\sqrt{50} \approx 0.707$. ME = $1.96 \cdot 0.707 \approx 1.39$.

    CI: $(501.6, 504.4)$ mL.

    Target 500 mL is outside the CI — evidence the batch mean exceeds target.

---

**Exercise 2.**
**Sample size planning.** Target ME = 10 km, $\sigma = 15$ km, 90% confidence. Required $n$?

??? success "Solution to Exercise 2"
    $z_{0.05} = 1.645$. $n = (1.645 \cdot 15/10)^2 = (2.47)^2 \approx 6.09$. Round up: $n = 7$.

    Very small — feasible. For higher confidence (95%), $z = 1.96$ and $n \approx 9$. For larger $\sigma$ (say 30): $n \approx 25$.

    Sample size scales as $\sigma^2$ and $z^2$ — quadratic in both.

---

**Exercise 3.**
**$t$-interval with small sample.** $n = 25$ bottles, $\bar X = 500$ mL, $s = 10$ mL. 95% CI.

??? success "Solution to Exercise 3"
    $\sigma$ unknown, use $t$: $t_{0.025, 24} = 2.064$. ME = $2.064 \cdot 10/\sqrt{25} = 4.13$.

    CI: $(495.87, 504.13)$.

    Note $t_{0.025, 24} > z_{0.025}$ — $t$ intervals are wider than $z$ intervals at the same confidence level to account for uncertainty in $s$.

---

**Exercise 4.**
**Exam scores with known $\sigma$.** $n = 36$, $\bar X = 78$, $\sigma = 12$. 95% CI.

??? success "Solution to Exercise 4"
    Known $\sigma$, use $z$: ME = $1.96 \cdot 12/\sqrt{36} = 1.96 \cdot 2 = 3.92$.

    CI: $(74.08, 81.92)$.

    With known $\sigma$, the $z$ interval is the same regardless of sample size beyond the SE formula. Cleanest scenario but rarely realistic (we usually don't know $\sigma$).

---

**Exercise 5.**
**Confidence level trade-off.** A 90% CI is $(72, 78)$. (a) Interpretation. (b) Why a 99% CI is wider.

??? success "Solution to Exercise 5"
    (a) Correct interpretation: "If we repeated this sampling and CI construction many times, ~90% of resulting intervals would contain the true mean." NOT "there is a 90% probability $\mu \in (72, 78)$" — the parameter is fixed.

    (b) 99% CI uses $z_{0.005} = 2.576$ vs 90% using $z_{0.05} = 1.645$. Ratio of widths: $2.576/1.645 \approx 1.57$. The 99% CI is 57% wider.

    Higher confidence requires accommodating more variability — wider interval. Lower confidence is tighter but less reliable. Trade-off: precision vs. assurance.

---

**Exercise 6.**
**Light bulbs.** $n = 25$, $\bar X = 1200$ h, $s = 150$ h. (a) 95% CI. (b) 99% CI. (c) $n$ for ME = 20 h.

??? success "Solution to Exercise 6"
    (a) $t_{0.025, 24} = 2.064$. ME = $2.064 \cdot 150/5 = 61.9$. CI: $(1138.1, 1261.9)$.

    (b) $t_{0.005, 24} = 2.797$. ME = $2.797 \cdot 30 = 83.9$. CI: $(1116.1, 1283.9)$. Wider as expected.

    (c) Using $z$-approximation (large $n$ will result): $n = (1.96 \cdot 150/20)^2 = (14.7)^2 \approx 216.1$. Round up: $n = 217$.

    Achieving ME of 20 requires nearly 9× more samples than the original $n = 25$. Reducing ME from $\sim 62$ to 20 requires $217/25 \approx 8.7\times$ more data — quadratic in ME reduction.
