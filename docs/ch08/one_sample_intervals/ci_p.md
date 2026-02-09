# CI for p

## One-Sample Proportion Confidence Interval

In many statistical problems, we are interested in estimating a population proportion $p$ — the fraction of individuals in a population that have a certain characteristic. For example, the proportion of voters who support a particular candidate or the proportion of defective items in a batch.

### Formula (Wald z-Interval)

The general form of a confidence interval for a population proportion $p$ is

$$
\hat{p} \pm z_{\alpha/2} \times \sqrt{\frac{\hat{p}(1 - \hat{p})}{n}}
$$

where

- $\hat{p}$ is the sample proportion,
- $\alpha$ is the significance level ($\text{significance level} = 1 - \text{confidence level}$),
- $z_{\alpha/2}$ is the critical value from the standard normal distribution, satisfying $P(Z > z_{\alpha/2}) = \alpha/2$,
- $n$ is the sample size,
- $\sqrt{\hat{p}(1 - \hat{p})/n}$ is the standard error of the sample proportion.

### Conditions for Validity

$$
\hat{p}\pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
\quad\text{if}\quad
\begin{cases}
n\hat{p}\ge 10 \text{ and } n(1-\hat{p})\ge 10 \text{ (CLT)} \\
n \ge 30 \text{ (LLN)} \\
n \le 0.1N \text{ (IID)}
\end{cases}
$$

The sample size $n$ must be large enough so that the sampling distribution of the proportion is approximately normal.

### Python Code

```python
import numpy as np
import scipy.stats as stats

n = 200          # sample size
x = 120          # number of successes
confidence_level = 0.95

p_hat = x / n
z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
standard_error = np.sqrt((p_hat * (1 - p_hat)) / n)
margin_of_error = z_critical * standard_error
confidence_interval = (p_hat - margin_of_error, p_hat + margin_of_error)

print(f"{confidence_interval = }")
```

---

## Alternatives to the Wald z-Interval

The Wald interval is named after [Abraham Wald](https://en.wikipedia.org/wiki/Abraham_Wald), who formalized this type of interval as a normal approximation. However, the **Wald interval performs poorly** when $n$ is small, $\hat{p}$ is near 0 or 1, or $n\hat{p}$ or $n(1-\hat{p}) < 10$. In these cases, the coverage probability can be much lower than the nominal level.

| Interval | Formula Type | Uses $z$? | Works Well When | Comments |
|---|---|---|---|---|
| **Wald (z)** | $\hat{p} \pm z\sqrt{\hat{p}(1-\hat{p})/n}$ | Yes | Large $n$ | Simple, but inaccurate for small samples |
| **Wilson score** | Derived from inverting z-test | Yes | Small–large $n$ | Much better coverage |
| **Agresti–Coull** | Adjusted Wald (add pseudo-observations) | Yes | Small–medium $n$ | Easy fix, near Wilson performance |
| **Clopper–Pearson** | Based on binomial | No | Small $n$ | Conservative but exact |

### Wilson Score Interval

The Wilson interval comes from *inverting* the z-test for proportions:

$$
\frac{(\hat{p} - p)^2}{p(1-p)/n} = z_{\alpha/2}^2
$$

Solving for $p$ gives:

$$
\text{CI} =
\frac{
\hat{p} + \frac{z^2}{2n} \pm
z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}
}{
1 + \frac{z^2}{n}
}
$$

The center of the interval is **not** $\hat{p}$, but a **shrunken value** toward 0.5:

$$
\tilde{p} = \frac{\hat{p} + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}
$$

The interval stays within $[0, 1]$ and performs much better for small or skewed samples, with nearly nominal coverage even for $n < 30$.

### Agresti–Coull Interval

Agresti and Coull observed that the Wilson formula can be approximated simply by adding "pseudo-observations." For a 95% confidence level ($z = 1.96 \approx 2$):

- Add 2 successes and 2 failures → effectively 4 extra observations.
- Use adjusted counts: $n' = n + 4$, $x' = x + 2$, $\tilde{p} = x'/n'$.
- Compute a Wald-style interval using the adjusted proportion:

$$
\tilde{p} \pm z_{\alpha/2} \sqrt{\frac{\tilde{p}(1 - \tilde{p})}{n'}}
$$

Coverage is very close to Wilson; easy to explain and compute by hand. Becomes identical to Wilson when $n$ is large.

### Comparison Summary

| Method | Centered at | Adjustment | Performance |
|---|---|---|---|
| **Wald (z)** | $\hat{p}$ | None | Poor for small/edge cases |
| **Wilson score** | Weighted avg of $\hat{p}$ and 0.5 | Shifts center & width | Excellent |
| **Agresti–Coull** | $(x+2)/(n+4)$ | Adds pseudo-data | Nearly as good as Wilson |

---

## Examples

### Example 1: 95% CI for Proportion of Voter Support

A random sample of 200 voters is taken, and 120 say they support a particular candidate. Construct a 95% CI for the true proportion.

**Solution.**

$$
\hat{p} = \frac{120}{200} = 0.60
$$

For 95% confidence, $z_{\alpha/2} \approx 1.96$.

$$
\text{SE} = \sqrt{\frac{0.60 \times 0.40}{200}} = \sqrt{0.0012} \approx 0.03464
$$

$$
\text{ME} = 1.96 \times 0.03464 \approx 0.0679
$$

$$
\boxed{(0.5321,\ 0.6679)}
$$

We are 95% confident that the true proportion of voters who support the candidate is between 0.5321 and 0.6679.

### Example 2: Sample Size for School Funding Survey

Della wants a margin of error smaller than $\pm 2\%$ at 95% confidence for a proportion. What minimum sample size is needed?

**Solution.** The worst-case standard error occurs at $\hat{p} = 0.5$ (maximizing $\hat{p}(1-\hat{p})$).

```python
import scipy.stats as stats
import numpy as np

confidence_level = 0.95
alpha = 1 - confidence_level
z_star = stats.norm().ppf(1 - alpha / 2)
margin_of_error_max = 0.02
p_max = 0.5

n = 1
while True:
    n += 1
    me = z_star * np.sqrt(p_max * (1 - p_max) / n)
    if me <= margin_of_error_max:
        break
print(f"{n = }")  # n = 2401
```

### Example 3: Female Artist's Songs (99% CI)

Della has over 500 songs. She randomly selects 50 songs and finds 20 are by a female artist. Construct a 99% CI.

**Solution.**

```python
import numpy as np
from scipy import stats

confidence_level = 0.99
alpha = 1 - confidence_level
p_hat = 20 / 50
n = 50

z_star = stats.norm().ppf(1 - alpha / 2)
margin_of_error = z_star * np.sqrt(p_hat * (1 - p_hat) / n)
print(f"{p_hat} ± {margin_of_error:.3f}")
# 0.4 ± 0.178
```

The 99% CI is approximately $(0.222, 0.578)$.

---

## Exercises

### Exercise: Confidence Interval for Pass Rate

A random sample of 100 vehicles is selected, and 74 pass the inspection. Construct a 95% confidence interval for the pass rate.

**Solution.**

$$
\hat{p} = \frac{74}{100} = 0.74, \qquad \text{SE} = \sqrt{\frac{0.74 \times 0.26}{100}} \approx 0.04386
$$

$$
\text{ME} = 1.96 \times 0.04386 \approx 0.086
$$

$$
\boxed{(0.6540,\ 0.8260)}
$$

We are 95% confident that the true pass rate lies between 65.40% and 82.60%.

### Exercise: 90% CI for Population Proportion

In a survey of 200 people, 120 prefer product A over product B. Construct a 90% confidence interval.

**Solution.**

$$
\hat{p} = 0.60, \qquad z_{0.05} \approx 1.645
$$

$$
\text{ME} = 1.645 \times \sqrt{\frac{0.60 \times 0.40}{200}} \approx 1.645 \times 0.03464 \approx 0.057
$$

$$
\boxed{(0.543,\ 0.657)}
$$
