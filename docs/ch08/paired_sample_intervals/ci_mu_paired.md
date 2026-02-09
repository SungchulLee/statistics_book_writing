# CI for μ_D (Mean of Differences)

## Paired-Sample Confidence Interval

When two related measurements are taken on the same subjects — such as pre-test and post-test scores, or before-and-after measurements — we use paired data to analyze the difference between the two measurements. The paired difference confidence interval estimates the mean difference $\mu_d$ between these two related measurements.

### Formula

Let $d_i = X_{i,1} - X_{i,2}$ represent the difference between the two measurements for the $i$-th subject. The confidence interval for the mean of the paired differences is

$$
\bar{d} \pm t_{\alpha/2, \, n-1} \times \frac{s_d}{\sqrt{n}}
$$

where

- $\bar{d}$ is the mean of the paired differences,
- $s_d$ is the standard deviation of the paired differences,
- $n$ is the number of paired observations,
- $\alpha$ is the significance level ($\text{significance level} = 1 - \text{confidence level}$),
- $t_{\alpha/2, \, n-1}$ is the critical value from the $t$-distribution with $n-1$ degrees of freedom.

The formula is essentially the same as for a confidence interval for a single mean, but applied to the differences between pairs of measurements.

### Conditions for Validity

$$
\bar{x}_d \pm t_{\alpha/2,n-1}\frac{s_d}{\sqrt{n}}
\quad\text{if}\quad
\begin{cases}
n < 30 \text{ (CLT does not apply)} \\
\text{population of differences is normal} \\
n \le 0.1N \text{ (IID)}
\end{cases}
$$

For large $n$ ($\ge 30$), the normality condition is less strict due to the Central Limit Theorem. The z-interval variants apply analogously:

$$
\bar{x}_d \pm z_{\alpha/2}\frac{\sigma_d}{\sqrt{n}} \quad (\sigma_d \text{ known, large } n)
\qquad\text{or}\qquad
\bar{x}_d \pm z_{\alpha/2}\frac{s_d}{\sqrt{n}} \quad (s_d \text{ plug-in, large } n)
$$

### Python Code

```python
import numpy as np
import scipy.stats as stats

differences = np.array([5, 3, 4, -2, 0, 6, -1, 2, 3, 4])
n = len(differences)
confidence_level = 0.95

mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)
standard_error = std_diff / np.sqrt(n)

t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, n - 1)
margin_of_error = t_critical * standard_error

confidence_interval = (mean_diff - margin_of_error, mean_diff + margin_of_error)
print(f"{confidence_interval = }")
```

---

## Examples

### Example 1: Blood Pressure Before and After Treatment

A researcher measures the blood pressure of 10 patients before and after a treatment. The differences (before minus after) are:

$$
d = [5, 3, 4, -2, 0, 6, -1, 2, 3, 4]
$$

Construct a 95% confidence interval for the mean difference in blood pressure.

**Solution.**

$$
\bar{d} = \frac{5+3+4+(-2)+0+6+(-1)+2+3+4}{10} = \frac{24}{10} = 2.4
$$

$$
s_d \approx 2.17, \qquad df = 9, \qquad t_{0.025, 9} \approx 2.262
$$

$$
\text{SE} = \frac{2.17}{\sqrt{10}} \approx 0.686, \qquad \text{ME} = 2.262 \times 0.686 \approx 1.552
$$

$$
\boxed{(0.848,\ 3.952)}
$$

We are 95% confident that the true mean difference in blood pressure after the treatment is between 0.848 and 3.952.

### Example 2: Finger-Snapping Speed

Each of 5 participants snapped with both dominant and non-dominant hands for 10 seconds. The order was randomized by coin toss.

| Participant | Dominant | Non-Dominant | Difference |
|---|---|---|---|
| Jeff | 44 | 35 | 9 |
| David | 42 | 37 | 5 |
| Kim | 40 | 32 | 8 |
| Charlotte | 37 | 31 | 6 |
| Jake | 42 | 36 | 6 |

Construct and interpret a 95% confidence interval for the mean difference.

**Solution.**

$$
\bar{d} = \frac{9+5+8+6+6}{5} = 6.8
$$

$$
s_d \approx 1.643, \qquad n = 5, \qquad df = 4, \qquad t_{0.025,4} \approx 2.776
$$

$$
\text{SE} = \frac{1.643}{\sqrt{5}} \approx 0.735, \qquad \text{ME} = 2.776 \times 0.735 \approx 2.04
$$

$$
\boxed{(4.76,\ 8.84)}
$$

We are 95% confident that the true mean difference in snaps between the dominant and non-dominant hands lies within $(4.76, 8.84)$.

```python
import numpy as np
from scipy import stats

differences = np.array([9, 5, 8, 6, 6])
n = len(differences)
mean_diff = np.mean(differences)
std_diff = differences.std(ddof=1)
standard_error = std_diff / np.sqrt(n)

df = n - 1
confidence_level = 0.95
alpha = 1 - confidence_level
t_critical = stats.t(df).ppf(1 - alpha / 2)
margin_of_error = t_critical * standard_error

print(f"95% CI: ({mean_diff - margin_of_error:.2f}, {mean_diff + margin_of_error:.2f})")
print(f"95% CI: {mean_diff:.2f} ± {margin_of_error:.2f}")
```

### Example 3: Two Watches (Four Steps)

A running magazine reviewed watches A and B that use GPS to measure distance. Five runners each wore both watches simultaneously on a 10-km route.

| Runner | Watch A | Watch B | Difference (A−B) |
|---|---|---|---|
| 1 | 9.8 | 10.1 | −0.3 |
| 2 | 9.8 | 10.0 | −0.2 |
| 3 | 10.1 | 10.2 | −0.1 |
| 4 | 10.1 | 9.9 | 0.2 |
| 5 | 10.2 | 10.1 | 0.1 |

Construct and interpret a 95% confidence interval for the mean difference.

**Step 1: Calculate Differences.** $d = [-0.3, -0.2, -0.1, 0.2, 0.1]$.

**Step 2: Check Conditions.**

- **Simple Random Sample:** Satisfied (magazine selected random subscribers).
- **Independence:** Satisfied (at least 50 subscribers in population).
- **Normal Population Distribution:** Since $n = 5$ is small, we check: the differences are symmetric with no outliers, so proceeding is safe.

**Step 3: Construct Interval.**

```python
import numpy as np
from scipy import stats

watch_A = np.array([9.8, 9.8, 10.1, 10.1, 10.2])
watch_B = np.array([10.1, 10, 10.2, 9.9, 10.1])
d = watch_A - watch_B

d_bar = d.mean()
s = d.std(ddof=1)
n = d.shape[0]
df = n - 1

confidence_level = 0.95
alpha = 1 - confidence_level
t_star = stats.t(df).ppf(1 - alpha / 2)
margin_of_error = t_star * s / np.sqrt(n)
print(f"{confidence_level:.0%} CI: {d_bar:.4f} ± {margin_of_error:.4f}")
```

**Step 4: Interpret Interval.** With 95% confidence, the mean difference between the distances reported by the watches is likely to fall within the interval $(-0.32, 0.20)$ km. The interval includes zero, so there is no significant difference between the distances Watch A and Watch B reported.

---

## Simulation: Paired Mean CI Coverage

```python
#!/usr/bin/env python3
"""
Paired-sample mean CI simulation: t, z_known, z_plugin.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

rng_seed = None
n_simulations = 100
n = 12
mu_x, mu_y = 0.5, 0.0
sigma_x, sigma_y = 1.0, 1.2
rho = 0.6
alpha = 0.05
method = "t"  # 't' | 'z_known' | 'z_plugin'


def main():
    rng = np.random.default_rng(rng_seed)
    delta_true = mu_x - mu_y
    var_d_true = sigma_x**2 + sigma_y**2 - 2 * rho * sigma_x * sigma_y
    sigma_d_true = np.sqrt(max(var_d_true, 0.0))

    cov = rho * sigma_x * sigma_y
    Sigma = np.array([[sigma_x**2, cov], [cov, sigma_y**2]])
    L = np.linalg.cholesky(Sigma)

    df = n - 1
    t_star = t.ppf(1 - alpha / 2.0, df=df)
    z_star = norm.ppf(1 - alpha / 2.0)

    lowers = np.empty(n_simulations)
    uppers = np.empty(n_simulations)
    centers = np.empty(n_simulations)

    for i in range(n_simulations):
        z = rng.standard_normal(size=(2, n))
        xy = (L @ z).T
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
    n_fail = int((~covered).sum())
    coverage_pct = 100.0 * covered.mean()

    fig, ax = plt.subplots(figsize=(12, 12))
    for i in range(n_simulations):
        color = "k" if covered[i] else "r"
        ax.plot([lowers[i], uppers[i]], [i, i], lw=2, color=color)
        ax.plot(centers[i], i, marker="o", ms=3, color=color)

    ax.axvline(delta_true, linestyle="--", linewidth=1.5, color="r")
    ax.set_title(
        f"{n_simulations} Paired {method} CIs for μ_D | n={n}, ρ={rho:.2f}, "
        f"CL={int((1 - alpha) * 100)}% | Fail={n_fail} (Coverage ≈ {coverage_pct:.1f}%)")
    ax.set_yticks([])
    for sp in ["left", "right", "top"]:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("μ_D = μ_X − μ_Y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

---

## Exercise

### Exercise: 99% CI for Mean Difference (Paired Data)

A sample of 10 pairs yields $\bar{d} = 4.5$ and $s_d = 2.0$. Construct a 99% CI for the mean difference. Assume the difference distribution is normal.

**Solution.** With $df = 9$ and 99% confidence, $t_{0.005, 9} \approx 3.2498$.

$$
\text{SE} = \frac{2.0}{\sqrt{10}} \approx 0.6325, \qquad \text{ME} = 3.2498 \times 0.6325 \approx 2.056
$$

$$
\boxed{(2.444,\ 6.556)}
$$

```python
import numpy as np
from scipy import stats

confidence_level = 0.99
alpha = 1 - confidence_level
n = 10
df = n - 1

t_star = stats.t(df).ppf(1 - alpha / 2)
d_bar = 4.5
s = 2

left = d_bar - t_star * s / np.sqrt(n)
right = d_bar + t_star * s / np.sqrt(n)
print(f"({left:.3f}, {right:.3f})")
```
