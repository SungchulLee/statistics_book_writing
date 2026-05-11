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

**Exercise 1.**
74/100 vehicles pass. 95% CI for $p$.

??? success "Solution to Exercise 1"
    $\hat p = 0.74$. $\mathrm{SE} = \sqrt{0.74 \cdot 0.26/100} \approx 0.0439$. ME = $1.96 \cdot 0.0439 \approx 0.086$.

    CI: $(0.654, 0.826)$.

---

**Exercise 2.**
120/200 prefer A. 90% CI for $p$.

??? success "Solution to Exercise 2"
    $\hat p = 0.60$. $z_{0.05} = 1.645$. ME = $1.645 \sqrt{0.60 \cdot 0.40/200} \approx 0.057$.

    CI: $(0.543, 0.657)$.

---

**Exercise 3.**
280/500 voters support. (a) Wald 95% CI. (b) Wilson 95% CI. (c) Evidence $p > 0.5$?

??? success "Solution to Exercise 3"
    (a) $\hat p = 0.56$. ME = $1.96 \cdot 0.0222 = 0.0435$. Wald CI: $(0.517, 0.604)$.

    (b) Wilson CI: center $= (0.56 + 1.96^2/1000)/(1 + 1.96^2/500) \approx 0.555$. CI $\approx (0.517, 0.603)$. Almost identical to Wald — Wilson and Wald agree when $n$ large and $\hat p$ moderate.

    (c) Entire CI above 0.5: evidence at 5% level that majority support the measure.

---

**Exercise 4.**
**Wilson interval near boundary.** Compare Wald and Wilson CIs for $n = 20$, $X = 2$ (so $\hat p = 0.1$).

??? success "Solution to Exercise 4"
    Wald: ME $= 1.96 \sqrt{0.1 \cdot 0.9/20} \approx 0.131$. Wald CI: $(0.1 - 0.131, 0.1 + 0.131) = (-0.031, 0.231)$ — **extends below 0**.

    Wilson: center $\approx (0.1 + 1.96^2/40)/(1 + 1.96^2/20) \approx 0.196/1.192 \approx 0.164$. Half-width: $1.96\sqrt{0.1 \cdot 0.9/20 + 1.96^2/1600}/1.192 \approx 0.110$. CI $\approx (0.054, 0.274)$ — stays in $[0, 1]$.

    Wilson is far better near boundaries. Default to Wilson for binomial CIs.

---

**Exercise 5.**
**Sample size for proportion CI.** Find $n$ so that 95% CI has ME $\le 0.03$, regardless of $\hat p$.

??? success "Solution to Exercise 5"
    $\mathrm{ME} = 1.96\sqrt{\hat p(1-\hat p)/n} \le 0.03$.

    Worst case at $\hat p = 1/2$: $\hat p(1-\hat p) = 0.25$.

    $1.96 \sqrt{0.25/n} \le 0.03 \Rightarrow n \ge (1.96)^2 \cdot 0.25/(0.03)^2 = 0.9604/0.0009 \approx 1068$.

    **$n = 1068$** suffices for ME $\le 0.03$ at 95% confidence, *regardless* of true $p$. Origin of the "$n \approx 1000$" rule for polls.

    If $p$ is suspected near 0.1 or 0.9: $p(1-p) = 0.09$, requiring $n \ge 385$. Knowledge of $p$ approximate value can reduce required $n$ by 60%.

---

**Exercise 6.**
**Continuity correction.** When does it improve the CI? Apply to Exercise 1.

??? success "Solution to Exercise 6"
    Continuity correction adds $1/(2n)$ to the half-width when using the normal approximation to the binomial. For Exercise 1: ME with correction $\approx 0.086 + 1/200 = 0.091$.

    CI with correction: $(0.649, 0.831)$ vs without: $(0.654, 0.826)$. Slightly wider.

    **When to use:**

    - Small to moderate $n$ where continuity matters.
    - When tail probabilities matter (one-sided tests).
    - When the discreteness of the binomial is non-negligible.

    Modern practice: rather than correction, use exact methods (Clopper-Pearson) or Wilson interval. Continuity correction is a legacy of pre-computer-era approximations.
