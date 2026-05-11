# Paired t-Test for mu_D

## Paired-Sample t Test

The paired sample t-test, also known as the dependent sample t-test or the matched pairs t-test, is a statistical procedure used to compare two population means where you have two samples in which observations in one sample can be paired with observations in the other sample. Common scenarios include case-control studies, repeated measures, and experiments where individuals are measured before and after a treatment.

### A. Hypothesis

The hypotheses focus on the differences between the paired observations, rather than on the individual values:

- **Null Hypothesis ($H_0$)**: The mean difference between the paired observations is zero.

  $$ H_0: \mu_d = 0 $$

- **Alternative Hypothesis ($H_a$)**: The mean difference between the paired observations is not zero:
    - Two-tailed: $H_a: \mu_d \neq 0$
    - One-tailed (greater): $H_a: \mu_d > 0$
    - One-tailed (less): $H_a: \mu_d < 0$

### B. Test Statistic

The test statistic for the paired sample t-test is computed as:

$$ t = \frac{\bar{d}}{s_d / \sqrt{n}} $$

where $\bar{d}$ is the mean of the differences of the paired samples, $s_d$ is the standard deviation of these differences, and $n$ is the number of pairs. This t-statistic follows a t-distribution with $n - 1$ degrees of freedom.

### C. Decision Rule

- **Two-tailed test**: Reject $H_0$ if $|t| > t_{\alpha/2, n-1}$.
- **One-tailed test (greater)**: Reject $H_0$ if $t > t_{\alpha, n-1}$.
- **One-tailed test (less)**: Reject $H_0$ if $t < -t_{\alpha, n-1}$.

### D. P-value

- For a two-tailed test: $p\text{-value} = 2P(T \geq |t|)$
- For a one-tailed test: $p\text{-value} = P(T \geq t)$ or $p\text{-value} = P(T \leq t)$, depending on the direction.

### E. Interpretation

- If the p-value $\leq \alpha$, there is statistically significant evidence to reject the null hypothesis, indicating a significant difference in the mean of the paired differences.
- If the p-value $> \alpha$, there is insufficient evidence to reject the null hypothesis.

---

## Examples

### Example: Running Shoes

The manager of the Olympic running team suspects that Harpo's shoes may lead to lower running times than Zeppo's shoes. Each of six runners runs two laps (one with each shoe), with coin flip determining order.

**Test**: Paired Sample t Test

$$H_0 : \mu_{\text{Harpo}-\text{Zeppo}} = 0 \quad\text{vs}\quad H_1: \mu_{\text{Harpo}-\text{Zeppo}} < 0$$

### Example: Pre/Post Test Scores

| Student | Post | Pre | Difference |
|:---:|:---:|:---:|:---:|
| 1 | 93 | 76 | 17 |
| 2 | 70 | 72 | -2 |
| 3 | 81 | 75 | 6 |
| 4 | 65 | 68 | -3 |
| 5 | 79 | 65 | 14 |
| 6 | 54 | 54 | 0 |
| 7 | 94 | 88 | 6 |
| 8 | 91 | 81 | 10 |
| 9 | 77 | 65 | 12 |
| 10 | 65 | 57 | 8 |
| 11 | 95 | 86 | 9 |
| 12 | 89 | 87 | 12 |
| 13 | 78 | 78 | 0 |
| 14 | 80 | 77 | 17 |
| 15 | 76 | 76 | 0 |

**Hypotheses:**

$$H_0 : \mu_{\text{Post}-\text{Pre}} = 0 \quad\text{vs}\quad H_1: \mu_{\text{Post}-\text{Pre}} \neq 0$$

```python
import numpy as np
import scipy.stats as stats

data = np.array([[93,76], [70,72], [81,75], [65,68], [79,65],
                 [54,54], [94,88], [91,81], [77,65], [65,57],
                 [95,86], [89,87], [78,78], [80,77], [76,76]])

difference = data[:,0] - data[:,1]
x_bar = difference.mean()
mu = 0
s = difference.std(ddof=1)
n = difference.shape[0]

t = (x_bar - mu) / (s / np.sqrt(n))
p_value = 2 * stats.t(n-1).cdf(-abs(t))

print(f"{t       = :.4f}")
print(f"{p_value = :.4f}")

# Verify with scipy
t2, p2 = stats.ttest_rel(data[:,0], data[:,1], alternative="two-sided")
print(f"\nscipy: t = {t2:.4f}, p = {p2:.4f}")
```

---

## Paired-Sample Wilcoxon Signed-Rank Test

The Paired-Sample Wilcoxon Signed-Rank Test is a non-parametric test used to determine whether the median difference between paired observations is significantly different from zero. It serves as a non-parametric alternative to the paired t-test when the data do not meet the normality assumption.

### Key Features

- **Null Hypothesis ($H_0$)**: The median of the differences between the paired samples is zero.
- **Data Requirements**: Data must be paired and continuous or ordinal. The differences between pairs should be symmetrically distributed.

### Test Procedure

1. Calculate differences: $d_i = X_i - Y_i$. Ignore $d_i = 0$.
2. Rank the absolute differences $|d_i|$ in ascending order.
3. Assign the sign of each difference to its corresponding rank.
4. Compute $W^+ = \sum(\text{positive ranks})$ and $W^- = \sum(\text{negative ranks})$.
5. Test statistic: $W = \min(W^+, W^-)$.
6. For larger samples ($n > 20$), use the normal approximation:

$$Z = \frac{W - \frac{n(n+1)}{4}}{\sqrt{\frac{n(n+1)(2n+1)}{24}}}$$

### Example: Teaching Method Improvement

A researcher tests whether a new teaching method improves test scores for 10 students.

- Before: $[70, 68, 75, 80, 72, 74, 69, 77, 73, 76]$
- After: $[72, 69, 78, 85, 75, 76, 70, 79, 74, 80]$

All differences are positive ($d = [2, 1, 3, 5, 3, 2, 1, 2, 1, 4]$), so $W^+ = 50.5$ and $W^- = 0$, giving $W = 0$. Since $W = 0 < 8$ (critical value for $n=10$, $\alpha=0.05$), we reject $H_0$.

```python
import numpy as np
from scipy.stats import wilcoxon

before = np.array([70, 68, 75, 80, 72, 74, 69, 77, 73, 76])
after = np.array([72, 69, 78, 85, 75, 76, 70, 79, 74, 80])

stat, p_value = wilcoxon(after, before)

print(f"Test Statistic: {stat}")
print(f"P-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("Reject H0: Significant improvement in scores.")
else:
    print("Fail to reject H0: No significant improvement.")
```

### Advantages and Limitations

**Advantages:** Does not assume normality; robust to outliers.

**Limitations:** Assumes symmetry of the distribution of differences; less powerful than parametric tests when normality holds.

---

## Conclusion

The paired sample t-test is an essential tool for analyzing data where pairs of related or dependent samples are compared. It is particularly useful in before-and-after studies, or when the same subjects are exposed to two different conditions. This test helps to control for variability between subjects, thereby focusing more accurately on the effects of the treatments or conditions being tested. By accounting for the dependency in the paired samples, this test provides a more powerful and sensitive analysis than two independent sample tests when the paired design is appropriate.

## Exercises

**Exercise 1.**
Workout program: 10 participants measured before/after. Differences yield $\bar d = 1.7$, $s_d \approx 0.483$. Test at $\alpha = 0.05$.

??? success "Solution to Exercise 1"
    Paired sample → $t_9$.

    $t = 1.7/(0.483/\sqrt{10}) = 1.7/0.153 \approx 11.13$.

    Critical: $t_{0.05, 9} = 1.833$ (one-sided). $11.13 \gg 1.833$. **Reject overwhelmingly.** Workout reduces body fat significantly.

---

**Exercise 2.**
Diet plan: 12 participants, $\bar d = 7.5$ mg/dL reduction, $s_d \approx 2.61$. Test at $\alpha = 0.05$.

??? success "Solution to Exercise 2"
    $t = 7.5/(2.61/\sqrt{12}) = 7.5/0.754 \approx 9.95$.

    Critical: $t_{0.05, 11} = 1.796$. $9.95 \gg 1.796$. **Reject.** Diet reduces cholesterol.

---

**Exercise 3.**
**Why paired beats independent.** For Exercise 1, suppose ignoring pairing: treat 10 before-values and 10 after-values as independent samples. What changes?

??? success "Solution to Exercise 3"
    Before: mean 28.2, SD 2.66. After: mean 26.5, SD 2.42. Pooled SD ≈ 2.55.

    Independent $t$ = $(28.2 - 26.5)/\sqrt{2.55^2(1/10 + 1/10)} = 1.7/1.14 \approx 1.49$. Critical (one-sided df = 18): 1.734. **Fail to reject.**

    Paired analysis ($t \approx 11.1$) gives overwhelming significance; independent ($t \approx 1.5$) fails to detect.

    **Why:** the within-subject correlation is high (each subject has similar before/after percentages). Paired analysis exploits this; independent analysis treats it as noise.

    **Lesson:** always match analysis to design. Ignoring pairing wastes information.

---

**Exercise 4.**
**Conditions** for paired $t$-test.

??? success "Solution to Exercise 4"
    1. **Random pairs / matched subjects:** appropriate pairing (same subject, twins, matched controls).
    2. **Independence between pairs:** subjects do not influence each other.
    3. **Approximate normality of differences $D_i$:** check Q-Q plot of differences.

    Note: the differences must be normal, *not* the pre or post values individually. If differences are not normal: use Wilcoxon signed-rank test.

---

**Exercise 5.**
**Wilcoxon signed-rank** as nonparametric alternative. Briefly describe.

??? success "Solution to Exercise 5"
    For paired data:

    1. Compute differences $D_i$.
    2. Rank $|D_i|$ (ignore signs).
    3. Sum ranks of positive and negative differences separately.
    4. Test statistic: smaller of the two sums (or both compared to critical values).

    Under $H_0: \mathrm{median}(D) = 0$, the test has a known null distribution (tabulated for small $n$, normal approximation for large $n$).

    **Advantages:** doesn't assume normality. Robust to outliers.

    **Disadvantages:** less powerful than paired $t$-test when normality holds. Lower interpretability.

    Modern practice: paired $t$ for clean normal differences; Wilcoxon for skewed or outlier-prone.

---

**Exercise 6.**
**Power for paired** vs unpaired. If subjects are paired with correlation $\rho$, how much smaller is $n$ for paired analysis?

??? success "Solution to Exercise 6"
    Variance of paired difference: $\sigma_D^2 = 2\sigma^2(1 - \rho)$. Unpaired: $2\sigma^2$.

    Ratio: $\sigma_D^2/\sigma_{\text{unp}}^2 = 1 - \rho$.

    Required $n$ for same power scales linearly with variance:

    $n_{\text{paired}}/n_{\text{unp}} = 1 - \rho$.

    For $\rho = 0.5$: half the data needed for paired. For $\rho = 0.9$: 1/10 as much.

    **Practical implication:** when feasible, pairing saves substantial sample-size budget. Especially valuable for expensive studies (clinical trials).
