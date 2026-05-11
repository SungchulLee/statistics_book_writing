# Paired Mean Test

## Overview

The paired t-test compares two related measurements taken on the same subjects (e.g., before vs. after a treatment). By computing the difference $D_i = X_i - Y_i$ for each pair, the problem reduces to a one-sample t-test on the differences. This approach eliminates subject-to-subject variability and increases statistical power compared to an independent two-sample test when pairing is meaningful.

## Test Formulation

**Hypotheses:** Let $\mu_D = E[D_i]$ be the population mean of the paired differences.

- Two-sided: $H_0\colon \mu_D = \mu_{D_0}$ vs $H_1\colon \mu_D \neq \mu_{D_0}$
- One-sided: $H_0\colon \mu_D = \mu_{D_0}$ vs $H_1\colon \mu_D > \mu_{D_0}$ (or $< \mu_{D_0}$)

Typically $\mu_{D_0} = 0$ (no difference).

**Test statistic:** Given $n$ pairs with mean difference $\bar{D}$ and standard deviation of differences $S_D$,

$$
T = \frac{\bar{D} - \mu_{D_0}}{S_D / \sqrt{n}} \sim t_{n-1}.
$$

## Code

```python
import math
from scipy.stats import t as tdist

def test_paired_mean(n, dbar, sd_d, mu_d0=0.0,
                     alt="two-sided", alpha=0.05):
    """
    Paired t-test: differences D = X - Y.
    Supply n, mean(D), sd(D).
    Returns (t, p, reject).
    """
    df = n - 1
    se = sd_d / math.sqrt(n)
    t = (dbar - mu_d0) / se
    if alt == "two-sided":
        p = 2 * min(tdist.cdf(t, df), 1 - tdist.cdf(t, df))
    elif alt == "less":
        p = tdist.cdf(t, df)
    else:
        p = 1 - tdist.cdf(t, df)
    return t, p, (p < alpha)
```

### Example

```python
t_stat, p, reject = test_paired_mean(
    n=12, dbar=0.4, sd_d=1.1, mu_d0=0.0, alt="less"
)
print("t:", t_stat, "p:", p, "reject:", reject)
```

### Interpretation

With $n=12$ pairs, $\bar{D}=0.4$, and $S_D=1.1$, the test statistic for $H_1\colon \mu_D < 0$ is

$$
T = \frac{0.4 - 0}{1.1/\sqrt{12}} = \frac{0.4}{0.3175} \approx 1.260.
$$

Since $T > 0$ and we are testing the left tail ($H_1\colon \mu_D < 0$), the p-value will be large and we fail to reject $H_0$.

## Exercises

**Exercise 1.** Ten patients have their blood pressure measured before and after a new drug. The differences (after $-$ before) are: $-7, -8, -8, -7, -7, -8, -4, -7, -6, -7$. Test whether the drug significantly lowers blood pressure at $\alpha = 0.05$.

??? success "Solution to Exercise 1"

    Compute: $\bar{D} = -6.9$, $S_D \approx 1.197$, $n = 10$.

    Test $H_0\colon \mu_D = 0$ vs $H_1\colon \mu_D < 0$:

    $$
    T = \frac{-6.9 - 0}{1.197/\sqrt{10}} = \frac{-6.9}{0.3785} \approx -18.23.
    $$

    With $\text{df} = 9$, $P(T_9 \leq -18.23) \approx 0$. We strongly reject $H_0$. The drug significantly lowers blood pressure. $\square$

---

**Exercise 2.** Explain why the paired t-test is more powerful than the independent two-sample t-test when pairing is appropriate.

??? success "Solution to Exercise 2"

    In an independent two-sample t-test, the variance of the difference in means is

    $$
    \text{Var}(\bar{X} - \bar{Y}) = \frac{\sigma_X^2}{n} + \frac{\sigma_Y^2}{n}.
    $$

    In the paired test, the variance of $\bar{D}$ is

    $$
    \text{Var}(\bar{D}) = \frac{\sigma_D^2}{n} = \frac{\sigma_X^2 + \sigma_Y^2 - 2\rho\,\sigma_X\sigma_Y}{n},
    $$

    where $\rho = \text{Corr}(X_i, Y_i)$. When the pairing induces positive correlation ($\rho > 0$), the variance of $\bar{D}$ is smaller, yielding a larger test statistic and more power. The reduction is proportional to $2\rho\sigma_X\sigma_Y/n$. $\square$

---

**Exercise 3.** Show that the paired t-test is algebraically equivalent to a one-sample t-test on the differences.

??? success "Solution to Exercise 3"

    Define $D_i = X_i - Y_i$ for $i = 1, \dots, n$. Then $\bar{D} = \bar{X} - \bar{Y}$ and

    $$
    S_D^2 = \frac{1}{n-1}\sum_{i=1}^n (D_i - \bar{D})^2.
    $$

    The paired t-statistic is

    $$
    T_{\text{paired}} = \frac{\bar{D} - 0}{S_D/\sqrt{n}}.
    $$

    The one-sample t-test applied to the sample $D_1, \dots, D_n$ with null value $\mu_0 = 0$ gives

    $$
    T_{\text{one-sample}} = \frac{\bar{D} - 0}{S_D/\sqrt{n}} = T_{\text{paired}}.
    $$

    Since both the test statistic and degrees of freedom ($n-1$) are identical, the two tests are the same. $\square$

---

**Exercise 4.** A study measures reaction times (in ms) for 8 subjects under caffeine and placebo conditions. The paired differences (caffeine $-$ placebo) are: $-15, -22, -8, -30, -12, -18, -25, -10$. Compute a 99% confidence interval for the mean difference.

??? success "Solution to Exercise 4"

    Compute: $\bar{D} = (-15-22-8-30-12-18-25-10)/8 = -140/8 = -17.5$.

    $$
    S_D = \sqrt{\frac{\sum(D_i - \bar{D})^2}{7}} = \sqrt{\frac{(2.5)^2+(-4.5)^2+(9.5)^2+(-12.5)^2+(5.5)^2+(-0.5)^2+(-7.5)^2+(7.5)^2}{7}}
    $$

    $$
    = \sqrt{\frac{6.25+20.25+90.25+156.25+30.25+0.25+56.25+56.25}{7}} = \sqrt{\frac{416}{7}} \approx 7.71.
    $$

    The 99% CI is $\bar{D} \pm t_{0.005,7} \cdot S_D/\sqrt{8}$. With $t_{0.005,7} = 3.499$:

    $$
    -17.5 \pm 3.499 \times \frac{7.71}{\sqrt{8}} = -17.5 \pm 3.499 \times 2.727 = -17.5 \pm 9.54.
    $$

    The 99% CI is $(-27.04, -7.96)$. Since 0 is not in this interval, we reject $H_0\colon \mu_D = 0$ at $\alpha = 0.01$. $\square$

---

**Exercise 5.** Under what conditions is the paired t-test inappropriate? Suggest an alternative for non-normal paired differences.

??? success "Solution to Exercise 5"

    The paired t-test assumes:

    1. The differences $D_i$ are independent (different subjects).
    2. The differences are approximately normally distributed (or $n$ is large enough for the CLT).

    It is inappropriate when:

    - The sample size is very small and the differences are clearly non-normal (heavy tails, strong skewness).
    - The pairs are not naturally matched, making the correlation structure meaningless.

    A nonparametric alternative is the **Wilcoxon signed-rank test**, which tests whether the median of the differences is zero. It ranks the absolute differences, assigns signs, and uses the sum of signed ranks as the test statistic. It is valid under the weaker assumption that the difference distribution is symmetric about its median. $\square$
