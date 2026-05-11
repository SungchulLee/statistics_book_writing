# Student's t Distribution

## Overview

The Student's $t$ distribution arises when estimating the mean of a normally distributed population using the **sample standard deviation** $S$ instead of the known population standard deviation $\sigma$. It accounts for the additional uncertainty introduced by estimating $\sigma$.

---

## Definition

Let $Z \sim N(0,1)$ and $V \sim \chi^2_d$ be independent. Then the ratio:

$$
T = \frac{Z}{\sqrt{V/d}} \sim t_d
$$

follows the Student's $t$ distribution with $d$ degrees of freedom.

---

## Degrees of Freedom

The degrees of freedom $d = n - 1$ reflects the number of independent pieces of information used to estimate the sample variance.

- **Small $d$**: Heavier tails than the normal, reflecting greater uncertainty.
- **Large $d$ ($> 30$)**: Virtually indistinguishable from $N(0, 1)$.

---

## Properties

$$
\begin{aligned}
\text{Mean} &= 0 \quad \text{for } d > 1 \\
\text{Variance} &= \frac{d}{d - 2} \quad \text{for } d > 2
\end{aligned}
$$

As $d \to \infty$, the variance approaches 1 and $t_d \to N(0, 1)$.

---

## PDF

$$
f_T(x) = \frac{1}{\sqrt{d}\,B\!\left(\tfrac{1}{2}, \tfrac{d}{2}\right)} \left(1 + \frac{x^2}{d}\right)^{-\frac{d+1}{2}}
$$

where $B(\cdot, \cdot)$ is the Beta function.

### Proof Sketch

With $T = Z / \sqrt{V/d}$, use the change-of-variables technique on the joint density of $(Z, V)$. The Jacobian factor is $\sqrt{v/d}$, and after integrating out the $\chi^2$ variable, the marginal density of $T$ takes the form above. The conditional distribution $V | T = t$ turns out to be Gamma.

---

## Fat Tails

The $t$ distribution has **heavier tails** than the normal distribution, meaning extreme values are more likely:

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

fig, (ax_full, ax_tail) = plt.subplots(1, 2, figsize=(12, 3))
x = np.linspace(-4, 4, 200)

ax_full.plot(x, stats.norm().pdf(x), label='Normal')
ax_full.plot(x, stats.t(df=10).pdf(x), label='t(10)')
ax_full.set_title('Full PDF')
ax_full.legend()

ax_tail.plot(x[-50:], stats.norm().pdf(x[-50:]), label='Normal')
ax_tail.plot(x[-50:], stats.t(df=10).pdf(x[-50:]), label='t(10)')
ax_tail.set_title('Right Tail (zoomed)')
ax_tail.legend()

plt.tight_layout()
plt.show()
```

### Convergence to Normal

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

fig, ax = plt.subplots(figsize=(12, 3))
x = np.linspace(-3, 3, 200)

for df in [1, 2, 5, 10, 20]:
    ax.plot(x, stats.t(df).pdf(x), label=f'df={df}')
ax.plot(x, stats.norm().pdf(x), 'r--', lw=2, label='Normal')
ax.legend()
ax.set_title('t-Distribution Converges to Normal as df Increases')
plt.show()
```

---

## Why t?
When the population is normal and $\sigma$ is unknown, replacing $\sigma$ with $S$ yields:

$$
\frac{\bar{X} - \mu}{S / \sqrt{n}} \sim t_{n-1}
$$

This arises because:

1. $\bar{X} \sim N(\mu, \sigma^2/n)$, so $\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0,1)$.
2. $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$.
3. $\bar{X}$ and $S^2$ are **independent** (a special property of the normal distribution).
4. The ratio $\frac{N(0,1)}{\sqrt{\chi^2_{n-1}/(n-1)}}$ is by definition $t_{n-1}$.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)
n, mu, sigma = 10, 0, 10
n_sim = 10_000

samples = np.random.normal(mu, sigma, (n, n_sim))
x_bar = samples.mean(axis=0)
s = samples.std(axis=0, ddof=1)
t_stats = (x_bar - mu) / (s / np.sqrt(n))

fig, ax = plt.subplots(figsize=(12, 3))
bins = np.arange(-6, 6, 0.1)
ax.hist(t_stats, bins=bins, density=True, alpha=0.7, label=f'Simulated $t_{{{n-1}}}$')
ax.plot(bins, stats.t(n-1).pdf(bins), '--r', lw=2, label=f'$t_{{{n-1}}}$ PDF')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

---

## Interpreting the Role of t
### Large n: CLT Justifies z
When $n$ is large, $S \approx \sigma$, and the difference between $t_{n-1}$ and $N(0,1)$ is negligible. In practice, $z$ is just as good.

### Small n: Where t Shines — But Only Under Normality
The $t$ distribution matters most for small $n$. Its heavier tails properly account for the extra variability from using $S$ instead of $\sigma$. However, this result is **exact only if the population is normal**.

### Non-Normal Populations

If the population is skewed or heavy-tailed, the $t$ approximation is **poor** for small $n$. Neither $t$ nor $z$ is trustworthy; robust or nonparametric methods are preferable.

### Summary

| Scenario | Recommendation |
|:---|:---|
| Large $n$ | Use $z$; the $t$ adjustment is negligible |
| Small $n$, normal population | $t$ is exact and appropriate |
| Small $n$, non-normal population | Use robust/nonparametric methods |

---

## Random Samples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)
df = 5
data = stats.t(df).rvs(10_000)

fig, ax = plt.subplots(figsize=(12, 3))
bins = np.linspace(-5, 5, 101)
ax.hist(data, bins=bins, density=True, histtype='step', label='t Samples')
ax.plot(bins, stats.t(df).pdf(bins), '--b', lw=2, label='t PDF')
ax.plot(bins, stats.norm(data.mean(), data.std()).pdf(bins),
        '--r', lw=2, label='Normal Approx')
ax.legend()
plt.show()
```

---

## Key Takeaways

- The $t$ distribution accounts for the uncertainty of estimating $\sigma$ with $S$.
- It has heavier tails than the normal, especially for small degrees of freedom.
- As $d \to \infty$, the $t$ distribution converges to $N(0,1)$.
- The exactness of the $t$ result depends critically on the normality of the population.

## Exercises

**Exercise 1.**
Salaries are normal with $\mu = \$40{,}000$. Sample $n = 9$, $s = \$8{,}000$. Compute $P(\bar X \ge \$45{,}000)$.

??? success "Solution to Exercise 1"
    $\mathrm{SE} = s/\sqrt n = 8000/3 \approx 2667$. Test statistic: $t = (45000 - 40000)/2667 \approx 1.875$.

    Under $t_8$: $P(T \ge 1.875) \approx 0.048$ — about 4.8%.

    Note: we use $t$ instead of $z$ because $\sigma$ is unknown and $s$ is estimated from the sample, introducing additional uncertainty.

---

**Exercise 2.**
**Derive the $t$-distribution.** Show that if $Z \sim N(0, 1)$ and $V \sim \chi^2_\nu$ independent, then $T = Z/\sqrt{V/\nu} \sim t_\nu$.

??? success "Solution to Exercise 2"
    By definition, $t_\nu$ is the distribution of $Z/\sqrt{V/\nu}$ where $Z \sim N(0, 1)$ and $V \sim \chi^2_\nu$ are independent.

    Derivation of PDF: condition on $V = v$. Given $V = v$, $T = Z/\sqrt{v/\nu}$, so $T \mid V \sim N(0, \nu/v)$. Density:

    $$
    f_{T \mid V}(t \mid v) = \frac{1}{\sqrt{2\pi \nu/v}} e^{-vt^2/(2\nu)}
    $$

    Marginal of $T$: integrate over $v$ using $V \sim \chi^2_\nu$ density. Result:

    $$
    f_T(t) = \frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\,\Gamma(\nu/2)} \left(1 + \frac{t^2}{\nu}\right)^{-(\nu+1)/2}
    $$

    The $t$ density has polynomial tails $\sim t^{-(\nu+1)}$, heavier than normal's $e^{-t^2/2}$.

---

**Exercise 3.**
**$t$ approaches normal.** Show $t_\nu \to N(0, 1)$ as $\nu \to \infty$.

??? success "Solution to Exercise 3"
    From the $t$ definition $T = Z/\sqrt{V/\nu}$ with $V \sim \chi^2_\nu$. Note $V/\nu = (1/\nu)\sum_{i=1}^\nu Z_i^2 \to 1$ in probability by LLN. So $\sqrt{V/\nu} \to 1$, and $T \to Z \sim N(0, 1)$.

    More precisely: by Slutsky's theorem, $T = Z/\sqrt{V/\nu} \xrightarrow{d} Z/1 = Z$.

    **Practical:** for $\nu \ge 30$, the $t$ distribution is nearly indistinguishable from normal; $t_{30}$ critical values are within 2% of $z$ critical values. This is why $n \ge 30$ is the rule of thumb for using $z$ instead of $t$.

---

**Exercise 4.**
**Why use $t$ instead of $z$.** A statistician computes $z = (\bar X - \mu_0)/(\sigma/\sqrt n)$ but realizes $\sigma$ is unknown. They substitute $s$. Show that the resulting $t = (\bar X - \mu_0)/(s/\sqrt n) \sim t_{n-1}$.

??? success "Solution to Exercise 4"
    Under the null $\mu = \mu_0$, $\bar X \sim N(\mu_0, \sigma^2/n)$, so $Z = (\bar X - \mu_0)/(\sigma/\sqrt n) \sim N(0, 1)$.

    The sample variance $s^2$ scaled by $\sigma^2$ has chi-squared distribution: $(n-1)s^2/\sigma^2 \sim \chi^2_{n-1}$ (for normal data).

    $\bar X$ and $s^2$ are independent for normal data (a non-trivial fact specific to normality).

    Therefore:

    $$
    t = \frac{\bar X - \mu_0}{s/\sqrt n} = \frac{(\bar X - \mu_0)/(\sigma/\sqrt n)}{\sqrt{((n-1) s^2/\sigma^2)/(n-1)}} = \frac{Z}{\sqrt{V/(n-1)}}
    $$

    with $V \sim \chi^2_{n-1}$ independent of $Z$. By the definition of the $t$, this is $t_{n-1}$.

    Using $s$ instead of $\sigma$ adds a chi-squared denominator — the $t$ distribution accounts for this extra uncertainty by having heavier tails than normal.

---

**Exercise 5.**
**Heavy tails of the $t$.** For $t_3$, compute $P(|T| > 2)$ and $P(|T| > 4)$. Compare with the normal.

??? success "Solution to Exercise 5"
    $t_3$: $P(|T| > 2) = 2 \cdot P(T > 2)$. From $t_3$ table: $P(T > 2) \approx 0.07$, so $P(|T| > 2) \approx 0.14$.

    $P(|T| > 4) \approx 2 \cdot 0.014 = 0.028$.

    Normal: $P(|Z| > 2) \approx 0.046$, $P(|Z| > 4) \approx 6 \times 10^{-5}$.

    Comparison at $|x| = 4$: normal probability is $6 \times 10^{-5}$ (essentially zero); $t_3$ probability is $0.028$ — over 400 times larger. The $t_3$ assigns vastly more probability to extreme outcomes than the normal.

    **Practical consequence:** small-sample $t$-tests have lower power than $z$-tests at the same significance level because critical values are larger to compensate for heavier tails. The trade-off is between assumptions (known $\sigma$) and tail conservativeness.

---

**Exercise 6.**
**Welch's $t$-test.** For two independent samples with unequal variances, Welch's test uses $t = (\bar X_1 - \bar X_2)/\sqrt{s_1^2/n_1 + s_2^2/n_2}$ with degrees of freedom approximated by the Welch–Satterthwaite formula. State this formula and explain why it is not an integer.

??? success "Solution to Exercise 6"
    **Welch–Satterthwaite degrees of freedom:**

    $$
    \nu_{WS} = \frac{(s_1^2/n_1 + s_2^2/n_2)^2}{(s_1^2/n_1)^2/(n_1 - 1) + (s_2^2/n_2)^2/(n_2 - 1)}
    $$

    This formula approximates the distribution of the linear combination $s_1^2/n_1 + s_2^2/n_2$ as a scaled chi-squared with $\nu_{WS}$ degrees of freedom. The approximation is moment-matching: equate the first two moments of the $\chi^2$ approximation to the exact distribution.

    **Why non-integer:** $\nu_{WS}$ depends on the *sample* variances $s_1^2, s_2^2$ — quantities that can take any positive real value. The formula doesn't produce integer outputs except by accident.

    Implementation: use $t_{\nu_{WS}}$ critical values with $\nu_{WS}$ rounded down (conservative) or use it directly in software that accepts non-integer df.

    Welch's test is the **default two-sample $t$-test** in R (`t.test`) and SciPy (`scipy.stats.ttest_ind(equal_var=False)`) precisely because it doesn't require the equal-variance assumption that the original Student's $t$-test makes.
