# Student's $t$ Distribution

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

## Why $t$?

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

## Interpreting the Role of $t$

### Large $n$: CLT Justifies $z$

When $n$ is large, $S \approx \sigma$, and the difference between $t_{n-1}$ and $N(0,1)$ is negligible. In practice, $z$ is just as good.

### Small $n$: Where $t$ Shines — But Only Under Normality

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
