# Rejection Region Demo

## Overview

The rejection region is the set of values of the test statistic that lead us to reject $H_0$. Its shape depends on whether the test is two-tailed, left-tailed, or right-tailed. This page visualizes rejection regions on the $t$-distribution for all three cases, showing both the original measurement scale (cm) and the standardized $t$-statistic scale.

## Two-Tailed Test

For $H_0\colon \mu = \mu_0$ versus $H_1\colon \mu \neq \mu_0$ at significance level $\alpha$, we reject when the test statistic falls in either tail:

$$
|t| > t_{\alpha/2,\, n-1}.
$$

The critical values split $\alpha$ equally between the two tails. In original units, the rejection region translates to

$$
\bar{x} < \mu_0 - t_{\alpha/2,\, n-1} \cdot \frac{s}{\sqrt{n}} \quad \text{or} \quad \bar{x} > \mu_0 + t_{\alpha/2,\, n-1} \cdot \frac{s}{\sqrt{n}}.
$$

### Code

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
data = stats.norm.rvs(loc=170, scale=8, size=250)

mu0 = 172
n = len(data)
df = n - 1
xbar = data.mean()
s = data.std(ddof=1)
se = s / np.sqrt(n)

alpha = 0.05
t_crit = stats.t.ppf(1 - alpha / 2, df)
t_stat = (xbar - mu0) / se

print(f"x-bar = {xbar:.2f}, SE = {se:.2f}")
print(f"t-stat = {t_stat:.4f}, t-crit = +/-{t_crit:.4f}")
print(f"Rejection boundaries: {mu0 - t_crit*se:.2f} and {mu0 + t_crit*se:.2f}")
```

### Visualization

The top panel shows the sampling distribution of $\bar{X}$ under $H_0$ in centimeters, with the rejection regions shaded in the tails. The bottom panel shows the same test on the $t$-statistic scale, where the rejection region is simply $|t| > t_{\text{crit}}$.

```python
x_t = np.linspace(-5, 5, 300)
y_t = stats.t.pdf(x_t, df)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x_t, y_t, "tomato", lw=2)
ax.fill_between(x_t[x_t <= -t_crit], stats.t.pdf(x_t[x_t <= -t_crit], df),
                color="tomato", alpha=0.5, label="Rejection region")
ax.fill_between(x_t[x_t >= t_crit], stats.t.pdf(x_t[x_t >= t_crit], df),
                color="tomato", alpha=0.5)
ax.axvline(t_stat, color="blue", linestyle="--", label=f"t = {t_stat:.2f}")
ax.set_xlabel("t")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.show()
```

## One-Tailed Tests

### Left-Tailed Test

For $H_0\colon \mu = \mu_0$ versus $H_1\colon \mu < \mu_0$, the entire rejection region is in the left tail:

$$
t < -t_{\alpha,\, n-1}.
$$

### Right-Tailed Test

For $H_0\colon \mu = \mu_0$ versus $H_1\colon \mu > \mu_0$, the rejection region is in the right tail:

$$
t > t_{\alpha,\, n-1}.
$$

### Code

```python
df = 100
alpha = 0.05
x = np.linspace(-5, 5, 300)
y = stats.t.pdf(x, df)

# Left-tailed critical value
t_lo = stats.t.ppf(alpha, df)
print(f"Left-tailed critical value: {t_lo:.4f}")

# Right-tailed critical value
t_hi = stats.t.ppf(1 - alpha, df)
print(f"Right-tailed critical value: {t_hi:.4f}")
```

Note that $t_{\alpha,\,\text{df}} = -t_{1-\alpha,\,\text{df}}$ by symmetry of the $t$-distribution.

## Interpretation

- In a **two-tailed test**, evidence against $H_0$ can come from either direction. The p-value is $2P(T \geq |t_{\text{obs}}|)$.
- In a **one-tailed test**, we only look for departures in one direction, giving more power to detect effects in that direction but none in the opposite direction.
- The **rejection region in original units** shows the actual measurement values (e.g., heights in cm) that would lead to rejection. This is often more intuitive for practitioners than the $t$-statistic scale.
- Moving from $\alpha = 0.05$ to $\alpha = 0.01$ shrinks the rejection region (critical value moves outward), requiring stronger evidence to reject $H_0$.

## Exercises

**Exercise 1.** For a two-tailed test with $n = 25$, $\alpha = 0.05$, and $\mu_0 = 100$, compute the critical values in $t$-units and in original units if $s = 15$.

??? success "Solution to Exercise 1"

    Degrees of freedom: $\text{df} = 24$. The critical $t$-value is

    $$
    t_{0.025,\,24} = 2.0639 \quad (\text{from tables or } \texttt{stats.t.ppf(0.975, 24)}).
    $$

    The standard error is $SE = 15/\sqrt{25} = 3.0$. In original units, the rejection boundaries are

    $$
    100 \pm 2.0639 \times 3.0 = 100 \pm 6.19,
    $$

    so the rejection region is $\bar{x} < 93.81$ or $\bar{x} > 106.19$. $\square$

---

**Exercise 2.** Explain why a one-tailed test is more powerful than a two-tailed test when the true effect is in the hypothesized direction. What is the cost?

??? success "Solution to Exercise 2"

    For a one-tailed test at level $\alpha$, the entire rejection probability is concentrated in one tail, so the critical value is $t_\alpha$ rather than $t_{\alpha/2}$. Since $t_\alpha < t_{\alpha/2}$, it is easier to exceed the critical value, giving higher power.

    Numerically, at $\alpha = 0.05$ with large $n$: the one-tailed critical value is $z_{0.05} = 1.645$ while the two-tailed value is $z_{0.025} = 1.960$. Any test statistic between 1.645 and 1.960 would reject with the one-tailed test but not the two-tailed test.

    The cost is that the one-tailed test has **zero power** against effects in the opposite direction. If the true effect is negative when we test for a positive effect, we can never reject $H_0$, no matter how large the negative effect. $\square$

---

**Exercise 3.** Show that the p-value for a two-tailed test equals twice the one-tailed p-value (for symmetric distributions). When might this relationship fail?

??? success "Solution to Exercise 3"

    For a symmetric distribution (like the $t$-distribution), $P(T \leq -|t|) = P(T \geq |t|)$. The two-tailed p-value is

    $$
    p_{\text{two}} = P(|T| \geq |t_{\text{obs}}|) = P(T \leq -|t_{\text{obs}}|) + P(T \geq |t_{\text{obs}}|) = 2P(T \geq |t_{\text{obs}}|) = 2p_{\text{one}}.
    $$

    This relationship fails when:

    - The test statistic has an **asymmetric null distribution** (e.g., chi-square, $F$-distribution).
    - The test is based on a **discrete distribution** where the two tails may not have equal probability mass at corresponding quantiles. $\square$

---

**Exercise 4.** A researcher tests $H_0\colon \mu = 50$ against $H_1\colon \mu > 50$ and obtains $t = 1.80$ with $\text{df} = 29$. Find the p-value and state the decision at $\alpha = 0.05$.

??? success "Solution to Exercise 4"

    For a right-tailed test, the p-value is

    $$
    p = P(T_{29} \geq 1.80) = 1 - F_{T_{29}}(1.80).
    $$

    Using Python: `1 - stats.t.cdf(1.80, 29)` $\approx 0.0411$.

    Since $0.0411 < 0.05$, we reject $H_0$ and conclude there is significant evidence that $\mu > 50$ at the 5% level. $\square$

---

**Exercise 5.** Sketch (or describe) how the two-tailed rejection region changes as $\alpha$ decreases from 0.10 to 0.01. What happens to the probability of a Type II error?

??? success "Solution to Exercise 5"

    As $\alpha$ decreases:

    - The critical values $\pm t_{\alpha/2}$ move further from zero (e.g., from $\pm 1.645$ at $\alpha=0.10$ to $\pm 2.576$ at $\alpha=0.01$ for large $n$).
    - The shaded rejection region in each tail shrinks.
    - It becomes harder to reject $H_0$, so the probability of a Type I error decreases.

    However, the probability of a **Type II error** ($\beta$) increases. With a stricter threshold, we are more likely to fail to reject $H_0$ even when $H_1$ is true. Power $= 1 - \beta$ decreases. This illustrates the fundamental trade-off: reducing one type of error increases the other, unless we also increase the sample size. $\square$
