# F-Distribution Density Function

## Overview

The **$F$-distribution** with parameters $d_1$ (numerator) and $d_2$ (denominator) degrees of freedom is the distribution of the ratio of two independent scaled chi-square random variables:

$$
F = \frac{U/d_1}{V/d_2}, \qquad U \sim \chi^2_{d_1},\; V \sim \chi^2_{d_2}
$$

It is the key distribution for ANOVA F-tests, comparing variances, and testing nested regression models.

---

## Key Properties

| Property | Condition | Value |
|---|---|---|
| Support | — | $[0, \infty)$ |
| Mean | $d_2 > 2$ | $\dfrac{d_2}{d_2 - 2}$ |
| Mode | $d_1 > 2$ | $\dfrac{d_1 - 2}{d_1} \cdot \dfrac{d_2}{d_2 + 2}$ |
| Variance | $d_2 > 4$ | $\dfrac{2d_2^2(d_1 + d_2 - 2)}{d_1(d_2-2)^2(d_2-4)}$ |

---

## Code

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

d1, d2 = 5, 12
f_dist = stats.f(dfn=d1, dfd=d2)

x = np.linspace(f_dist.ppf(1e-6), f_dist.ppf(1 - 1e-6), 600)
y = f_dist.pdf(x)

mean = d2 / (d2 - 2)
mode = ((d1 - 2) / d1) * (d2 / (d2 + 2))

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, y, lw=2, label=f"F PDF (d1={d1}, d2={d2})")
ax.axvline(mean, linestyle='--', alpha=0.85, label=f"mean = {mean:.3f}")
ax.axvline(mode, linestyle=':', alpha=0.85, label=f"mode = {mode:.3f}")
ax.set_title("F Distribution — PDF")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.legend()
ax.grid(True, linestyle=":")
plt.tight_layout()
plt.show()
```

---

## Interpretation

The $F$-distribution is always right-skewed (supported on $[0, \infty)$). For large $d_1$ and $d_2$, it approaches a normal distribution centered near 1. The mean exceeds 1 (equal to $d_2/(d_2-2)$), reflecting the slight positive bias of variance ratios.

---

## Exercises

**Exercise 1.**
Compute the mean of $F_{5, 12}$ and explain why the mean of an $F$-distribution is always greater than 1 (when it exists).

??? success "Solution to Exercise 1"
    $E[F] = d_2/(d_2 - 2) = 12/10 = 1.2$.

    The mean exceeds 1 because the denominator chi-square is divided by $d_2$ while contributing $d_2$ to the mean (since $E[\chi^2_{d_2}] = d_2$). The ratio $E[V/d_2] = 1$ but $E[d_1/U]$ introduces an upward correction due to Jensen's inequality (the reciprocal is convex). Formally, $E[1/(V/d_2)] > 1/E[V/d_2] = 1$.

---

**Exercise 2.**
If $T \sim t_\nu$, show that $T^2 \sim F_{1, \nu}$.

??? success "Solution to Exercise 2"
    By definition, $T = Z/\sqrt{V/\nu}$ where $Z \sim N(0,1)$ and $V \sim \chi^2_\nu$ are independent. Then:

    $$
    T^2 = \frac{Z^2}{V/\nu} = \frac{Z^2/1}{V/\nu}
    $$

    Since $Z^2 \sim \chi^2_1$, this is the ratio $(\chi^2_1/1)/(\chi^2_\nu/\nu) \sim F_{1,\nu}$ by definition. $\square$

---

**Exercise 3.**
In a one-way ANOVA with 3 groups of size 10, what are the degrees of freedom for the $F$-test? What is the critical value at the 5% significance level?

??? success "Solution to Exercise 3"

    - Between-group df: $d_1 = k - 1 = 2$
    - Within-group df: $d_2 = N - k = 30 - 3 = 27$

    The critical value: `stats.f.ppf(0.95, 2, 27) ≈ 3.354`.

    If the observed $F$-statistic exceeds 3.354, we reject the null hypothesis that all group means are equal.

---

**Exercise 4.**
Show that $1/F_{d_1, d_2} \sim F_{d_2, d_1}$.

??? success "Solution to Exercise 4"
    If $F = (U/d_1)/(V/d_2)$ where $U \sim \chi^2_{d_1}$ and $V \sim \chi^2_{d_2}$ are independent, then:

    $$
    \frac{1}{F} = \frac{V/d_2}{U/d_1}
    $$

    This is the ratio of two independent scaled chi-squares with the numerator having $d_2$ df and denominator having $d_1$ df, so $1/F \sim F_{d_2, d_1}$. $\square$
