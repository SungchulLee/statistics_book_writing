# F-Test Tail Region Visualization

## Overview

Visualizing the tail regions of the F-distribution is essential for understanding how the F-test for equality of variances reaches its decision. The observed F-statistic is placed on the density curve, and the shaded tail areas correspond to the $p$-value. This page demonstrates how to construct such plots for one-sided and two-sided alternatives, building geometric intuition for hypothesis testing with the F-distribution.

## The F-Distribution and Tail Areas

For two independent samples of sizes $n_1$ and $n_2$ from normal populations, the F-statistic

$$
F_{\text{obs}} = \frac{S_1^2}{S_2^2}
$$

follows an $F(d_1, d_2)$ distribution under $H_0: \sigma_1^2 = \sigma_2^2$, where $d_1 = n_1 - 1$ and $d_2 = n_2 - 1$.

The $p$-value depends on the alternative hypothesis:

- **Right-tail** ($H_1: \sigma_1^2 > \sigma_2^2$): $p = P(F \ge F_{\text{obs}})$.
- **Left-tail** ($H_1: \sigma_1^2 < \sigma_2^2$): $p = P(F \le F_{\text{obs}})$.
- **Two-sided** ($H_1: \sigma_1^2 \neq \sigma_2^2$): $p = 2\min\!\bigl(P(F \le F_{\text{obs}}),\; P(F \ge F_{\text{obs}})\bigr)$.

## Code

The following code computes the F-statistic from two samples and plots the $F(d_1, d_2)$ density with both tail regions shaded:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

sample1 = [12, 15, 14, 10, 13, 14, 12, 11]
sample2 = [22, 25, 20, 18, 24, 23, 19, 21]

x1 = np.asarray(sample1, dtype=float)
x2 = np.asarray(sample2, dtype=float)
df1, df2 = x1.size - 1, x2.size - 1

F_obs = x1.var(ddof=1) / x2.var(ddof=1)

xs = np.linspace(0.01, max(6, F_obs + 2), 400)
pdf = f(df1, df2).pdf(xs)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(xs, pdf, linewidth=2, label="F({},{}) PDF".format(df1, df2))

# Left-tail shading
mask_left = xs <= F_obs
ax.fill_between(xs[mask_left], pdf[mask_left], 0, alpha=0.15, label="Left tail")

# Right-tail shading
mask_right = xs >= F_obs
ax.fill_between(xs[mask_right], pdf[mask_right], 0, alpha=0.15, label="Right tail")

ax.axvline(F_obs, linestyle="--", color="black", label=f"F_obs = {F_obs:.3f}")
ax.set_xlabel("F")
ax.set_ylabel("Density")
ax.set_title(f"F({df1}, {df2}) with observed F = {F_obs:.3f}")
ax.legend()
plt.tight_layout()
plt.show()
```

To compute the $p$-values explicitly:

```python
p_left = f(df1, df2).cdf(F_obs)
p_right = f(df1, df2).sf(F_obs)
p_two = 2 * min(p_left, p_right)

print(f"Left-tail  p-value: {p_left:.4f}")
print(f"Right-tail p-value: {p_right:.4f}")
print(f"Two-sided  p-value: {p_two:.4f}")
```

## Interpretation

- The area under the left tail up to $F_{\text{obs}}$ represents the probability of observing a variance ratio as small as or smaller than the one obtained, under $H_0$.
- The area under the right tail from $F_{\text{obs}}$ onward represents the probability of a ratio as large as or larger.
- For the two-sided test, we double the smaller tail area. If $F_{\text{obs}}$ is close to 1 (the expected value under $H_0$ for large $d_2$), both tails are large and the $p$-value is near 1.
- The F-distribution is right-skewed, especially for small degrees of freedom. This asymmetry is why the left- and right-tail $p$-values are generally not equal for a given $F_{\text{obs}}$.

## Exercises

**Exercise 1.** For the samples given in the code above, compute $F_{\text{obs}}$ by hand and verify against the code output. State the degrees of freedom.

??? success "Solution to Exercise 1"

    Sample 1: $\{12, 15, 14, 10, 13, 14, 12, 11\}$, $n_1 = 8$, $\bar{x}_1 = 12.625$.

    $$
    S_1^2 = \frac{1}{7}\sum(x_i - 12.625)^2 = \frac{1}{7}(0.390625 + 5.640625 + 1.890625 + 6.890625 + 0.140625 + 1.890625 + 0.390625 + 2.640625) \approx 2.839.
    $$

    Sample 2: $\{22, 25, 20, 18, 24, 23, 19, 21\}$, $n_2 = 8$, $\bar{x}_2 = 21.5$.

    $$
    S_2^2 = \frac{1}{7}\sum(x_i - 21.5)^2 = \frac{1}{7}(0.25 + 12.25 + 2.25 + 12.25 + 6.25 + 2.25 + 6.25 + 0.25) = 6.0.
    $$

    $$
    F_{\text{obs}} = \frac{2.839}{6.0} \approx 0.473, \quad d_1 = 7, \; d_2 = 7.
    $$

---

**Exercise 2.** Modify the code to shade only the right tail for a one-sided test $H_1: \sigma_1^2 > \sigma_2^2$. What is the right-tail $p$-value?

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import f

    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)
    df1, df2 = 7, 7
    F_obs = x1.var(ddof=1) / x2.var(ddof=1)

    xs = np.linspace(0.01, 6, 400)
    pdf = f(df1, df2).pdf(xs)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, pdf, lw=2)
    mask = xs >= F_obs
    ax.fill_between(xs[mask], pdf[mask], 0, alpha=0.3, color="red",
                    label="Right tail")
    ax.axvline(F_obs, ls="--", color="black")
    ax.set_title(f"Right-tail test: p = {f(df1,df2).sf(F_obs):.4f}")
    ax.legend()
    plt.tight_layout()
    plt.show()
    ```

    Since $F_{\text{obs}} \approx 0.473 < 1$, the right-tail area is very large ($p \approx 0.83$), providing no evidence that $\sigma_1^2 > \sigma_2^2$.

---

**Exercise 3.** Explain why the $F(d_1, d_2)$ density is not symmetric around 1. How does this asymmetry affect the two-sided $p$-value calculation?

??? success "Solution to Exercise 3"

    The F-distribution is defined as the ratio of two scaled chi-squared variables, both of which are non-negative and right-skewed. The density is supported on $(0, \infty)$ and has mode $(d_1-2)/d_1 \cdot d_2/(d_2+2) < 1$ for $d_1 > 2$. This inherent right skewness means that $P(F > c) \neq P(F < 1/c)$ in general.

    For the two-sided $p$-value, we cannot simply double the one-tail area as we do with symmetric distributions like the normal. Instead we use $p = 2\min(P(F \le F_{\text{obs}}), P(F \ge F_{\text{obs}}))$, which picks the smaller tail and doubles it. This ensures the test is valid but means that the critical region is not symmetric around 1 on the $F$ scale. $\square$

---

**Exercise 4.** Create a figure with three subplots showing the $F(d_1, d_2)$ density for $(d_1, d_2) \in \{(5,5), (10,10), (30,30)\}$. Discuss how the shape changes with increasing degrees of freedom.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import f

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, (d1, d2) in zip(axes, [(5,5), (10,10), (30,30)]):
        xs = np.linspace(0.01, 4, 300)
        ax.plot(xs, f(d1, d2).pdf(xs), lw=2)
        ax.axvline(1, ls=":", color="gray")
        ax.set_title(f"F({d1}, {d2})")
        ax.set_xlabel("F")
    plt.tight_layout()
    plt.show()
    ```

    As degrees of freedom increase, the F-density becomes more concentrated around 1 (its expected value approaches $d_2/(d_2-2) \to 1$) and more symmetric. For large $d_1, d_2$, $\ln F$ is approximately normal, and the distribution closely resembles a normal centered near 1.

---

**Exercise 5.** Prove that if $F \sim F(d_1, d_2)$, then $1/F \sim F(d_2, d_1)$. Use this to show that the left-tail $p$-value for $F_{\text{obs}}$ under $F(d_1, d_2)$ equals the right-tail $p$-value for $1/F_{\text{obs}}$ under $F(d_2, d_1)$.

??? success "Solution to Exercise 5"

    By definition, $F = (U/d_1)/(V/d_2)$ where $U \sim \chi^2(d_1)$ and $V \sim \chi^2(d_2)$ are independent. Then

    $$
    \frac{1}{F} = \frac{V/d_2}{U/d_1} \sim F(d_2, d_1)
    $$

    by the definition of the F-distribution with swapped degrees of freedom.

    Now, $P(F \le F_{\text{obs}}) = P(1/F \ge 1/F_{\text{obs}})$. Since $1/F \sim F(d_2, d_1)$, this is the right-tail probability $P(F(d_2, d_1) \ge 1/F_{\text{obs}})$, which is exactly the survival function of $F(d_2, d_1)$ evaluated at $1/F_{\text{obs}}$. $\square$
