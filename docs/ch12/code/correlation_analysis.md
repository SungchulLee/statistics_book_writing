# Correlation Analysis Demonstrations

## Overview

This page demonstrates how to compute and compare the three most common correlation coefficients -- Pearson, Spearman, and Kendall -- on bivariate data with a known linear relationship. We verify that Spearman's rank correlation equals the Pearson correlation computed on ranks, and we visualize the data with a scatter plot and ordinary least-squares regression line.

---

## Pearson, Spearman, and Kendall Coefficients

Given paired observations $(x_1, y_1), \ldots, (x_n, y_n)$, the three standard correlation measures are defined as follows.

**Pearson's $r$** measures the strength of the *linear* relationship:

$$
r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}\;\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
$$

**Spearman's $\rho_s$** is Pearson's $r$ applied to the ranks of $x$ and $y$:

$$
\rho_s = r(\text{rank}(x),\; \text{rank}(y))
$$

**Kendall's $\tau$** counts the proportion of concordant minus discordant pairs:

$$
\tau = \frac{(\text{concordant pairs}) - (\text{discordant pairs})}{\binom{n}{2}}
$$

All three coefficients lie in $[-1, 1]$, but they emphasize different aspects of association. Pearson captures linear relationships; Spearman and Kendall capture monotonic relationships and are more robust to outliers.

---

## Generating Bivariate Data

We generate $n = 120$ points from a linear model with Gaussian noise:

$$
y_i = 0.8\, x_i + 5 + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, 8^2)
$$

where $x_i \sim \text{Uniform}(10, 60)$.

```python
import numpy as np
from scipy import stats

np.random.seed(42)

n = 120
x = np.random.uniform(10, 60, n)
noise = np.random.normal(0, 8, n)
y = 0.8 * x + 5 + noise
```

---

## Computing the Correlation Coefficients

SciPy provides functions for each measure, returning both the coefficient and its p-value under the null hypothesis of no association:

```python
r_pearson, p_pearson = stats.pearsonr(x, y)
r_spearman, p_spearman = stats.spearmanr(x, y)
r_kendall, p_kendall = stats.kendalltau(x, y)

print(f"Pearson  r = {r_pearson:.4f}  (p = {p_pearson:.2e})")
print(f"Spearman rho = {r_spearman:.4f}  (p = {p_spearman:.2e})")
print(f"Kendall  tau = {r_kendall:.4f}  (p = {p_kendall:.2e})")
```

---

## Verifying the Rank Equivalence

A useful identity: Spearman's $\rho_s$ equals the Pearson $r$ computed on the rank-transformed data. We verify this numerically:

```python
r_rank = stats.pearsonr(stats.rankdata(x), stats.rankdata(y))[0]
print(f"Pearson r on ranks = {r_rank:.4f}")
print(f"Spearman rho       = {r_spearman:.4f}")
# These two values should match.
```

---

## Scatter Plot with Regression Line

Overlaying the ordinary least-squares (OLS) regression line on the scatter plot provides a visual check that the linear model is appropriate:

```python
import matplotlib.pyplot as plt

slope, intercept, _, _, _ = stats.linregress(x, y)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x, y, alpha=0.6, edgecolors='k', linewidths=0.3)
x_line = np.array([x.min(), x.max()])
ax.plot(x_line, intercept + slope * x_line, 'r-', linewidth=2,
        label=f'OLS: y = {slope:.2f}x + {intercept:.2f}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Pearson r = {r_pearson:.3f}')
ax.legend()
plt.tight_layout()
plt.show()
```

---

## Interpretation

For data generated from a linear model with moderate noise, all three coefficients are positive and highly significant. The ordering $|r| \ge |\rho_s| \ge |\tau|$ is typical: Pearson's $r$ is most powerful when the true relationship is linear, while Kendall's $\tau$ is the most conservative. Spearman's $\rho_s$ sits between the two.

When the relationship is nonlinear but monotonic, Spearman and Kendall will outperform Pearson. When the data contain outliers, rank-based measures are more robust. Always inspect a scatter plot before relying on any single correlation number.

---

## Exercises

**Exercise 1.**
Generate $n = 200$ observations from the model $y = 3x + 2 + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, 5^2)$ and $x \sim \text{Uniform}(0, 20)$. Compute all three correlation coefficients and their p-values. Which coefficient is largest in absolute value, and why?

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(0)
    n = 200
    x = np.random.uniform(0, 20, n)
    y = 3 * x + 2 + np.random.normal(0, 5, n)

    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)
    r_k, p_k = stats.kendalltau(x, y)

    print(f"Pearson  r = {r_p:.4f}, p = {p_p:.2e}")
    print(f"Spearman rho = {r_s:.4f}, p = {p_s:.2e}")
    print(f"Kendall  tau = {r_k:.4f}, p = {p_k:.2e}")
    ```

    Since the true relationship is linear, Pearson's $r$ is the most efficient estimator and will be the largest. Spearman's $\rho_s$ is close but slightly lower, and Kendall's $\tau$ is the smallest. All three are highly significant because the linear signal is strong relative to the noise. $\square$

---

**Exercise 2.**
Construct a dataset of $n = 100$ points where Spearman's $\rho_s > 0.9$ but Pearson's $r < 0.5$. Explain what kind of relationship produces this discrepancy.

??? success "Solution to Exercise 2"

    A monotonic but strongly nonlinear relationship produces a high Spearman coefficient with a low Pearson coefficient. For example:

    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(42)
    x = np.random.uniform(0, 5, 100)
    y = np.exp(x) + np.random.normal(0, 1, 100)

    r_p, _ = stats.pearsonr(x, y)
    r_s, _ = stats.spearmanr(x, y)
    print(f"Pearson r = {r_p:.4f}")
    print(f"Spearman rho = {r_s:.4f}")
    ```

    The exponential relationship is strongly monotonic (high $\rho_s$) but far from linear, so Pearson's $r$ is substantially lower. This illustrates that Pearson only captures linear association while Spearman captures any monotonic relationship. $\square$

---

**Exercise 3.**
Prove that Pearson's $r$ is invariant under positive affine transformations. That is, show that for constants $a, c > 0$ and any $b, d$,

$$
r(aX + b,\; cY + d) = r(X, Y)
$$

??? success "Solution to Exercise 3"

    Let $U = aX + b$ and $V = cY + d$ with $a, c > 0$. Then $\bar{U} = a\bar{X} + b$ and $\bar{V} = c\bar{Y} + d$, so $U_i - \bar{U} = a(X_i - \bar{X})$ and $V_i - \bar{V} = c(Y_i - \bar{Y})$.

    The numerator of $r(U, V)$ becomes:

    $$
    \sum_{i=1}^n (U_i - \bar{U})(V_i - \bar{V}) = ac \sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})
    $$

    The denominator becomes:

    $$
    \sqrt{\sum(U_i - \bar{U})^2}\;\sqrt{\sum(V_i - \bar{V})^2} = a\sqrt{\sum(X_i - \bar{X})^2}\;\cdot\; c\sqrt{\sum(Y_i - \bar{Y})^2}
    $$

    Therefore:

    $$
    r(U, V) = \frac{ac \sum(X_i - \bar{X})(Y_i - \bar{Y})}{ac\sqrt{\sum(X_i - \bar{X})^2}\;\sqrt{\sum(Y_i - \bar{Y})^2}} = r(X, Y)
    $$

    The positive constants $a$ and $c$ cancel in the ratio. $\square$

---

**Exercise 4.**
Write a Python function that takes two arrays and returns all three correlation coefficients as a dictionary. Test it on three different scenarios: (a) strong linear, (b) weak nonlinear, (c) data with outliers. Discuss which coefficient is most affected by the outliers.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy import stats

    def all_correlations(x, y):
        r_p, _ = stats.pearsonr(x, y)
        r_s, _ = stats.spearmanr(x, y)
        r_k, _ = stats.kendalltau(x, y)
        return {"pearson": r_p, "spearman": r_s, "kendall": r_k}

    np.random.seed(42)
    n = 100

    # (a) Strong linear
    x_a = np.random.normal(0, 1, n)
    y_a = 2 * x_a + np.random.normal(0, 0.5, n)
    print("Linear:", all_correlations(x_a, y_a))

    # (b) Weak nonlinear
    x_b = np.random.uniform(-3, 3, n)
    y_b = x_b ** 2 + np.random.normal(0, 1, n)
    print("Quadratic:", all_correlations(x_b, y_b))

    # (c) With outliers
    x_c = np.random.normal(0, 1, n)
    y_c = 0.8 * x_c + np.random.normal(0, 0.3, n)
    x_c[:3] = [6, -6, 7]
    y_c[:3] = [-6, 6, -7]
    print("Outliers:", all_correlations(x_c, y_c))
    ```

    Pearson's $r$ is most affected by outliers because it depends on means and standard deviations, which are sensitive to extreme values. Spearman and Kendall, being rank-based, are more robust. In scenario (c), the outliers pull Pearson's $r$ toward zero (or even negative), while Spearman and Kendall remain closer to the true positive association. $\square$

---

**Exercise 5.**
Show that for a bivariate sample of size $n$, if Pearson's $r = 1$, then all points $(x_i, y_i)$ lie on a line with positive slope. Provide a formal proof.

??? success "Solution to Exercise 5"

    The Cauchy--Schwarz inequality states that for vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$:

    $$
    \left(\sum_{i=1}^n a_i b_i\right)^2 \le \left(\sum_{i=1}^n a_i^2\right)\left(\sum_{i=1}^n b_i^2\right)
    $$

    with equality if and only if $\mathbf{a} = \lambda \mathbf{b}$ for some scalar $\lambda$.

    Set $a_i = x_i - \bar{x}$ and $b_i = y_i - \bar{y}$. Then $r = 1$ means:

    $$
    \frac{\sum a_i b_i}{\sqrt{\sum a_i^2}\sqrt{\sum b_i^2}} = 1
    $$

    By Cauchy--Schwarz equality, we must have $y_i - \bar{y} = \lambda(x_i - \bar{x})$ for all $i$, with some $\lambda > 0$ (positive because $r > 0$). Rearranging: $y_i = \lambda x_i + (\bar{y} - \lambda \bar{x})$. This is a line with positive slope $\lambda$. $\square$
