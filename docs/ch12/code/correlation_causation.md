# Correlation Causation

## Overview

This page explores the critical distinction between correlation and causation. We compare Pearson, Spearman, and Kendall correlation coefficients on different relationship types, construct confidence intervals via the Fisher $z$-transformation, demonstrate Simpson's paradox, compute partial correlations to control for confounders, and show how multiple testing creates spurious correlations. Understanding these topics is essential for drawing valid conclusions from observational data.

---

## Comparing Correlation Measures

Given paired observations $(x_1, y_1), \ldots, (x_n, y_n)$, three standard correlation measures capture different aspects of association:

- **Pearson's $r$** measures linear association.
- **Spearman's $\rho_s$** is Pearson's $r$ applied to ranks, capturing monotonic relationships.
- **Kendall's $\tau$** counts concordant versus discordant pairs.

When the relationship is linear, all three agree. When it is monotonic but nonlinear, Spearman and Kendall outperform Pearson. When outliers are present, the rank-based measures are more robust.

```python
import numpy as np
from scipy import stats

np.random.seed(42)
n = 100

# Linear relationship
x_lin = np.random.normal(0, 1, n)
y_lin = 2 * x_lin + np.random.normal(0, 1, n)

# Monotonic nonlinear (exponential)
x_mono = np.random.uniform(0, 3, n)
y_mono = np.exp(x_mono) + np.random.normal(0, 2, n)

# Quadratic — r close to 0 but strong relationship
x_quad = np.random.normal(0, 2, n)
y_quad = x_quad**2 + np.random.normal(0, 1, n)

datasets = [
    ("Linear", x_lin, y_lin),
    ("Monotonic Nonlinear", x_mono, y_mono),
    ("Quadratic (r ~ 0)", x_quad, y_quad),
]

for name, x, y in datasets:
    r_p, _ = stats.pearsonr(x, y)
    r_s, _ = stats.spearmanr(x, y)
    r_k, _ = stats.kendalltau(x, y)
    print(f"{name:<25} r={r_p:.4f}  rho_s={r_s:.4f}  tau={r_k:.4f}")
```

For the quadratic case, Pearson's $r$ is near zero even though $y$ is a deterministic function of $x$. This illustrates that a low Pearson correlation does not mean "no relationship."

---

## Fisher z-Transformation and Confidence Interval

Pearson's $r$ has a skewed sampling distribution, especially when $|\rho|$ is large. The **Fisher $z$-transformation** stabilizes the variance:

$$
z = \operatorname{arctanh}(r) = \frac{1}{2}\ln\!\left(\frac{1+r}{1-r}\right)
$$

Under the null $\rho = \rho_0$, the transformed statistic is approximately normal:

$$
z \;\dot\sim\; \mathcal{N}\!\left(\operatorname{arctanh}(\rho_0),\; \frac{1}{n-3}\right)
$$

A $(1-\alpha)$ confidence interval for $\rho$ is obtained by inverting the transformation:

$$
\left(\tanh\!\bigl(z - z_{\alpha/2}\,\text{SE}\bigr),\;\; \tanh\!\bigl(z + z_{\alpha/2}\,\text{SE}\bigr)\right), \qquad \text{SE} = \frac{1}{\sqrt{n-3}}
$$

```python
def fisher_z_ci(x, y, alpha=0.05):
    n = len(x)
    r, p_val = stats.pearsonr(x, y)
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    z_lo, z_hi = z - z_crit * se, z + z_crit * se
    rho_lo, rho_hi = np.tanh(z_lo), np.tanh(z_hi)
    print(f"r = {r:.4f}, 95% CI for rho: ({rho_lo:.4f}, {rho_hi:.4f})")
    return rho_lo, rho_hi

np.random.seed(42)
x = np.random.normal(0, 1, 80)
y = 0.6 * x + np.random.normal(0, 0.8, 80)
fisher_z_ci(x, y)
```

---

## Simpson's Paradox

Simpson's paradox occurs when the direction of an association reverses after conditioning on a confounding variable. Formally, it is possible for:

$$
r(X, Y) > 0 \qquad \text{but} \qquad r(X, Y \mid Z = z) < 0 \;\;\text{for every } z
$$

This happens when a lurking variable $Z$ is positively associated with both $X$ and $Y$, creating a spurious positive overall correlation even though the within-group relationship is negative.

```python
np.random.seed(42)
groups = {"Group A": (50, 0.2, 8, -0.5),
          "Group B": (50, 0.5, 5, -0.5),
          "Group C": (50, 0.8, 2, -0.5)}

all_x, all_y = [], []
for name, (n, xm, yb, slope) in groups.items():
    x = np.random.normal(xm, 0.15, n)
    y = yb + slope * x + np.random.normal(0, 0.3, n)
    all_x.extend(x)
    all_y.extend(y)

all_x, all_y = np.array(all_x), np.array(all_y)
m_all, b_all = np.polyfit(all_x, all_y, 1)
r_overall, _ = stats.pearsonr(all_x, all_y)

print(f"Overall slope: {m_all:.2f} (positive)")
print(f"Within-group slope: -0.5 (negative)")
print(f"Overall r = {r_overall:.4f}")
```

The overall regression line has a positive slope, even though every within-group regression line has a negative slope. Ignoring the grouping variable leads to the wrong conclusion.

---

## Partial Correlation

Partial correlation removes the linear effect of a confounding variable $Z$ from both $X$ and $Y$. The first-order partial correlation is:

$$
r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ}\,r_{YZ}}{\sqrt{(1 - r_{XZ}^2)(1 - r_{YZ}^2)}}
$$

Equivalently, one can regress $X$ on $Z$ and $Y$ on $Z$, then compute the Pearson correlation of the residuals.

```python
n = 200
np.random.seed(42)

Z = np.random.normal(0, 1, n)
X = 0.7 * Z + np.random.normal(0, 0.5, n)
Y = 0.6 * Z + np.random.normal(0, 0.5, n)

r_xy, p_xy = stats.pearsonr(X, Y)
r_xz, _ = stats.pearsonr(X, Z)
r_yz, _ = stats.pearsonr(Y, Z)

r_xy_z = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2))

print(f"r(X, Y)    = {r_xy:.4f}  (appears significant)")
print(f"r(X,Y | Z) = {r_xy_z:.4f}  (nearly vanishes)")
```

Although $r(X, Y)$ is large, the partial correlation $r_{XY \cdot Z}$ is near zero, revealing that the apparent association is entirely driven by the confounder $Z$.

---

## Spurious Correlations from Multiple Testing

When many independent variables are tested pairwise, a substantial number of "significant" correlations arise purely by chance. With $p$ variables, there are $\binom{p}{2}$ pairs. At significance level $\alpha$, the expected number of false positives is:

$$
E[\text{false positives}] = \alpha \binom{p}{2}
$$

```python
def spurious_correlations_demo(n_vars=100, n_obs=30):
    data = np.random.normal(0, 1, (n_obs, n_vars))
    n_pairs = n_vars * (n_vars - 1) // 2
    p_values = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            _, p = stats.pearsonr(data[:, i], data[:, j])
            p_values.append(p)
    p_values = np.array(p_values)
    n_sig = np.sum(p_values < 0.05)
    print(f"Pairs tested: {n_pairs}")
    print(f"Significant at 0.05: {n_sig} ({100*n_sig/n_pairs:.1f}%)")
    print(f"Expected false positives: {0.05 * n_pairs:.0f}")

spurious_correlations_demo()
```

Even though all 100 variables are independent, roughly 5% of pairs appear significant. This is the multiple-testing problem, and corrections such as Bonferroni or Benjamini--Hochberg are needed.

---

## Comparing Two Independent Correlations

To test whether two population correlations are equal, $H_0\colon \rho_1 = \rho_2$, we use the Fisher $z$-transformation on each:

$$
z = \frac{\operatorname{arctanh}(r_1) - \operatorname{arctanh}(r_2)}{\sqrt{\dfrac{1}{n_1-3} + \dfrac{1}{n_2-3}}}
$$

Under $H_0$, $z$ is approximately standard normal.

```python
def compare_two_correlations(r1, n1, r2, n2, alpha=0.05):
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
    z_stat = (z1 - z2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    print(f"r1={r1:.4f} (n={n1}), r2={r2:.4f} (n={n2})")
    print(f"z = {z_stat:.4f}, p = {p_value:.4f}")

compare_two_correlations(r1=0.72, n1=100, r2=0.65, n2=120)
```

---

## Interpretation

The examples on this page illustrate a central principle of statistics: **correlation does not imply causation**. Specifically:

- A high Pearson $r$ only captures linear association. Nonlinear relationships or outliers can render it misleading.
- Simpson's paradox shows that aggregated data can reverse the sign of an association that holds within every subgroup.
- Partial correlation reveals that apparent associations may be entirely driven by confounders.
- Multiple testing generates spurious "significant" correlations from pure noise.

To establish causation, one needs either a randomized experiment or a carefully justified causal model (e.g., via instrumental variables or directed acyclic graphs).

---

## Exercises

**Exercise 1.** Generate $n = 150$ observations from the model $y = \sin(x) + \varepsilon$ where $x \sim \text{Uniform}(-\pi, \pi)$ and $\varepsilon \sim \mathcal{N}(0, 0.3^2)$. Compute Pearson's $r$, Spearman's $\rho_s$, and Kendall's $\tau$. Explain why all three are near zero even though $y$ is a deterministic function of $x$ (up to noise).

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(42)
    n = 150
    x = np.random.uniform(-np.pi, np.pi, n)
    y = np.sin(x) + np.random.normal(0, 0.3, n)

    r_p, _ = stats.pearsonr(x, y)
    r_s, _ = stats.spearmanr(x, y)
    r_k, _ = stats.kendalltau(x, y)
    print(f"Pearson r  = {r_p:.4f}")
    print(f"Spearman   = {r_s:.4f}")
    print(f"Kendall    = {r_k:.4f}")
    ```

    The sine function is symmetric about zero on $[-\pi, \pi]$: it increases on $[-\pi/2, \pi/2]$ and decreases on the tails. Over the full interval the positive and negative monotone segments cancel out, so there is no net linear or monotonic trend. All three coefficients measure monotonic or linear association and are therefore near zero despite the strong deterministic relationship. This is an example of a non-monotonic dependence that correlation cannot detect. $\square$

---

**Exercise 2.** A researcher finds $r = 0.45$ with $n = 50$ and claims the population correlation is "about 0.45." Construct a 99% confidence interval using the Fisher $z$-transformation and assess whether the researcher's point estimate is precise enough to support this claim.

??? success "Solution to Exercise 2"

    ```python
    import numpy as np
    from scipy import stats

    r = 0.45
    n = 50
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(0.995)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    print(f"99% CI: ({lo:.4f}, {hi:.4f})")
    ```

    The 99% confidence interval is approximately $(0.13, 0.68)$. This is quite wide: the true $\rho$ could be anywhere from a weak to a strong positive correlation. The point estimate of 0.45 is far from precise with only $n = 50$ observations; reporting just the point estimate without the interval is misleading. $\square$

---

**Exercise 3.** Construct a Simpson's paradox example with two groups. Let Group 1 have $n = 60$ observations with $x$-mean near 1 and $y$-intercept near 10, and Group 2 have $n = 60$ with $x$-mean near 4 and $y$-intercept near 2. Use a within-group slope of $-1$ for both. Verify that the overall slope is positive while both within-group slopes are negative.

??? success "Solution to Exercise 3"

    ```python
    import numpy as np

    np.random.seed(0)
    # Group 1: high baseline, low x
    x1 = np.random.normal(1, 0.3, 60)
    y1 = 10 - 1.0 * x1 + np.random.normal(0, 0.5, 60)

    # Group 2: low baseline, high x
    x2 = np.random.normal(4, 0.3, 60)
    y2 = 2 - 1.0 * x2 + np.random.normal(0, 0.5, 60)

    # Within-group slopes
    m1, _ = np.polyfit(x1, y1, 1)
    m2, _ = np.polyfit(x2, y2, 1)

    # Overall slope
    x_all = np.concatenate([x1, x2])
    y_all = np.concatenate([y1, y2])
    m_all, _ = np.polyfit(x_all, y_all, 1)

    print(f"Group 1 slope: {m1:.3f}")
    print(f"Group 2 slope: {m2:.3f}")
    print(f"Overall slope: {m_all:.3f}")
    ```

    Both within-group slopes are approximately $-1$ (negative), but the overall slope is positive because the group with higher $x$ values (Group 2) has lower $y$ values due to the intercept difference. The confounding group membership variable creates the paradox. $\square$

---

**Exercise 4.** Given the partial correlation formula

$$
r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ}\,r_{YZ}}{\sqrt{(1 - r_{XZ}^2)(1 - r_{YZ}^2)}}
$$

prove that $|r_{XY \cdot Z}| \le 1$. Under what conditions does $r_{XY \cdot Z} = 0$?

??? success "Solution to Exercise 4"

    Consider the residuals $e_X = X - \hat{X}_{Z}$ and $e_Y = Y - \hat{Y}_{Z}$ after regressing $X$ and $Y$ on $Z$. It is known that:

    $$
    r_{XY \cdot Z} = r(e_X, e_Y)
    $$

    Since $r(e_X, e_Y)$ is a Pearson correlation, it satisfies $|r(e_X, e_Y)| \le 1$ by the Cauchy--Schwarz inequality. Therefore $|r_{XY \cdot Z}| \le 1$.

    We have $r_{XY \cdot Z} = 0$ if and only if the residuals $e_X$ and $e_Y$ are uncorrelated, which happens when:

    $$
    r_{XY} = r_{XZ} \cdot r_{YZ}
    $$

    This means the entire marginal correlation between $X$ and $Y$ is explained by their mutual linear dependence on $Z$. $\square$

---

**Exercise 5.** Run a simulation: generate 200 independent standard normal variables with $n = 25$ observations each. Compute all $\binom{200}{2} = 19{,}900$ pairwise Pearson correlations. (a) How many are significant at $\alpha = 0.05$? (b) Apply the Bonferroni correction and report how many remain significant. (c) Discuss the tradeoff between Type I and Type II error in this context.

??? success "Solution to Exercise 5"

    ```python
    import numpy as np
    from scipy import stats

    np.random.seed(42)
    n_vars, n_obs = 200, 25
    data = np.random.normal(0, 1, (n_obs, n_vars))

    p_values = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            _, p = stats.pearsonr(data[:, i], data[:, j])
            p_values.append(p)

    p_values = np.array(p_values)
    n_pairs = len(p_values)
    n_sig = np.sum(p_values < 0.05)
    bonf_threshold = 0.05 / n_pairs
    n_sig_bonf = np.sum(p_values < bonf_threshold)

    print(f"Total pairs: {n_pairs}")
    print(f"Significant at 0.05: {n_sig}")
    print(f"Bonferroni threshold: {bonf_threshold:.2e}")
    print(f"Significant after Bonferroni: {n_sig_bonf}")
    ```

    (a) Approximately $0.05 \times 19{,}900 \approx 995$ pairs will appear significant by chance. (b) After the Bonferroni correction (threshold $\approx 2.5 \times 10^{-6}$), essentially none remain significant, which is correct since all variables are independent. (c) Bonferroni is conservative: it controls the family-wise error rate but reduces power. If a few genuine correlations existed among thousands of tests, Bonferroni might miss them. The Benjamini--Hochberg procedure, which controls the false discovery rate, provides a less conservative alternative. $\square$
