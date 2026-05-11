# One-Way Analysis of Variance with scipy and Plots

## Overview

This page demonstrates how to perform a one-way ANOVA using SciPy's `f_oneway` function and how to visualize both the data distributions and the resulting $F$-statistic on its reference distribution. The PlantGrowth dataset is used to compare plant yields across a control and two treatment groups. Two plots -- a boxplot of group distributions and the $F$-distribution PDF with the observed tail shaded -- provide complementary views of the test.

## Data and Degrees of Freedom

The PlantGrowth dataset contains a response variable (weight) measured across $k = 3$ groups: ctrl, trt1, trt2. With $N = 30$ total observations, the degrees of freedom for the $F$-test are

$$
df_1 = k - 1 = 2, \qquad df_2 = N - k = 27
$$

```python
import pandas as pd
from scipy import stats

url = ('https://raw.githubusercontent.com/vincentarelbundock/'
       'Rdatasets/master/csv/datasets/PlantGrowth.csv')
df = pd.read_csv(url, usecols=[1, 2])
g = df.groupby('group')
ctrl = g.get_group('ctrl').weight.values
trt1 = g.get_group('trt1').weight.values
trt2 = g.get_group('trt2').weight.values
```

## Running the ANOVA

SciPy's `f_oneway` takes each group as a separate array and returns the $F$-statistic and $p$-value:

$$
F = \frac{MSB}{MSW} = \frac{SSB / (k-1)}{SSW / (N-k)}
$$

```python
F, p = stats.f_oneway(ctrl, trt1, trt2)
print(f"F = {F:.4f}, p = {p:.4f}")
```

Under $H_0: \mu_{\text{ctrl}} = \mu_{\text{trt1}} = \mu_{\text{trt2}}$, the statistic $F \sim F_{2,27}$.

## Visualization: Boxplot

A boxplot shows the median, interquartile range, and outliers for each group, providing an immediate sense of whether group centers and spreads differ.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([ctrl, trt1, trt2], labels=['ctrl', 'trt1', 'trt2'])
ax.set_xlabel('Group')
ax.set_ylabel('Weight')
ax.set_title('Plant weights by group')
plt.tight_layout()
plt.show()
```

## Visualization: F-Distribution with Observed Tail

Plotting the $F_{2,27}$ PDF and shading the area beyond the observed $F$-statistic gives a geometric interpretation of the $p$-value: it is the probability of observing an $F$-value at least as extreme under $H_0$.

```python
import numpy as np

x = np.linspace(0, 8, 400)
pdf = stats.f(2, 27).pdf(x)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, pdf, label='F(2, 27) PDF')
mask = x >= F
ax.fill_between(x[mask], pdf[mask], alpha=0.3, label='Observed tail')
ax.set_title('F-distribution and observed tail')
ax.legend()
plt.tight_layout()
plt.show()
```

The shaded area corresponds to

$$
p = P(F_{2,27} \ge F_{\text{obs}})
$$

## Interpretation

- **Boxplot:** If the boxes are vertically separated with little overlap, the group means likely differ. Overlapping boxes suggest weaker evidence against $H_0$.
- **F-distribution plot:** A large observed $F$ places the test statistic far into the right tail, producing a small $p$-value. The shaded region shrinks as $F_{\text{obs}}$ grows.
- **Decision:** If $p < \alpha$ (typically 0.05), reject $H_0$ and proceed with post-hoc comparisons (e.g., Tukey HSD) to identify which pairs differ.

## Exercises

**Exercise 1.**
For the PlantGrowth dataset ($k = 3$, $N = 30$), compute the critical value $F_{0.05,\, 2,\, 27}$ and determine whether $F_{\text{obs}} = 4.85$ leads to rejection of $H_0$.

??? success "Solution to Exercise 1"
    Using the $F$-distribution with $df_1 = 2$ and $df_2 = 27$:

    $$
    F_{0.05,\, 2,\, 27} \approx 3.35
    $$

    Since $F_{\text{obs}} = 4.85 > 3.35$, we reject $H_0$ at the $\alpha = 0.05$ significance level. There is significant evidence that at least one group mean differs.

---

**Exercise 2.**
Explain geometrically why the $F$-distribution is right-skewed and bounded below by zero. How do the degrees of freedom $df_1$ and $df_2$ affect the shape?

??? success "Solution to Exercise 2"
    The $F$-statistic is a ratio of two independent chi-squared random variables, each divided by its degrees of freedom:

    $$
    F = \frac{\chi^2_{df_1} / df_1}{\chi^2_{df_2} / df_2}
    $$

    Since chi-squared variables are non-negative, $F \ge 0$, and the distribution is bounded below by zero. The ratio of two positive quantities tends to concentrate near 1 (under $H_0$) but can take arbitrarily large values when the numerator is large, producing right skew.

    As $df_2 \to \infty$, the denominator converges to 1, and $F \to \chi^2_{df_1}/df_1$, which is still right-skewed but less so. As both $df_1$ and $df_2$ increase, the distribution becomes more symmetric and concentrated around 1. Small $df_1$ produces a more peaked distribution near zero (especially $df_1 = 1$ or $2$), while larger $df_1$ shifts the mode to the right.

---

**Exercise 3.**
SciPy's `f_oneway` assumes equal variances. If the group variances are $s_{\text{ctrl}}^2 = 0.25$, $s_{\text{trt1}}^2 = 0.64$, and $s_{\text{trt2}}^2 = 0.20$, is the equal-variance assumption reasonable? What alternative would you use?

??? success "Solution to Exercise 3"
    The ratio of the largest to smallest sample variance is $0.64 / 0.20 = 3.2$. A common rule of thumb is that the ANOVA $F$-test is robust when the largest variance is no more than about 3 to 4 times the smallest, provided group sizes are equal.

    Here the ratio is borderline. A formal Levene's test should be conducted. If it rejects equal variances, the appropriate alternative is Welch's ANOVA (`scipy.stats.alexandergovern` or `pingouin.welch_anova`), which uses a Satterthwaite-type degrees-of-freedom adjustment and does not assume homoscedasticity.

---

**Exercise 4.**
The $p$-value from `f_oneway` is computed as $p = 1 - F_{df_1, df_2}.\text{cdf}(F_{\text{obs}})$. Derive this from the definition of a $p$-value for a one-sided (right-tail) test and explain why ANOVA uses only the right tail of the $F$-distribution.

??? success "Solution to Exercise 4"
    The $p$-value is defined as the probability of observing a test statistic at least as extreme as $F_{\text{obs}}$ under $H_0$:

    $$
    p = P(F \ge F_{\text{obs}} \mid H_0) = 1 - P(F < F_{\text{obs}} \mid H_0) = 1 - F_{df_1, df_2}(F_{\text{obs}})
    $$

    where $F_{df_1, df_2}(\cdot)$ is the CDF of the $F$-distribution.

    ANOVA uses only the right tail because under the alternative hypothesis (at least one mean differs), the between-group variance $MSB$ increases while the within-group variance $MSW$ remains approximately equal to $\sigma^2$. This means $F = MSB/MSW$ can only increase, never decrease, relative to its null distribution. An unusually small $F$ does not provide evidence against $H_0$; it simply means the group means are similar. Therefore, only large values of $F$ constitute evidence against the null, making the test inherently one-sided (right-tailed).

---

**Exercise 5.**
Show that for $k = 2$ groups, `stats.f_oneway(x, y)` produces the same $p$-value as a two-sided two-sample $t$-test with equal variances. Hint: use the identity $t^2_{N-2} = F_{1, N-2}$.

??? success "Solution to Exercise 5"
    For $k = 2$ groups with sizes $n_1$ and $n_2$, the ANOVA $F$-statistic has $df_1 = 1$ and $df_2 = n_1 + n_2 - 2$. The between-group mean square is

    $$
    MSB = \frac{n_1 n_2}{n_1 + n_2}(\bar{x} - \bar{y})^2
    $$

    and the within-group mean square is the pooled variance $s_p^2$. Therefore

    $$
    F = \frac{MSB}{MSW} = \frac{n_1 n_2(\bar{x} - \bar{y})^2}{(n_1 + n_2)\, s_p^2}
    $$

    The pooled two-sample $t$-statistic is

    $$
    t = \frac{\bar{x} - \bar{y}}{s_p \sqrt{1/n_1 + 1/n_2}}
    $$

    Squaring gives

    $$
    t^2 = \frac{(\bar{x} - \bar{y})^2}{s_p^2 (1/n_1 + 1/n_2)} = \frac{n_1 n_2 (\bar{x} - \bar{y})^2}{(n_1 + n_2)\, s_p^2} = F
    $$

    Since $t^2_{N-2} \sim F_{1, N-2}$ and the $p$-value from the two-sided $t$-test is $P(|t| \ge |t_{\text{obs}}|) = P(t^2 \ge t_{\text{obs}}^2) = P(F_{1,N-2} \ge F_{\text{obs}})$, the two $p$-values are identical. $\square$
