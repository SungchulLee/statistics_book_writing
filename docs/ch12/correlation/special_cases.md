# Point-Biserial and Phi Coefficients

The Pearson correlation coefficient is defined for two continuous variables, but many practical situations involve binary (dichotomous) variables. When one or both variables are binary, the Pearson formula still applies and produces special-case coefficients with their own names and interpretations. The **point-biserial coefficient** handles the case of one binary and one continuous variable, while the **phi coefficient** handles two binary variables.

---

## Point-Biserial Correlation

### Motivation

Suppose we want to measure the association between a binary group variable (e.g., treatment vs. control) and a continuous outcome (e.g., test score). The point-biserial correlation quantifies how much the continuous variable differs between the two groups, expressed on the familiar $[-1, 1]$ scale.

### Definition

Let the binary variable be coded as $X \in \{0, 1\}$, with $n_0$ observations in group 0 and $n_1$ observations in group 1, where $n = n_0 + n_1$. Let $Y$ be the continuous variable. The **point-biserial correlation** is

$$
r_{pb} = \frac{\bar{Y}_1 - \bar{Y}_0}{S_Y} \sqrt{\frac{n_0 \, n_1}{n^2}}
$$

where $\bar{Y}_0$ and $\bar{Y}_1$ are the group means of $Y$, and $S_Y$ is the overall sample standard deviation of $Y$.

This is algebraically identical to the Pearson correlation between $X$ and $Y$ when $X$ takes only the values 0 and 1.

### Connection to the Two-Sample t-Test

The point-biserial correlation is directly related to the two-sample t-test statistic. If $t$ is the equal-variance two-sample t-statistic with $n - 2$ degrees of freedom, then

$$
r_{pb} = \frac{t}{\sqrt{t^2 + (n - 2)}}
$$

This means testing $H_0\!: r_{pb} = 0$ is equivalent to testing $H_0\!: \mu_0 = \mu_1$ via the two-sample t-test.

### Interpretation

- $r_{pb} > 0$: group 1 tends to have higher values of $Y$.
- $r_{pb} < 0$: group 0 tends to have higher values of $Y$.
- $r_{pb} = 0$: no difference in means between the groups.
- $r_{pb}^2$: the proportion of variance in $Y$ explained by group membership.

---

## Phi Coefficient

### Motivation

When both variables are binary (e.g., gender and pass/fail), we need a measure of association for a $2 \times 2$ contingency table. The **phi coefficient** is the Pearson correlation between two binary variables and provides a natural measure of the strength of association.

### Definition

Consider a $2 \times 2$ table:

|  | $Y = 1$ | $Y = 0$ | Total |
|:---:|:---:|:---:|:---:|
| $X = 1$ | $a$ | $b$ | $a + b$ |
| $X = 0$ | $c$ | $d$ | $c + d$ |
| Total | $a + c$ | $b + d$ | $n$ |

The **phi coefficient** is

$$
\phi = \frac{ad - bc}{\sqrt{(a+b)(c+d)(a+c)(b+d)}}
$$

This is algebraically identical to the Pearson correlation computed on the 0/1 coded data for $X$ and $Y$.

### Connection to Chi-Square

The phi coefficient is related to the Pearson chi-square statistic for a $2 \times 2$ table:

$$
\chi^2 = n \, \phi^2
$$

Therefore

$$
\phi = \sqrt{\frac{\chi^2}{n}}
$$

with an appropriate sign determined by the direction of association ($ad > bc$ yields $\phi > 0$).

Testing $H_0\!: \phi = 0$ is equivalent to the chi-square test of independence for the $2 \times 2$ table.

### Properties

1. **Range.** $-1 \le \phi \le 1$, but $\phi$ can only achieve $\pm 1$ when the marginal distributions of $X$ and $Y$ are equal (i.e., $a + b = a + c$ and $c + d = b + d$).

2. **Maximum attainable value.** When the marginals are unequal, $|\phi|$ is bounded below $1$. The maximum depends on the marginal proportions.

3. **Symmetry.** $\phi_{XY} = \phi_{YX}$.

---

## Example: Point-Biserial

A researcher assigns 10 students to a study group ($X = 1$) or no intervention ($X = 0$) and records their exam scores:

| Group ($X$) | Scores ($Y$) |
|:---:|:---|
| 0 | 65, 70, 68, 72, 66 |
| 1 | 78, 82, 85, 80, 76 |

Computing: $\bar{Y}_0 = 68.2$, $\bar{Y}_1 = 80.2$, $S_Y = 6.76$, $n_0 = n_1 = 5$, $n = 10$.

$$
r_{pb} = \frac{80.2 - 68.2}{6.76} \sqrt{\frac{5 \times 5}{100}} = \frac{12.0}{6.76} \times 0.5 = 0.888
$$

The strong positive $r_{pb}$ indicates that the study group scored substantially higher.

---

## Example: Phi Coefficient

A survey asks 200 people about exercise habits ($X$: exercises regularly) and sleep quality ($Y$: reports good sleep):

|  | Good sleep ($Y=1$) | Poor sleep ($Y=0$) | Total |
|:---:|:---:|:---:|:---:|
| Exercises ($X=1$) | 60 | 30 | 90 |
| No exercise ($X=0$) | 40 | 70 | 110 |
| Total | 100 | 100 | 200 |

$$
\phi = \frac{(60)(70) - (30)(40)}{\sqrt{(90)(110)(100)(100)}} = \frac{4200 - 1200}{\sqrt{99{,}000{,}000}} = \frac{3000}{9949.87} \approx 0.301
$$

There is a moderate positive association between regular exercise and good sleep quality.

---

## Related Measures for Larger Tables

The phi coefficient is specific to $2 \times 2$ tables. For larger contingency tables, related measures include:

- **Cramer's V**: generalizes phi to $r \times c$ tables via $V = \sqrt{\chi^2 / (n \cdot \min(r-1, c-1))}$.
- **Contingency coefficient**: $C = \sqrt{\chi^2 / (\chi^2 + n)}$, bounded above by a value less than 1.

These are covered in the chapter on [chi-square tests](../../ch10/test/independence.md).

---

## Computation in Python

```python
import numpy as np
from scipy import stats

# Point-biserial example
group = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
scores = np.array([65, 70, 68, 72, 66, 78, 82, 85, 80, 76])

r_pb, p_val = stats.pointbiserialr(group, scores)
print(f"Point-biserial r = {r_pb:.4f}, p-value = {p_val:.4f}")

# This equals Pearson r on the same data
r_pearson, _ = stats.pearsonr(group, scores)
print(f"Pearson r        = {r_pearson:.4f}")

# Phi coefficient via chi-square
table = np.array([[60, 30], [40, 70]])
chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
phi = np.sqrt(chi2 / table.sum())
print(f"Phi coefficient  = {phi:.4f}")
```

---

## Summary

The point-biserial and phi coefficients are special cases of the Pearson correlation for binary data. The point-biserial coefficient measures the association between a binary grouping variable and a continuous outcome, and its test is equivalent to the two-sample t-test. The phi coefficient measures the association between two binary variables in a $2 \times 2$ table, and its test is equivalent to the chi-square test of independence. Recognizing these connections unifies several seemingly distinct statistical procedures under the single framework of correlation.
