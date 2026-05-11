# Post-Hoc Comparison Examples

## Overview

When a one-way ANOVA rejects the null hypothesis, it tells us that at least one group mean differs from the rest, but it does not identify which pairs of means are different. Post-hoc comparison procedures fill this gap by performing pairwise tests while controlling the family-wise error rate. This page reviews the most common methods -- Tukey's HSD, Bonferroni correction, and Scheffe's method -- with worked examples and simulated data.

## The Multiple-Comparisons Problem

With $k$ groups there are $\binom{k}{2}$ pairwise comparisons. If each test is conducted at significance level $\alpha$, the probability of at least one false rejection under the global null is

$$
1 - (1 - \alpha)^{\binom{k}{2}}
$$

For $k = 5$ groups and $\alpha = 0.05$ there are 10 comparisons, giving a family-wise error rate of roughly $1 - 0.95^{10} \approx 0.40$. Post-hoc methods keep this inflated error under control.

## Tukey's Honest Significant Difference

Tukey's HSD declares groups $i$ and $j$ significantly different when

$$
|\bar{y}_{i\cdot} - \bar{y}_{j\cdot}| > q_{\alpha,\, k,\, N-k} \sqrt{\frac{MSW}{n}}
$$

where $q_{\alpha,k,N-k}$ is the critical value of the studentized range distribution, $MSW$ is the mean square within groups, and $n$ is the common group size (for balanced designs).

```python
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

np.random.seed(42)
n = 100
data = np.random.normal(loc=0, scale=1, size=n)
```

For unbalanced designs, the Tukey-Kramer modification replaces $\sqrt{MSW/n}$ with $\sqrt{MSW \cdot (1/n_i + 1/n_j)/2}$.

## Bonferroni Correction

The Bonferroni method adjusts each pairwise $p$-value by multiplying it by the number of comparisons $m = \binom{k}{2}$:

$$
p_{\text{adj}} = \min(m \cdot p_{\text{raw}},\; 1)
$$

This is simple and widely applicable but can be conservative when $m$ is large. The Holm-Bonferroni step-down variant is uniformly more powerful while still controlling the family-wise error rate.

## Scheffe's Method

Scheffe's procedure controls the family-wise error rate for all possible linear contrasts, not just pairwise differences. The critical value is based on the $F$-distribution:

$$
|\bar{y}_{i\cdot} - \bar{y}_{j\cdot}| > \sqrt{(k-1)\, F_{\alpha,\, k-1,\, N-k}} \cdot \sqrt{MSW \left(\frac{1}{n_i} + \frac{1}{n_j}\right)}
$$

Scheffe's method is the most conservative of the three for pairwise comparisons because it controls error over a broader family of contrasts.

## Comparison of Methods

| Method | Controls error for | Power | Best used when |
|---|---|---|---|
| Tukey HSD | All pairwise comparisons | High | All pairwise comparisons are of interest |
| Bonferroni | Any pre-specified set | Moderate | Few planned comparisons |
| Scheffe | All possible contrasts | Low (for pairs) | Complex contrasts are of interest |

## Interpretation

- **Tukey HSD** is the default choice when you want to compare every pair of group means. It is exact for balanced designs and approximate (Tukey-Kramer) for unbalanced designs.
- **Bonferroni** is most useful when only a small number of planned comparisons are needed, since its power decreases as the number of comparisons grows.
- **Scheffe** is the method of choice when you are interested in arbitrary linear contrasts (e.g., comparing the average of two groups against a third). For purely pairwise comparisons, Tukey HSD has higher power.

## Exercises

**Exercise 1.**
A one-way ANOVA with $k = 4$ groups and $n_i = 15$ per group yields $MSW = 12.3$. The studentized range critical value at $\alpha = 0.05$ is $q_{0.05,4,56} = 3.74$. Compute the Tukey HSD threshold and determine whether a mean difference of $\bar{y}_1 - \bar{y}_3 = 3.5$ is significant.

??? success "Solution to Exercise 1"
    The Tukey HSD threshold for balanced designs is

    $$
    \text{HSD} = q_{\alpha,k,N-k} \sqrt{\frac{MSW}{n}} = 3.74 \sqrt{\frac{12.3}{15}} = 3.74 \sqrt{0.82} = 3.74 \times 0.9055 \approx 3.39
    $$

    Since $|\bar{y}_1 - \bar{y}_3| = 3.5 > 3.39$, the difference is significant at the $\alpha = 0.05$ level.

---

**Exercise 2.**
A researcher plans $m = 6$ pairwise comparisons (from $k = 4$ groups) using the Bonferroni method at family-wise $\alpha = 0.05$. The raw $p$-values are $0.003, 0.012, 0.041, 0.078, 0.210, 0.530$. Which comparisons are significant under Bonferroni? Which additional comparisons become significant under the Holm-Bonferroni step-down procedure?

??? success "Solution to Exercise 2"
    **Bonferroni:** Multiply each raw $p$-value by $m = 6$.

    | Raw $p$ | Bonferroni $p_{\text{adj}}$ | Significant? |
    |---|---|---|
    | 0.003 | 0.018 | Yes |
    | 0.012 | 0.072 | No |
    | 0.041 | 0.246 | No |
    | 0.078 | 0.468 | No |
    | 0.210 | 1.000 | No |
    | 0.530 | 1.000 | No |

    Only the first comparison is significant under Bonferroni.

    **Holm-Bonferroni:** Sort the raw $p$-values in ascending order. Compare $p_{(j)}$ to $\alpha / (m - j + 1)$:

    - $p_{(1)} = 0.003 < 0.05/6 = 0.00833$ -- reject.
    - $p_{(2)} = 0.012 < 0.05/5 = 0.01$ -- not rejected (since $0.012 > 0.01$). Stop.

    Under Holm-Bonferroni, only the first comparison is significant. However, if the second raw $p$-value were slightly smaller (e.g., $0.009$), it would also be rejected, illustrating that Holm is strictly more powerful than Bonferroni.

---

**Exercise 3.**
Show that for $k = 2$ groups the Tukey HSD test is equivalent to the two-sample $t$-test. Specifically, prove that $q_{\alpha,2,\nu}^2 = 2\, F_{\alpha,1,\nu}$ where $\nu = N - 2$.

??? success "Solution to Exercise 3"
    For $k = 2$ groups the studentized range distribution with parameters $(2, \nu)$ relates to the $t$-distribution by $q_{2,\nu} = \sqrt{2}\, |t_\nu|$. Squaring both sides gives $q_{2,\nu}^2 = 2\, t_\nu^2$. Since $t_\nu^2 \sim F_{1,\nu}$, we have

    $$
    q_{\alpha,2,\nu}^2 = 2\, F_{\alpha,1,\nu}
    $$

    The Tukey HSD test rejects when $|\bar{y}_1 - \bar{y}_2| / \sqrt{MSW/n} > q_{\alpha,2,\nu}$, which is equivalent to

    $$
    \frac{(\bar{y}_1 - \bar{y}_2)^2}{MSW/n} > 2\, F_{\alpha,1,\nu}
    $$

    The left side equals $2 F_{\text{obs}}$ where $F_{\text{obs}} = MSB/MSW$ is the standard ANOVA $F$-statistic for $k = 2$. Hence the Tukey test reduces to rejecting when $F_{\text{obs}} > F_{\alpha,1,\nu}$, which is exactly the two-sample $t$-test (equivalently, the $F$-test with $k-1 = 1$ numerator degree of freedom). $\square$

---

**Exercise 4.**
Explain why Scheffe's method is more conservative than Tukey's HSD for pairwise comparisons but can detect effects that Tukey cannot. Give a concrete example of a contrast that Scheffe can test but Tukey cannot.

??? success "Solution to Exercise 4"
    Scheffe's method controls the family-wise error rate over **all possible linear contrasts** $\psi = \sum c_i \mu_i$ with $\sum c_i = 0$, not just pairwise differences. Since this family is much larger than the set of pairwise comparisons, the critical value must be larger, making the test more conservative for any single pairwise comparison.

    However, Scheffe's method can test complex contrasts such as:

    $$
    \psi = \frac{\mu_1 + \mu_2}{2} - \mu_3
    $$

    This contrast asks whether the average of groups 1 and 2 differs from group 3. Tukey's HSD is not designed for this type of comparison. For example, in a study comparing three teaching methods, a researcher might want to test whether the average effect of two lecture-based methods differs from a project-based method. Scheffe's method handles this directly, while Tukey cannot.

---

**Exercise 5.**
Prove that the Bonferroni correction controls the family-wise error rate at level $\alpha$. That is, if each of $m$ tests is conducted at level $\alpha/m$, show that $P(\text{at least one false rejection under } H_0) \le \alpha$.

??? success "Solution to Exercise 5"
    Let $R_j$ be the event that the $j$-th null hypothesis is falsely rejected, for $j = 1, \ldots, m$. Each test is conducted at level $\alpha/m$, so $P(R_j) \le \alpha/m$. By Boole's inequality (the union bound),

    $$
    P\!\left(\bigcup_{j=1}^{m} R_j\right) \le \sum_{j=1}^{m} P(R_j) \le \sum_{j=1}^{m} \frac{\alpha}{m} = \alpha
    $$

    This holds regardless of the dependence structure among the tests. The bound is tight when the tests are perfectly positively correlated and becomes conservative when the tests are independent or weakly correlated. $\square$
