# Cochran's Q Test (k Related Outcomes)

## Overview

**Cochran's Q test** is a generalization of McNemar's test to $k \ge 2$ related dichotomous (binary) outcomes measured on the same subjects. It tests whether the proportion of "successes" is the same across all $k$ conditions. The test is commonly used in repeated-measures designs where each subject is assessed under multiple treatments, tasks, or time points with a binary outcome.

## Study Design

- $n$ subjects are each observed under $k$ conditions.
- The outcome for each subject-condition pair is binary (0 or 1).
- The data form an $n \times k$ matrix $X$ where $X_{ij} \in \{0, 1\}$.

## Hypotheses

- **Null Hypothesis** ($H_0$): The success probability is the same across all $k$ conditions, i.e., $p_1 = p_2 = \cdots = p_k$.
- **Alternative Hypothesis** ($H_A$): At least one condition has a different success probability.

## Test Statistic

Let $T_j = \sum_{i=1}^{n} X_{ij}$ be the column total (number of successes in condition $j$) and $L_i = \sum_{j=1}^{k} X_{ij}$ be the row total (number of successes for subject $i$). Let $T = \sum_j T_j$ be the grand total. Cochran's Q statistic is

$$
Q = \frac{(k-1)\left(k \sum_{j=1}^{k} T_j^2 - T^2\right)}{k\,T - \sum_{i=1}^{n} L_i^2}
$$

Under $H_0$, $Q$ approximately follows a $\chi^2(k-1)$ distribution for large $n$.

## Relationship to McNemar's Test

When $k = 2$, Cochran's Q reduces to McNemar's test. Specifically, the Q statistic with $k = 2$ equals the McNemar chi-square statistic (without continuity correction). This makes Cochran's Q the natural extension for comparing more than two related binary outcomes.

## Code

### Implementation

```python
import numpy as np
from scipy import stats

def cochran_q(data):
    """
    Perform Cochran's Q test.

    Parameters
    ----------
    data : array-like, shape (n_subjects, k_conditions)
        Binary (0/1) matrix. Each row is a subject,
        each column a condition/task.

    Returns
    -------
    Q       : float   Cochran's Q statistic
    p_value : float   p-value from chi-square(k-1) approximation
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape

    T_j = data.sum(axis=0)          # column totals
    L_i = data.sum(axis=1)          # row totals
    grand_T = T_j.sum()

    numerator = (k - 1) * (k * np.sum(T_j**2) - grand_T**2)
    denominator = k * grand_T - np.sum(L_i**2)

    Q = numerator / denominator
    p_value = stats.chi2(k - 1).sf(Q)
    return Q, p_value
```

### Running the Test

```python
# 12 subjects rated on 3 tasks (pass=1, fail=0)
tasks = np.array([
    [0, 1, 0],
    [1, 1, 0],
    [1, 1, 1],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 1],
    [0, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
])

Q, p = cochran_q(tasks)

print(f"Cochran's Q = {Q:.4f}")
print(f"p-value     = {p:.4f}")

if p < 0.05:
    print("Reject H0: success rates differ across tasks.")
else:
    print("Fail to reject H0: no significant difference.")
```

**Summary of the example data:**

- Column totals (successes per task): $T_1 = 4$, $T_2 = 9$, $T_3 = 2$.
- Task 2 has the highest success rate ($9/12 = 75\%$) while Task 3 has the lowest ($2/12 \approx 17\%$).
- The Q statistic quantifies whether these differences are larger than expected by chance.

## Interpretation

If the p-value is less than $\alpha = 0.05$, we conclude that the success probabilities are not equal across all $k$ conditions. Cochran's Q does **not** tell us which specific conditions differ. Post-hoc pairwise comparisons (e.g., multiple McNemar tests with Bonferroni correction) are needed to identify which pairs of conditions have significantly different success rates.

The chi-square approximation for Q is generally adequate when:

- The number of subjects $n$ is reasonably large.
- The product $nk$ is large enough that the distribution of Q is well-approximated by $\chi^2(k-1)$.

A common guideline is $n \ge 4$ and $nk \ge 24$.

## Exercises

**1.** For the example data above, verify the column totals $T_1 = 4$, $T_2 = 9$, $T_3 = 2$ and compute the grand total $T$.

??? success "Solution to Exercise 1"

    Summing column 1: $0+1+1+0+1+0+0+1+0+0+0+0 = 4$. Check.

    Summing column 2: $1+1+1+0+0+1+0+1+1+1+1+1 = 9$. Check.

    Summing column 3: $0+0+1+0+0+1+0+0+0+0+0+0 = 2$. Check.

    Grand total: $T = 4 + 9 + 2 = 15$. $\square$

---

**2.** Using the values from Exercise 1, compute the row totals $L_i$ for all 12 subjects and then evaluate $\sum_{i=1}^{12} L_i^2$.

??? success "Solution to Exercise 2"

    Row totals: $L = [1, 2, 3, 0, 1, 2, 0, 2, 1, 1, 1, 1]$.

    $$
    \sum_{i=1}^{12} L_i^2 = 1 + 4 + 9 + 0 + 1 + 4 + 0 + 4 + 1 + 1 + 1 + 1 = 27
    $$

    $\square$

---

**3.** Substitute the values from Exercises 1 and 2 into the Cochran's Q formula and verify the test statistic.

??? success "Solution to Exercise 3"

    We have $k = 3$, $T = 15$, $\sum T_j^2 = 4^2 + 9^2 + 2^2 = 16 + 81 + 4 = 101$, and $\sum L_i^2 = 27$.

    Numerator:

    $$
    (k-1)\left(k\sum T_j^2 - T^2\right) = 2 \times (3 \times 101 - 225) = 2 \times (303 - 225) = 2 \times 78 = 156
    $$

    Denominator:

    $$
    k \cdot T - \sum L_i^2 = 3 \times 15 - 27 = 45 - 27 = 18
    $$

    $$
    Q = \frac{156}{18} = 8.667
    $$

    With $\text{df} = k - 1 = 2$, $p = P(\chi^2_2 \ge 8.667) \approx 0.013$. Since $p < 0.05$, we reject $H_0$ and conclude that the success rates differ significantly across the three tasks. $\square$

---

**4.** After rejecting $H_0$ with Cochran's Q, a researcher performs all $\binom{k}{2}$ pairwise McNemar tests to identify which conditions differ. For $k = 4$ conditions, how many pairwise tests are needed, and what is the Bonferroni-adjusted significance level if the overall $\alpha = 0.05$?

??? success "Solution to Exercise 4"

    The number of pairwise comparisons is

    $$
    \binom{4}{2} = \frac{4!}{2!\,2!} = 6
    $$

    With Bonferroni correction, each individual test uses a significance level of

    $$
    \alpha^* = \frac{0.05}{6} \approx 0.00833
    $$

    A pairwise McNemar test is significant only if its p-value is less than $0.00833$. This controls the family-wise error rate at $0.05$. $\square$

---

**5.** Show that when $k = 2$, Cochran's Q statistic reduces to the uncorrected McNemar statistic $(b - c)^2 / (b + c)$.

??? success "Solution to Exercise 5"

    With $k = 2$ columns, let the data for subject $i$ be $(X_{i1}, X_{i2})$. Define the discordant pair counts: $b$ = number of subjects with $(X_{i1}, X_{i2}) = (1, 0)$ and $c$ = number with $(X_{i1}, X_{i2}) = (0, 1)$. Also let $a$ = number with $(1,1)$ and $d$ = number with $(0,0)$.

    Column totals: $T_1 = a + b$ and $T_2 = a + c$, so $T = 2a + b + c$.

    Row totals: subjects with $L_i = 2$ contribute $a$ subjects, $L_i = 1$ contributes $b + c$ subjects, and $L_i = 0$ contributes $d$ subjects. Therefore

    $$
    \sum L_i^2 = 4a + (b + c) + 0 = 4a + b + c
    $$

    Numerator of Q:

    $$
    (2-1)\bigl[2(T_1^2 + T_2^2) - T^2\bigr] = 2\bigl[(a+b)^2 + (a+c)^2\bigr] - (2a+b+c)^2
    $$

    Expanding:

    $$
    = 2(a^2 + 2ab + b^2 + a^2 + 2ac + c^2) - (4a^2 + b^2 + c^2 + 4ab + 4ac + 2bc)
    $$

    $$
    = (4a^2 + 4ab + 2b^2 + 4ac + 2c^2) - (4a^2 + 4ab + 4ac + b^2 + c^2 + 2bc)
    $$

    $$
    = b^2 + c^2 - 2bc = (b - c)^2
    $$

    Denominator of Q:

    $$
    2T - \sum L_i^2 = 2(2a + b + c) - (4a + b + c) = b + c
    $$

    Therefore

    $$
    Q = \frac{(b-c)^2}{b+c}
    $$

    which is exactly the McNemar statistic without continuity correction. $\square$
