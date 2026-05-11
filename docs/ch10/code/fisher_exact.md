# Fisher's Exact Test (2x2)

## Overview

**Fisher's exact test** computes the exact probability of observing a $2 \times 2$ contingency table as extreme as (or more extreme than) the one obtained, given fixed marginal totals. Unlike the chi-square test, it does not rely on a large-sample approximation and is therefore the preferred method when sample sizes are small or expected cell counts fall below 5.

## When to Use Fisher's Exact Test

- The contingency table is $2 \times 2$.
- One or more expected cell counts are less than 5.
- The total sample size is small (roughly $n < 20$--$30$).
- You want an exact p-value rather than an asymptotic approximation.

For larger tables or larger sample sizes, the chi-square test of independence is computationally simpler and provides an excellent approximation.

## Hypotheses

- **Null Hypothesis** ($H_0$): The row variable and column variable are independent (i.e., the odds ratio equals 1).
- **Alternative Hypothesis** ($H_A$): The row and column variables are associated (i.e., the odds ratio is not equal to 1).

## Mathematical Foundation

Consider a $2 \times 2$ table with fixed marginal totals:

$$
\begin{array}{c|cc|c}
 & \text{Col 1} & \text{Col 2} & \text{Row Total} \\
\hline
\text{Row 1} & a & b & a+b \\
\text{Row 2} & c & d & c+d \\
\hline
\text{Col Total} & a+c & b+d & n
\end{array}
$$

Under $H_0$, the cell count $a$ follows a **hypergeometric distribution**. The probability of observing a specific table (given the marginals) is

$$
P(a) = \frac{\binom{a+b}{a}\binom{c+d}{c}}{\binom{n}{a+c}}
$$

The p-value sums the probabilities of all tables that are as extreme as or more extreme than the observed table.

## Odds Ratio

The **sample odds ratio** measures the strength of association:

$$
\text{OR} = \frac{a \cdot d}{b \cdot c}
$$

- $\text{OR} = 1$: no association (consistent with $H_0$).
- $\text{OR} > 1$: Row 1 is more likely to be in Column 1 than Row 2 is.
- $\text{OR} < 1$: Row 1 is less likely to be in Column 1 than Row 2 is.

## Code

```python
import numpy as np
from scipy import stats

# 2x2 contingency table
#               Success   Failure
# Treatment        1         5
# Control          8         2
observed = np.array([[1, 5],
                     [8, 2]])

print("Observed contingency table:")
print(observed)

# Fisher's exact test
odds_ratio, p_value = stats.fisher_exact(observed)

print(f"Odds ratio : {odds_ratio:.4f}")
print(f"p-value    : {p_value:.4f}")

if p_value < 0.05:
    print("Reject H0: significant association (alpha = 0.05).")
else:
    print("Fail to reject H0: no significant association (alpha = 0.05).")
```

**Output:**

- Odds ratio: $\text{OR} = (1 \times 2) / (5 \times 8) = 0.05$
- The p-value is computed exactly using the hypergeometric distribution.

## Interpretation

For the treatment-vs-control example:

- The odds ratio of approximately $0.05$ indicates that the treatment group had dramatically lower odds of success compared to the control group.
- If the p-value is less than $0.05$, we conclude a statistically significant association between group membership and outcome.

Fisher's exact test is especially important in medical and biological studies where sample sizes per group may be very small (e.g., rare diseases, pilot studies).

## Exercises

**1.** For the table $\begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$, compute the odds ratio and use `stats.fisher_exact` to find the exact p-value. Is the result significant at $\alpha = 0.05$?

??? success "Solution to Exercise 1"

    $$
    \text{OR} = \frac{3 \times 3}{1 \times 1} = 9.0
    $$

    ```python
    odds_ratio, p_value = stats.fisher_exact([[3, 1], [1, 3]])
    ```

    The exact p-value (two-sided) is approximately $0.486$. Since $p > 0.05$, we **fail to reject** $H_0$. Despite the seemingly large odds ratio, the sample size ($n = 8$) is too small to achieve significance. $\square$

---

**2.** Explain why Fisher's exact test is preferred over the chi-square test when expected cell counts are below 5.

??? success "Solution to Exercise 2"

    The chi-square test approximates the distribution of the test statistic with a continuous $\chi^2$ distribution. This approximation relies on the Central Limit Theorem and is accurate only when expected cell counts are sufficiently large (the common rule of thumb is $E_{ij} \ge 5$). When expected counts are small, the actual distribution of the test statistic deviates substantially from $\chi^2$, leading to unreliable p-values.

    Fisher's exact test avoids any approximation. It enumerates all possible tables with the same marginal totals and computes the exact probability of each under $H_0$ using the hypergeometric distribution. The resulting p-value is exact regardless of sample size, making it the appropriate choice when the chi-square approximation cannot be trusted. $\square$

---

**3.** Write out the hypergeometric probability for the observed table $\begin{pmatrix} 1 & 5 \\ 8 & 2 \end{pmatrix}$ using the combinatorial formula.

??? success "Solution to Exercise 3"

    The marginals are: $R_1 = 6$, $R_2 = 10$, $C_1 = 9$, $C_2 = 7$, $n = 16$. The hypergeometric probability of observing $a = 1$ in cell $(1,1)$ is

    $$
    P(a = 1) = \frac{\binom{6}{1}\binom{10}{8}}{\binom{16}{9}}
    $$

    Computing each binomial coefficient:

    $$
    \binom{6}{1} = 6, \quad \binom{10}{8} = \binom{10}{2} = 45, \quad \binom{16}{9} = \binom{16}{7} = 11440
    $$

    $$
    P(a = 1) = \frac{6 \times 45}{11440} = \frac{270}{11440} \approx 0.0236
    $$

    The two-sided p-value sums the probabilities of all tables with $a \le 1$ plus those with $a$ at the upper extreme that are equally or less probable. $\square$

---

**4.** The `alternative` parameter in `stats.fisher_exact` can be `"two-sided"`, `"less"`, or `"greater"`. What does each option test in terms of the odds ratio?

??? success "Solution to Exercise 4"

    - `"two-sided"` (default): Tests $H_A: \text{OR} \ne 1$. The p-value includes tables that are as extreme or more extreme in either direction.
    - `"less"`: Tests $H_A: \text{OR} < 1$. The p-value includes only tables where the odds ratio is less than or equal to the observed value.
    - `"greater"`: Tests $H_A: \text{OR} > 1$. The p-value includes only tables where the odds ratio is greater than or equal to the observed value.

    In our treatment example with $\text{OR} = 0.05$, using `alternative="less"` would test whether the treatment has significantly lower odds of success than the control. $\square$

---

**5.** Prove that for a $2 \times 2$ table with all marginal totals fixed, the value of any single cell determines the entire table.

??? success "Solution to Exercise 5"

    Let the table be

    $$
    \begin{pmatrix} a & b \\ c & d \end{pmatrix}
    $$

    with row totals $R_1 = a + b$ and $R_2 = c + d$, and column totals $C_1 = a + c$ and $C_2 = b + d$ all fixed. Suppose we know $a$. Then:

    - $b = R_1 - a$ (determined by the first row total)
    - $c = C_1 - a$ (determined by the first column total)
    - $d = R_2 - c = R_2 - (C_1 - a) = R_2 - C_1 + a$ (determined by the second row total)

    All four cells are determined by the single value $a$. This is why the hypergeometric distribution for $a$ alone is sufficient to characterize the exact distribution of the entire table under $H_0$, and it explains why the test has only one degree of freedom. $\square$
