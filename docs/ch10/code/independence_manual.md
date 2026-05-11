# Independence Test (Manual with Plot)

## Overview

This page walks through a **manual** computation of the chi-square test of independence for a two-way contingency table. Expected counts are derived from marginal totals, the test statistic and p-value are computed with NumPy and SciPy, and the result is visualized by plotting the $\chi^2$ probability density function with the rejection region shaded. Understanding the manual procedure clarifies what higher-level functions such as `scipy.stats.chi2_contingency` do behind the scenes.

## Hypotheses

- **Null Hypothesis** ($H_0$): The two categorical variables are independent.
- **Alternative Hypothesis** ($H_A$): The two categorical variables are not independent (they are associated).

## Expected Counts

For an $r \times c$ contingency table with observed counts $O_{ij}$, the expected count under independence is

$$
E_{ij} = \frac{R_i \cdot C_j}{n}
$$

where $R_i = \sum_j O_{ij}$ is the $i$-th row total, $C_j = \sum_i O_{ij}$ is the $j$-th column total, and $n$ is the grand total. In matrix notation:

$$
E = \frac{\mathbf{r}\,\mathbf{c}^\top}{n}
$$

where $\mathbf{r}$ and $\mathbf{c}$ are column vectors of row and column totals respectively.

## Test Statistic

$$
\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

Under $H_0$, this statistic approximately follows a $\chi^2$ distribution with

$$
\text{df} = (r - 1)(c - 1)
$$

degrees of freedom.

## Code

### Computing Expected Counts

```python
import numpy as np
from scipy import stats

def compute_expected(observed_counts: np.ndarray) -> np.ndarray:
    row_totals = observed_counts.sum(axis=1, keepdims=True)
    col_totals = observed_counts.sum(axis=0, keepdims=True)
    total = observed_counts.sum()
    return (row_totals @ col_totals) / total
```

The `keepdims=True` argument preserves the 2-D shape so that the matrix product `row_totals @ col_totals` broadcasts correctly into an $r \times c$ matrix of expected counts.

### Full Computation

```python
observed_counts = np.array([[934, 1070],
                            [113,   92],
                            [ 20,    8]], dtype=float)

expected_counts = compute_expected(observed_counts)
df = (observed_counts.shape[0] - 1) * (observed_counts.shape[1] - 1)

chi2 = np.sum((observed_counts - expected_counts)**2 / expected_counts)
p_value = stats.chi2(df).sf(chi2)

print(f"chi_squared_statistic = {chi2:.2f}")
print(f"p_value = {p_value:.2%}")
```

### Visualization

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 4))

x_left = np.linspace(0, chi2, 200)
y_left = stats.chi2(df).pdf(x_left)
ax.plot(x_left, y_left, linewidth=3)
x_fill = np.concatenate([[0], x_left, [chi2], [0]])
y_fill = np.concatenate([[0], y_left, [0], [0]])
ax.fill(x_fill, y_fill, alpha=0.1)

x_right = np.linspace(chi2, max(20, chi2 + 5), 200)
y_right = stats.chi2(df).pdf(x_right)
ax.plot(x_right, y_right, linewidth=3)
x_fill_r = np.concatenate([[chi2], x_right, [max(20, chi2 + 5)], [chi2]])
y_fill_r = np.concatenate([[0], y_right, [0], [0]])
ax.fill(x_fill_r, y_fill_r, alpha=0.1)

ax.annotate(f"p_value = {p_value:.2%}",
            xy=(chi2 * 0.8, y_left.max() * 0.15),
            xytext=(chi2 * 0.9 + 5, y_left.max() * 0.6),
            fontsize=12, arrowprops=dict(width=0.2, headwidth=8))

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_position("zero")
ax.spines["left"].set_position("zero")
plt.tight_layout()
plt.show()
```

## Interpretation

The observed $3 \times 2$ table yields a chi-square statistic of approximately $5.10$ with $\text{df} = (3-1)(2-1) = 2$. The p-value is roughly $0.078$, which exceeds the conventional $\alpha = 0.05$ threshold. We therefore **fail to reject** $H_0$ and conclude that there is insufficient evidence of an association between the row and column variables at the 5% level.

The plot makes the decision visually clear: the observed statistic lies near the boundary of the right tail, but the shaded area (p-value) is not small enough to warrant rejection.

## Exercises

**1.** Given the $2 \times 2$ table

$$
\begin{pmatrix} 20 & 30 \\ 40 & 10 \end{pmatrix}
$$

compute the expected counts matrix manually using the formula $E_{ij} = R_i C_j / n$.

??? success "Solution to Exercise 1"

    Row totals: $R_1 = 50$, $R_2 = 50$. Column totals: $C_1 = 60$, $C_2 = 40$. Grand total: $n = 100$.

    $$
    E_{11} = \frac{50 \times 60}{100} = 30, \quad E_{12} = \frac{50 \times 40}{100} = 20
    $$

    $$
    E_{21} = \frac{50 \times 60}{100} = 30, \quad E_{22} = \frac{50 \times 40}{100} = 20
    $$

    So the expected counts matrix is

    $$
    E = \begin{pmatrix} 30 & 20 \\ 30 & 20 \end{pmatrix}
    $$

    $\square$

---

**2.** Using the expected counts from Exercise 1, compute the chi-square statistic and the degrees of freedom. At $\alpha = 0.05$, would you reject $H_0$?

??? success "Solution to Exercise 2"

    $$
    \chi^2 = \frac{(20-30)^2}{30} + \frac{(30-20)^2}{20} + \frac{(40-30)^2}{30} + \frac{(10-20)^2}{20}
    $$

    $$
    = \frac{100}{30} + \frac{100}{20} + \frac{100}{30} + \frac{100}{20} = 3.333 + 5 + 3.333 + 5 = 16.667
    $$

    Degrees of freedom: $\text{df} = (2-1)(2-1) = 1$.

    The critical value $\chi^2_{0.05, 1} = 3.841$. Since $16.667 > 3.841$, we **reject** $H_0$. The two variables are significantly associated. $\square$

---

**3.** Explain why `keepdims=True` is necessary in the `compute_expected` function. What error would occur without it?

??? success "Solution to Exercise 3"

    Without `keepdims=True`, `sum(axis=1)` returns a 1-D array of shape $(r,)$ and `sum(axis=0)` returns a 1-D array of shape $(c,)$. The matrix product `@` of two 1-D arrays produces a scalar (inner product), not the desired $r \times c$ outer-product matrix.

    With `keepdims=True`, the shapes become $(r, 1)$ and $(1, c)$ respectively. The product of an $(r, 1)$ matrix and a $(1, c)$ matrix is an $(r, c)$ matrix, which is exactly the outer product needed for the expected counts. $\square$

---

**4.** Show that for any contingency table, the sum of expected counts equals the sum of observed counts, i.e., $\sum_{i,j} E_{ij} = n$.

??? success "Solution to Exercise 4"

    By definition, $E_{ij} = R_i C_j / n$. Summing over all cells:

    $$
    \sum_{i=1}^{r}\sum_{j=1}^{c} E_{ij} = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{R_i C_j}{n} = \frac{1}{n}\sum_{i=1}^{r} R_i \sum_{j=1}^{c} C_j = \frac{1}{n} \cdot n \cdot n = n
    $$

    The second step uses the fact that $R_i$ does not depend on $j$ and can be factored out of the inner sum. The final step uses $\sum_i R_i = \sum_j C_j = n$. $\square$

---

**5.** Prove that the degrees of freedom for the chi-square test of independence is $(r-1)(c-1)$ by counting the number of free parameters in the expected-count table.

??? success "Solution to Exercise 5"

    Under $H_0$ (independence), the joint probability of cell $(i,j)$ factors as $p_{ij} = p_{i\cdot} \cdot p_{\cdot j}$. The row marginal probabilities have $r - 1$ free parameters (since they must sum to 1), and the column marginal probabilities have $c - 1$ free parameters. Under independence the total number of free parameters is therefore $(r-1) + (c-1)$.

    The unrestricted model for the $r \times c$ table has $rc - 1$ free cell probabilities. The degrees of freedom for the test is the difference:

    $$
    \text{df} = (rc - 1) - [(r - 1) + (c - 1)] = rc - 1 - r - c + 2 = rc - r - c + 1 = (r-1)(c-1)
    $$

    Equivalently, once the row and column marginal totals are fixed (which is what the expected counts enforce), the number of freely varying cells in the table is $(r-1)(c-1)$. $\square$
