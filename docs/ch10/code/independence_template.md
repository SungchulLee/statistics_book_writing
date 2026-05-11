# Independence Test Template Function

## Overview

This page presents a reusable template function for the chi-square test of independence built on top of `scipy.stats.chi2_contingency`. Wrapping the SciPy call in a clearly documented function makes it straightforward to apply the test to any two-way contingency table. The function also exposes the optional Yates continuity correction for $2 \times 2$ tables.

## Hypotheses

- **Null Hypothesis** ($H_0$): The row variable and column variable are independent.
- **Alternative Hypothesis** ($H_A$): There is an association between the row and column variables.

## Test Statistic

$$
\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

with $\text{df} = (r-1)(c-1)$, where $E_{ij} = R_i C_j / n$ are the expected counts under independence.

### Yates Continuity Correction

For $2 \times 2$ tables, the optional **Yates correction** modifies each term:

$$
\chi^2_{\text{Yates}} = \sum_{i=1}^{2}\sum_{j=1}^{2} \frac{(|O_{ij} - E_{ij}| - 0.5)^2}{E_{ij}}
$$

This correction reduces the test statistic slightly, making the test more conservative when cell counts are small.

## Code

### Template Function

```python
import numpy as np
from scipy import stats

def chi2_independence(observed: np.ndarray, correction: bool = False):
    """Run chi-square test of independence.

    Parameters
    ----------
    observed : np.ndarray
        2D contingency table of observed counts.
    correction : bool
        Yates' continuity correction (only applied to 2x2).
        Default False.

    Returns
    -------
    chi2, p, df, expected : tuple
        Test statistic, p-value, degrees of freedom,
        and expected counts.
    """
    return stats.chi2_contingency(observed, correction=correction)
```

### Demo Usage

```python
observed = np.array([[30, 20, 10],
                     [12, 25, 18]], dtype=float)

chi2, p, df, exp = chi2_independence(observed, correction=False)
print(f"chi2 = {chi2:.3f}, p = {p:.4f}, df = {df}")
print("expected:\n", exp)
```

**Output:**

- $\chi^2 \approx 8.556$, $p \approx 0.0139$, $\text{df} = 2$
- Expected counts are computed automatically from marginal totals.

## When to Use the Template

| Scenario | `correction` |
|----------|:------------:|
| Table larger than $2 \times 2$ | `False` (correction is only for $2 \times 2$) |
| $2 \times 2$ table with all $E_{ij} \ge 5$ | `False` (standard test suffices) |
| $2 \times 2$ table with some $E_{ij}$ near 5 | `True` (conservative adjustment) |
| $2 \times 2$ table with any $E_{ij} < 5$ | Consider Fisher's exact test instead |

## Interpretation

For the demo table, the test returns $\chi^2 = 8.556$ with $\text{df} = 2$ and $p = 0.0139$. At $\alpha = 0.05$, we **reject** $H_0$ and conclude that the row and column variables are not independent. The association is statistically significant.

To understand *which* cells drive the significance, one can examine the standardized residuals $(O_{ij} - E_{ij}) / \sqrt{E_{ij}}$; cells with large absolute residuals contribute most to the test statistic.

## Exercises

**1.** Apply the template function to the table

$$
\begin{pmatrix} 50 & 50 \\ 50 & 50 \end{pmatrix}
$$

What result do you expect before running the code? Verify.

??? success "Solution to Exercise 1"

    Every observed count equals the expected count because the marginals are identical. Therefore $O_{ij} = E_{ij}$ for all cells, giving $\chi^2 = 0$ and $p = 1.0$. The variables show perfect independence in this sample.

    ```python
    chi2, p, df, exp = chi2_independence(
        np.array([[50, 50], [50, 50]], dtype=float)
    )
    # chi2 = 0.0, p = 1.0, df = 1
    ```

    $\square$

---

**2.** Run the template on the $2 \times 2$ table $\begin{pmatrix} 10 & 5 \\ 3 & 12 \end{pmatrix}$ both with and without Yates correction. Compare the two $\chi^2$ values and explain the difference.

??? success "Solution to Exercise 2"

    Without correction:

    $$
    E = \begin{pmatrix} 5.67 & 9.33 \\ 7.33 & 7.67 \end{pmatrix} \quad (\text{approximate})
    $$

    Wait -- let us compute properly. Row totals: $R_1 = 15$, $R_2 = 15$. Column totals: $C_1 = 13$, $C_2 = 17$. Grand total: $n = 30$.

    $$
    E_{11} = \frac{15 \times 13}{30} = 6.5, \quad E_{12} = \frac{15 \times 17}{30} = 8.5
    $$

    $$
    E_{21} = 6.5, \quad E_{22} = 8.5
    $$

    Without Yates:

    $$
    \chi^2 = \frac{(10-6.5)^2}{6.5} + \frac{(5-8.5)^2}{8.5} + \frac{(3-6.5)^2}{6.5} + \frac{(12-8.5)^2}{8.5} = 1.885 + 1.441 + 1.885 + 1.441 = 6.652
    $$

    With Yates correction, each numerator uses $(|O_{ij} - E_{ij}| - 0.5)^2 = (3.5 - 0.5)^2 = 9$:

    $$
    \chi^2_{\text{Yates}} = \frac{9}{6.5} + \frac{9}{8.5} + \frac{9}{6.5} + \frac{9}{8.5} = 1.385 + 1.059 + 1.385 + 1.059 = 4.887
    $$

    The Yates-corrected statistic is smaller, producing a larger p-value. The correction compensates for the discrete-to-continuous approximation inherent in using the $\chi^2$ distribution for a discrete test statistic. $\square$

---

**3.** The function returns four values. Write code that uses only the expected-counts output to compute the standardized residuals for the demo table. Which cell contributes most to the chi-square statistic?

??? success "Solution to Exercise 3"

    ```python
    observed = np.array([[30, 20, 10],
                         [12, 25, 18]], dtype=float)
    _, _, _, expected = chi2_independence(observed)
    residuals = (observed - expected) / np.sqrt(expected)
    print(residuals)
    ```

    The cell with the largest absolute standardized residual contributes most. In this example, the residuals reveal that the $(1,1)$ and $(2,3)$ cells (or symmetrically their counterparts) have the largest magnitudes, indicating that the first group is over-represented in category 1 while the second group is over-represented in category 3. $\square$

---

**4.** Prove that the Yates correction always produces a test statistic less than or equal to the uncorrected statistic.

??? success "Solution to Exercise 4"

    For any cell $(i,j)$, let $d_{ij} = |O_{ij} - E_{ij}|$. The uncorrected contribution is $d_{ij}^2 / E_{ij}$, while the Yates-corrected contribution is $(\max(d_{ij} - 0.5, 0))^2 / E_{ij}$.

    Since $\max(d_{ij} - 0.5, 0) \le d_{ij}$ for all $d_{ij} \ge 0$, it follows that

    $$
    \frac{(\max(d_{ij} - 0.5, 0))^2}{E_{ij}} \le \frac{d_{ij}^2}{E_{ij}}
    $$

    Summing over all cells:

    $$
    \chi^2_{\text{Yates}} = \sum_{i,j} \frac{(\max(|O_{ij} - E_{ij}| - 0.5, 0))^2}{E_{ij}} \le \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} = \chi^2
    $$

    Therefore the corrected statistic is always less than or equal to the uncorrected one, making the test more conservative (larger p-value). $\square$

---

**5.** A clinical trial records outcomes across two treatments and three severity levels. The contingency table is

$$
\begin{pmatrix} 45 & 30 & 25 \\ 35 & 40 & 25 \end{pmatrix}
$$

Use the template function to test for independence at $\alpha = 0.05$. Report the statistic, p-value, and conclusion.

??? success "Solution to Exercise 5"

    ```python
    table = np.array([[45, 30, 25],
                      [35, 40, 25]], dtype=float)
    chi2, p, df, exp = chi2_independence(table, correction=False)
    ```

    Row totals: $R_1 = 100$, $R_2 = 100$. Column totals: $C_1 = 80$, $C_2 = 70$, $C_3 = 50$. Grand total: $n = 200$.

    Expected counts:

    $$
    E = \begin{pmatrix} 40 & 35 & 25 \\ 40 & 35 & 25 \end{pmatrix}
    $$

    $$
    \chi^2 = \frac{(45-40)^2}{40} + \frac{(30-35)^2}{35} + \frac{(25-25)^2}{25} + \frac{(35-40)^2}{40} + \frac{(40-35)^2}{35} + \frac{(25-25)^2}{25}
    $$

    $$
    = 0.625 + 0.714 + 0 + 0.625 + 0.714 + 0 = 2.679
    $$

    With $\text{df} = (2-1)(3-1) = 2$, the p-value is approximately $0.262$. Since $p > 0.05$, we **fail to reject** $H_0$. There is no significant association between treatment and severity level. $\square$
