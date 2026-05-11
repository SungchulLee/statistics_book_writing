# Homogeneity Test (scipy)

## Overview

The **chi-square test of homogeneity** evaluates whether two or more populations share the same distribution across a set of categories. Although the computation is identical to the chi-square test of independence (both use `scipy.stats.chi2_contingency`), the study design and interpretation differ: rows represent independently sampled populations and we ask whether the column proportions are the same across those populations.

## Study Design Distinction

| Aspect | Independence | Homogeneity |
|--------|:------------|:-----------|
| Sampling | Single sample, two variables recorded | Separate sample from each population |
| Question | Are the two variables associated? | Do the populations have the same distribution? |
| Table rows | Levels of variable A | Populations |
| Table columns | Levels of variable B | Categories of the response |

Despite the different framing, the test statistic, degrees of freedom, and p-value computation are identical.

## Hypotheses

- **Null Hypothesis** ($H_0$): The distribution across categories is the same for all populations.
- **Alternative Hypothesis** ($H_A$): At least one population has a different distribution.

## Test Statistic

$$
\chi^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

where $E_{ij} = R_i C_j / n$ and $\text{df} = (r-1)(c-1)$.

## Code

```python
import numpy as np
from scipy import stats

# 3 populations (rows), 4 categories (columns)
observed = np.array([
    [25, 30, 20, 25],   # Population 1
    [18, 22, 35, 25],   # Population 2
    [30, 25, 15, 30],   # Population 3
], dtype=float)

chi2, p, df, expected = stats.chi2_contingency(observed, correction=False)

print("=== Chi-square Test of Homogeneity (scipy) ===")
print(f"chi2 = {chi2:.4f}, df = {df}, p = {p:.6f}")
print()
print("Expected counts under H0 (same proportions):")
print(expected)
```

**Key outputs:**

- `chi2_contingency` returns four values: the test statistic, the p-value, the degrees of freedom, and the matrix of expected counts.
- Setting `correction=False` ensures no Yates correction is applied (Yates is only relevant for $2 \times 2$ tables anyway).

## Expected Counts Under Homogeneity

Under $H_0$, each population has the same category proportions as the pooled (overall) proportions. The expected count for population $i$ in category $j$ is

$$
E_{ij} = n_i \cdot \hat{p}_j = n_i \cdot \frac{C_j}{n}
$$

where $n_i = R_i$ is the sample size for population $i$, $C_j$ is the total count in category $j$, and $n$ is the grand total. This is algebraically equivalent to $R_i C_j / n$.

## Interpretation

For the example data with 3 populations and 4 categories:

- $\text{df} = (3-1)(4-1) = 6$
- The test statistic and p-value determine whether the observed differences in category proportions across the three populations are larger than what we would expect from sampling variability alone.

If the p-value is less than $\alpha = 0.05$, we conclude that at least one population has a significantly different distribution of responses. The test does not tell us *which* population differs; post-hoc analysis (e.g., examining standardized residuals) is needed for that.

## Exercises

**1.** Two schools administer the same exam. The grade distributions are:

$$
\begin{array}{ccccc}
 & A & B & C & D \\
\text{School 1} & 20 & 35 & 30 & 15 \\
\text{School 2} & 25 & 25 & 35 & 15
\end{array}
$$

Compute the expected counts under $H_0$ (same grade distribution) and the chi-square statistic by hand.

??? success "Solution to Exercise 1"

    Row totals: $R_1 = 100$, $R_2 = 100$. Column totals: $C_A = 45$, $C_B = 60$, $C_C = 65$, $C_D = 30$. Grand total: $n = 200$.

    Since both row totals are equal, each expected count is simply $C_j / 2$:

    $$
    E = \begin{pmatrix} 22.5 & 30 & 32.5 & 15 \\ 22.5 & 30 & 32.5 & 15 \end{pmatrix}
    $$

    $$
    \chi^2 = \frac{(20-22.5)^2}{22.5} + \frac{(35-30)^2}{30} + \frac{(30-32.5)^2}{32.5} + \frac{(15-15)^2}{15}
    $$

    $$

    + \frac{(25-22.5)^2}{22.5} + \frac{(25-30)^2}{30} + \frac{(35-32.5)^2}{32.5} + \frac{(15-15)^2}{15}
    $$

    $$
    = 0.278 + 0.833 + 0.192 + 0 + 0.278 + 0.833 + 0.192 + 0 = 2.607
    $$

    With $\text{df} = (2-1)(4-1) = 3$. $\square$

---

**2.** Why is `correction=False` the appropriate choice for tables larger than $2 \times 2$?

??? success "Solution to Exercise 2"

    The Yates continuity correction was designed specifically for $2 \times 2$ tables, where the discrete distribution of the test statistic is being approximated by the continuous $\chi^2$ distribution with 1 degree of freedom. For larger tables, the discrete-to-continuous approximation is already quite good because the test statistic is a sum of many terms, and the $\chi^2$ approximation improves with more cells. Applying Yates correction to larger tables would make the test unnecessarily conservative (inflating the p-value), and `scipy.stats.chi2_contingency` only applies it to $2 \times 2$ tables even when `correction=True`. $\square$

---

**3.** Suppose you add a fourth population whose observed counts are identical to the pooled proportions (i.e., $O_{4j} = n_4 \cdot C_j / n$ for each category $j$). How does this affect the overall test statistic? Explain.

??? success "Solution to Exercise 3"

    The new population contributes zero to the chi-square statistic because $O_{4j} = E_{4j}$ for every category. Each term $(O_{4j} - E_{4j})^2 / E_{4j} = 0$. The overall statistic remains the same as before (the contributions from the original three populations do not change because the expected counts for the original populations are recomputed with the new marginals, but the new population exactly follows the pooled distribution so it does not disturb the overall proportions).

    More precisely, adding a population that perfectly matches the pooled proportions will slightly change the expected counts for the original populations (because $n$ and $C_j$ change), but the dominant effect is that the new rows contribute zero, and the original contributions shift only slightly. In the limiting case where $n_4$ is very small, the effect is negligible. $\square$

---

**4.** A researcher samples 50 people from each of 5 regions and records whether they prefer Brand X or Brand Y. The contingency table is

$$
\begin{pmatrix} 30 & 20 \\ 28 & 22 \\ 35 & 15 \\ 25 & 25 \\ 32 & 18 \end{pmatrix}
$$

Use `chi2_contingency` to test homogeneity at $\alpha = 0.05$. State your conclusion.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from scipy import stats

    observed = np.array([[30,20],[28,22],[35,15],[25,25],[32,18]], dtype=float)
    chi2, p, df, expected = stats.chi2_contingency(observed, correction=False)
    ```

    Row totals are all 50. Column totals: $C_1 = 150$, $C_2 = 100$. Grand total: $n = 250$. Overall proportions: $\hat{p}_1 = 0.6$, $\hat{p}_2 = 0.4$. Expected counts for each region: $(30, 20)$.

    $$
    \chi^2 = \frac{0}{30} + \frac{0}{20} + \frac{(28-30)^2}{30} + \frac{(22-20)^2}{20} + \frac{(35-30)^2}{30} + \frac{(15-20)^2}{20}
    $$

    $$

    + \frac{(25-30)^2}{30} + \frac{(25-20)^2}{20} + \frac{(32-30)^2}{30} + \frac{(18-20)^2}{20}
    $$

    $$
    = 0 + 0 + 0.133 + 0.2 + 0.833 + 1.25 + 0.833 + 1.25 + 0.133 + 0.2 = 4.833
    $$

    With $\text{df} = (5-1)(2-1) = 4$, the p-value is approximately $0.305$. Since $p > 0.05$, we **fail to reject** $H_0$. There is no significant evidence that brand preference differs across regions. $\square$

---

**5.** Prove that if all populations have exactly the same sample size $n_0$ and the same observed proportions in every category, then $\chi^2 = 0$ regardless of the number of populations or categories.

??? success "Solution to Exercise 5"

    Let there be $r$ populations each of size $n_0$, so $n = r \cdot n_0$. If every population has the same counts, then $O_{ij} = O_{1j}$ for all $i$, and the column total is $C_j = r \cdot O_{1j}$. The row total is $R_i = n_0$ for all $i$. The expected count is

    $$
    E_{ij} = \frac{R_i \cdot C_j}{n} = \frac{n_0 \cdot r \cdot O_{1j}}{r \cdot n_0} = O_{1j}
    $$

    Since all rows are identical, $O_{ij} = O_{1j} = E_{ij}$ for every cell. Therefore

    $$
    \chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} = \sum_{i,j} \frac{0}{E_{ij}} = 0
    $$

    This makes intuitive sense: if every population exhibits exactly the same distribution, there is zero evidence against homogeneity. $\square$
