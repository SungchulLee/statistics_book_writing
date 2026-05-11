# Homogeneity Residual Heatmap

## Overview

After a chi-square test of homogeneity rejects the null hypothesis, a natural follow-up question is: **which cells are responsible for the departure from homogeneity?** This page demonstrates post-hoc diagnostics using standardized (Pearson) residuals, Bonferroni-corrected per-cell significance tests, and a heatmap visualization. The approach helps pinpoint which population-category combinations deviate most from the expected distribution.

## Standardized Residuals

The **Pearson standardized residual** for cell $(i, j)$ is

$$
R_{ij} = \frac{O_{ij} - E_{ij}}{\sqrt{E_{ij}}}
$$

Under $H_0$, each $R_{ij}$ is approximately standard normal for large samples. A residual with $|R_{ij}| > 2$ suggests that cell $(i,j)$ contributes notably to the overall chi-square statistic.

Note that the overall chi-square statistic is simply the sum of squared residuals:

$$
\chi^2 = \sum_{i,j} R_{ij}^2
$$

## Per-Cell Significance with Bonferroni Correction

Each standardized residual can be treated as an approximate $z$-score. The two-sided p-value for cell $(i,j)$ is

$$
p_{ij} = 2\bigl[1 - \mathcal{N}(|R_{ij}|)\bigr]
$$

where $\mathcal{N}$ is the standard normal CDF. Since we test all $r \times c$ cells simultaneously, we apply a **Bonferroni correction** to control the family-wise error rate:

$$
p_{ij}^{\text{Bonf}} = \min\bigl(r \cdot c \cdot p_{ij},\; 1\bigr)
$$

A cell is flagged as significant if $p_{ij}^{\text{Bonf}} < \alpha$.

## Code

### Computing Residuals and Adjusted p-Values

```python
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

observed = np.array([
    [25, 30, 20, 25],
    [18, 22, 35, 25],
    [30, 25, 15, 30],
], dtype=float)

row_tot = observed.sum(axis=1, keepdims=True)
col_tot = observed.sum(axis=0, keepdims=True)
tot = observed.sum()
expected = (row_tot @ col_tot) / tot

# Standardized residuals (Pearson)
resid = (observed - expected) / np.sqrt(expected)

# Per-cell z-tests (approximate), two-sided
z = resid.ravel()
pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
reject, pvals_bonf, _, _ = multipletests(pvals, method="bonferroni")
pvals_bonf = pvals_bonf.reshape(observed.shape)
reject = reject.reshape(observed.shape)

print("Standardized residuals:")
print(resid)
print()
print("Bonferroni-adjusted per-cell p-values:")
print(pvals_bonf)
```

### Heatmap Visualization

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(resid, aspect="auto")
ax.set_title("Standardized residuals heatmap")
ax.set_xlabel("Category")
ax.set_ylabel("Population")
plt.colorbar(im, ax=ax, shrink=0.8)

# Annotate significant cells after Bonferroni
for i in range(observed.shape[0]):
    for j in range(observed.shape[1]):
        mark = "*" if reject[i, j] else ""
        ax.text(j, i, f"{resid[i, j]:.2f}{mark}",
                ha="center", va="center", fontsize=10)

plt.tight_layout()
plt.show()
```

Cells marked with `*` are statistically significant after Bonferroni correction. The color gradient makes it easy to spot which cells have the largest positive (over-represented) or negative (under-represented) residuals.

## Interpretation

The heatmap provides an at-a-glance summary of where the observed data diverges from the expected pattern under homogeneity. Key takeaways:

- **Positive residuals** (warm colors) indicate that a population has more observations in that category than expected.
- **Negative residuals** (cool colors) indicate fewer observations than expected.
- The asterisk (`*`) marks cells where the departure is statistically significant even after adjusting for multiple comparisons.

This kind of post-hoc analysis is essential because the overall chi-square test only tells us *that* the populations differ, not *how* they differ.

## Exercises

**1.** Given observed counts $O = 40$ and expected counts $E = 25$, compute the standardized residual and the unadjusted two-sided p-value.

??? success "Solution to Exercise 1"

    $$
    R = \frac{O - E}{\sqrt{E}} = \frac{40 - 25}{\sqrt{25}} = \frac{15}{5} = 3.0
    $$

    The two-sided p-value is

    $$
    p = 2[1 - \mathcal{N}(3.0)] = 2 \times 0.00135 = 0.0027
    $$

    This cell shows a highly significant over-representation even before any multiple-comparison adjustment. $\square$

---

**2.** A $4 \times 3$ table has 12 cells. After computing per-cell p-values, the smallest unadjusted p-value is $0.006$. Is this cell significant at $\alpha = 0.05$ after Bonferroni correction?

??? success "Solution to Exercise 2"

    The Bonferroni-adjusted p-value is

    $$
    p^{\text{Bonf}} = 12 \times 0.006 = 0.072
    $$

    Since $0.072 > 0.05$, the cell is **not** significant after Bonferroni correction, despite having a small unadjusted p-value. This illustrates how conservative Bonferroni can be when the number of comparisons is large. $\square$

---

**3.** Explain the difference between a **standardized residual** $R_{ij} = (O_{ij} - E_{ij})/\sqrt{E_{ij}}$ and an **adjusted standardized residual** $R_{ij}^{\text{adj}} = (O_{ij} - E_{ij})/\sqrt{E_{ij}(1 - R_i/n)(1 - C_j/n)}$. Which one has a distribution closer to $N(0,1)$ under $H_0$?

??? success "Solution to Exercise 3"

    The Pearson standardized residual divides by $\sqrt{E_{ij}}$, which is only an approximation to the standard deviation of $O_{ij} - E_{ij}$ under $H_0$. The true variance of the residual also depends on the marginal totals through the factor $(1 - R_i/n)(1 - C_j/n)$.

    The **adjusted standardized residual** incorporates this correction:

    $$
    R_{ij}^{\text{adj}} = \frac{O_{ij} - E_{ij}}{\sqrt{E_{ij}(1 - R_i/n)(1 - C_j/n)}}
    $$

    Under $H_0$, $R_{ij}^{\text{adj}}$ has a distribution closer to $N(0,1)$ than the unadjusted residual. For per-cell hypothesis testing, the adjusted residual is therefore preferred, though the unadjusted version is still commonly used for exploratory heatmaps. $\square$

---

**4.** Why is Bonferroni correction described as "conservative"? Name one alternative multiple-comparison method and describe how it differs.

??? success "Solution to Exercise 4"

    Bonferroni correction is conservative because it controls the **family-wise error rate** (FWER) by dividing the significance level $\alpha$ equally among all comparisons, effectively using $\alpha / m$ as the threshold for each of $m$ tests. When many tests are conducted, this threshold becomes very small, making it difficult to reject any individual hypothesis even when a real effect exists (low power).

    An alternative is the **Benjamini-Hochberg (BH) procedure**, which controls the **false discovery rate** (FDR) instead of the FWER. The FDR is the expected proportion of false positives among all rejected hypotheses. The BH method ranks the p-values, then rejects all hypotheses whose p-value falls below $p_{(i)} \le (i/m)\alpha$ for a suitable cutoff $i$. This is less conservative than Bonferroni, providing more statistical power at the cost of allowing a controlled fraction of false discoveries. In `statsmodels`, it is available via `multipletests(pvals, method="fdr_bh")`. $\square$

---

**5.** Prove that $\sum_{i,j} R_{ij}^2 = \chi^2$, i.e., the chi-square statistic equals the sum of squared standardized residuals.

??? success "Solution to Exercise 5"

    By definition, the standardized residual is $R_{ij} = (O_{ij} - E_{ij}) / \sqrt{E_{ij}}$. Squaring:

    $$
    R_{ij}^2 = \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
    $$

    Summing over all cells:

    $$
    \sum_{i=1}^{r}\sum_{j=1}^{c} R_{ij}^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} = \chi^2
    $$

    This is exactly the definition of the Pearson chi-square statistic. The decomposition shows that $\chi^2$ aggregates contributions from every cell, and the heatmap of $R_{ij}$ (or $R_{ij}^2$) reveals how that aggregate is distributed across the table. $\square$
