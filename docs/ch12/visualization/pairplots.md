# Pair Plots and Scatter Matrices

When exploring a dataset with multiple quantitative variables, examining each pair of variables individually is essential for understanding their relationships. A **pair plot** (also called a **scatter matrix**) displays all pairwise scatter plots in a single grid, providing a comprehensive overview of the bivariate relationships in the data. This is one of the most important exploratory data analysis tools for multivariate datasets.

---

## What Is a Pair Plot

A pair plot for $p$ variables creates a $p \times p$ grid of panels:

- **Off-diagonal panels**: scatter plots of each pair of variables $(X_i, X_j)$.
- **Diagonal panels**: univariate plots (histograms, kernel density estimates, or box plots) for each variable.

For $p$ variables, there are $\binom{p}{2} = p(p-1)/2$ unique pairs. The grid displays each pair twice (once in the upper triangle and once in the lower triangle, with axes transposed), so some implementations show different plot types in the upper and lower triangles.

---

## Reading a Pair Plot

When examining a pair plot, look for:

1. **Direction of association**: do the points slope upward (positive) or downward (negative)?
2. **Strength of association**: how tightly do the points cluster around a trend? Tight clustering indicates strong correlation.
3. **Linearity**: is the trend approximately linear, or is there curvature?
4. **Outliers**: are there points far from the main cluster?
5. **Clusters**: do the points form distinct groups, suggesting subpopulations?
6. **Heteroscedasticity**: does the spread of $Y$ change across the range of $X$?

The diagonal panels reveal the marginal distributions -- their shapes (symmetric, skewed, bimodal) inform the choice of correlation measure and statistical test.

---

## Basic Pair Plot with Seaborn

The `seaborn` library provides the `pairplot` function, which creates publication-quality pair plots with minimal code.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load example dataset
df = sns.load_dataset("iris")

# Basic pair plot
sns.pairplot(df, hue="species", diag_kind="kde")
plt.suptitle("Iris Dataset: Pairwise Relationships by Species", y=1.02)
plt.show()
```

In this plot:

- Each off-diagonal panel shows a scatter plot colored by species.
- Each diagonal panel shows kernel density estimates (KDE) of the marginal distributions, separated by species.
- The `hue` parameter colors points by a categorical variable, revealing whether relationships differ across groups.

---

## Customizing Pair Plots

### Selecting Variables

For datasets with many columns, plotting all pairs may produce an overwhelming grid. Select a subset of variables:

```python
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("iris")

# Select specific variables
sns.pairplot(
    df,
    vars=["sepal_length", "sepal_width", "petal_length"],
    hue="species",
    diag_kind="hist",
    plot_kws={"alpha": 0.6},
)
plt.show()
```

### Correlation Coefficients in the Plot

Adding numerical correlation values to each panel helps quantify what the scatter plots show visually:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = sns.load_dataset("iris")
numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

g = sns.PairGrid(df[numeric_cols])
g.map_lower(sns.scatterplot)
g.map_diag(sns.histplot, kde=True)

def annotate_corr(x, y, **kwargs):
    r = np.corrcoef(x, y)[0, 1]
    ax = plt.gca()
    ax.annotate(f"r = {r:.2f}", xy=(0.5, 0.5),
                xycoords="axes fraction", ha="center",
                fontsize=14, fontweight="bold")

g.map_upper(annotate_corr)
plt.show()
```

This version displays scatter plots in the lower triangle, histograms on the diagonal, and Pearson $r$ values in the upper triangle.

---

## When to Use Pair Plots

Pair plots are most useful when:

- The number of variables is moderate ($p \le 10$). For very large $p$, the grid becomes unwieldy.
- You are in the exploratory phase of analysis and want to survey all pairwise relationships quickly.
- You suspect that relationships may vary across subgroups (use the `hue` parameter).
- You want to check regression assumptions (linearity, homoscedasticity) before fitting models.

For high-dimensional data ($p > 10$), consider:

- Selecting the most important variables based on domain knowledge.
- Using a **correlation heatmap** (see [Heatmaps](heatmaps.md)) to identify the strongest relationships, then examining those specific pairs in detail.
- Dimensionality reduction techniques (PCA, t-SNE) for an overview.

---

## Pair Plots vs Correlation Matrices

| Feature | Pair plot | Correlation matrix / heatmap |
|:---|:---|:---|
| Shows nonlinear patterns | Yes | No |
| Shows outliers | Yes | No |
| Shows clusters | Yes | No |
| Scales to many variables | Poorly ($p > 10$) | Well ($p > 50$) |
| Provides a number | No (unless annotated) | Yes ($r$ for each pair) |

The pair plot and correlation heatmap are complementary tools. The heatmap summarizes many relationships compactly, while the pair plot reveals the details that a single number cannot capture.

---

## Interpretation Pitfalls

!!! warning "Do not over-interpret patterns in small samples"
    With few data points, scatter plots can show apparent patterns that are due entirely to chance. Always consider the sample size when interpreting pair plots.

!!! warning "Pair plots show marginal relationships only"
    Each panel shows the bivariate relationship between two variables, ignoring all others. A strong marginal correlation may weaken or reverse after conditioning on other variables (see [Simpson's Paradox](../ecological_correlation/simpsons_paradox.md) and [Partial Correlation](../correlation/partial.md)).

---

## Summary

Pair plots provide a comprehensive visual summary of all pairwise relationships in a multivariate dataset. They reveal the direction, strength, and shape of associations, as well as outliers and clusters. The `seaborn.pairplot` function makes them easy to create, with options for coloring by groups, customizing diagonal plots, and annotating with correlation coefficients. Pair plots are most effective for datasets with a moderate number of variables and should be paired with correlation heatmaps and formal statistical tests for a complete analysis.
