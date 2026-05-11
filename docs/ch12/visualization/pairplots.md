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

## Exercises

**Exercise 1.**
Describe what a pair plot (scatter matrix) shows and explain what to look for when interpreting one with 4 variables.

??? success "Solution to Exercise 1"
    A pair plot is a grid of scatter plots showing every pairwise combination of variables. For $p = 4$ variables, it is a $4 \times 4$ grid with $\binom{4}{2} = 6$ unique scatter plots (the matrix is symmetric). The diagonal panels typically show univariate distributions (histograms or KDE plots) for each variable.

    When interpreting a pair plot with 4 variables, look for:

    1. **Linear vs. nonlinear relationships:** Are scatter plots approximately linear, or do they show curvature?
    2. **Strength and direction of association:** Tight clouds indicate strong correlation; dispersed clouds indicate weak correlation.
    3. **Outliers:** Points far from the main cloud in any panel.
    4. **Clusters:** Groups of points that may indicate subpopulations.
    5. **Heteroscedasticity:** Fan-shaped scatter (variance changing with the level of one variable).
    6. **Marginal distributions:** Skewness, multimodality, or heavy tails visible in diagonal panels.

---

**Exercise 2.**
Write Python code to create a pair plot for the Iris dataset using seaborn, colored by species.

??? success "Solution to Exercise 2"
    ```python
    import seaborn as sns
    import matplotlib.pyplot as plt

    iris = sns.load_dataset("iris")
    g = sns.pairplot(iris, hue="species", diag_kind="kde")
    g.fig.suptitle("Iris Dataset Pair Plot", y=1.02)
    plt.show()
    ```

    The `hue="species"` parameter colors points by species, revealing whether the pairwise relationships differ across groups. The KDE on the diagonal shows each species' distribution for each measurement. Separation between colors in the scatter panels indicates which variable pairs best discriminate between species.

---

**Exercise 3.**
Why can a pair plot be misleading when the number of variables $p$ is large (e.g., $p > 10$)? Suggest an alternative approach.

??? success "Solution to Exercise 3"
    With $p > 10$, a pair plot has $p^2 > 100$ panels, making it:

    1. **Visually overwhelming:** Too many panels to inspect individually. Important patterns are lost in the grid.
    2. **Computationally expensive:** Rendering hundreds of scatter plots with thousands of points is slow.
    3. **Statistically limited:** Pairwise scatter plots miss higher-dimensional structure (e.g., three variables may be jointly correlated in ways invisible in any pair).

    **Alternatives for high-dimensional data:**

    - **Correlation heatmap:** Summarizes all pairwise correlations in a single colored matrix.
    - **PCA or t-SNE:** Reduce to 2-3 dimensions and visualize the reduced representation.
    - **Focused pair plots:** Select the 5-6 most important variables (based on domain knowledge or correlation screening) and create a pair plot of those.

---

**Exercise 4.**
How can you use a pair plot to visually detect multicollinearity in a regression context?

??? success "Solution to Exercise 4"
    Multicollinearity appears in a pair plot as strong linear relationships between predictor variables:

    1. **Tightly clustered scatter plots:** If two predictors show a nearly perfect linear trend (points falling along a line), they are highly collinear. Including both in a regression model will inflate standard errors.

    2. **Identical patterns:** If predictor $X_2$ looks like a shifted/scaled version of $X_1$ in every panel (both have similar scatter patterns with the response and with other predictors), they carry redundant information.

    3. **Correlation values:** Overlaying Pearson's $r$ on each panel (or using a combined pair plot with correlation coefficients in the upper triangle) immediately flags pairs with $|r| > 0.8$ or $0.9$.

    To address detected multicollinearity: drop one of the correlated predictors, combine them (e.g., average or PCA), or use regularization (ridge/LASSO).
