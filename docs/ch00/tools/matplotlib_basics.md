# Basic Visualization with Matplotlib

## Overview

**Matplotlib** is Python's foundational plotting library. It can produce publication-quality static figures in a wide variety of formats and integrates tightly with NumPy and pandas. This section covers the core plotting patterns used throughout the book for exploratory data analysis and statistical visualization.

```python
import matplotlib.pyplot as plt
import numpy as np
```

## The Figure–Axes Model

Every Matplotlib plot lives inside a **Figure**, which contains one or more **Axes** (individual plots). The recommended way to create figures is with `plt.subplots()`.

```python
fig, ax = plt.subplots()            # single plot
fig, axes = plt.subplots(1, 2)      # 1 row, 2 columns
fig, axes = plt.subplots(2, 3, figsize=(12, 6))  # 2×3 grid
```

!!! tip "Axes vs Axis"
    In Matplotlib terminology, an **Axes** object is an entire plot (with its own title, labels, and data). An **axis** (lowercase) refers to the x-axis or y-axis within that plot.

## Line Plots

Line plots are the most basic visualization type and are commonly used for time series and function curves.

```python
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

fig, ax = plt.subplots()
ax.plot(x, y, marker="o", linestyle="-", color="b", label="primes")

ax.set_title("Simple Line Plot")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.legend()
plt.show()
```

### Plotting Mathematical Functions

```python
x = np.linspace(-2 * np.pi, 2 * np.pi, 201)

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(x, np.sin(x), label="sin(x)")
ax.plot(x, np.cos(x), label="cos(x)", linestyle="--")

# Custom x-ticks with π labels
ax.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi])
ax.set_xticklabels([r"$-2\pi$", r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])

# Move spines to origin
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_position("zero")
ax.spines["left"].set_position("zero")

ax.legend()
plt.show()
```

## Scatter Plots

Scatter plots visualize the relationship between two continuous variables—central to correlation analysis and regression diagnostics.

```python
rng = np.random.default_rng(42)
x = rng.normal(0, 1, 100)
y = 2 * x + rng.normal(0, 0.5, 100)

fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.6, edgecolors="k", linewidths=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Scatter Plot")
plt.show()
```

## Histograms

Histograms display the distribution of a single variable—the visual counterpart to density estimation.

```python
data = rng.normal(loc=50, scale=10, size=500)

fig, ax = plt.subplots()
ax.hist(data, bins=25, edgecolor="black", alpha=0.7)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Histogram")
plt.show()
```

### Normalized Histogram with Density Overlay

```python
from scipy.stats import norm

fig, ax = plt.subplots()
ax.hist(data, bins=30, density=True, alpha=0.6, edgecolor="black", label="data")

# Overlay theoretical density
x_grid = np.linspace(data.min(), data.max(), 200)
ax.plot(x_grid, norm.pdf(x_grid, loc=50, scale=10), "r-", lw=2, label="N(50, 10²)")

ax.set_xlabel("Value")
ax.set_ylabel("Density")
ax.legend()
plt.show()
```

## Bar Charts

Bar charts compare categorical quantities—useful for frequency tables and group comparisons.

```python
categories = ["A", "B", "C", "D"]
values = [23, 45, 12, 37]

fig, ax = plt.subplots()
ax.bar(categories, values, color="steelblue", edgecolor="black")
ax.set_xlabel("Category")
ax.set_ylabel("Count")
ax.set_title("Bar Chart")
plt.show()
```

## Box Plots

Box plots summarize the five-number summary (min, Q1, median, Q3, max) and flag outliers.

```python
groups = [rng.normal(0, 1, 100),
          rng.normal(1, 1.5, 100),
          rng.normal(-0.5, 0.8, 100)]

fig, ax = plt.subplots()
ax.boxplot(groups, labels=["Group A", "Group B", "Group C"])
ax.set_ylabel("Value")
ax.set_title("Box Plot Comparison")
plt.show()
```

## Subplots

Subplots allow multiple plots to share a single figure, enabling side-by-side comparisons.

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Plot 1: Histogram
axes[0].hist(rng.normal(0, 1, 500), bins=25, edgecolor="black")
axes[0].set_title("Histogram")

# Plot 2: Scatter
x = rng.normal(0, 1, 100)
axes[1].scatter(x, x + rng.normal(0, 0.3, 100), alpha=0.6)
axes[1].set_title("Scatter")

# Plot 3: Line
t = np.linspace(0, 4 * np.pi, 200)
axes[2].plot(t, np.sin(t))
axes[2].set_title("Sine Wave")

fig.suptitle("Three Subplots", fontsize=14)
fig.tight_layout()
plt.show()
```

## Customization Reference

### Colors, Markers, and Line Styles

```python
# Named colors: "steelblue", "coral", "seagreen", "slategray"
# Hex colors:   "#1f77b4"
# Markers:      "o", "s", "^", "D", "x", "+"
# Line styles:  "-", "--", "-.", ":"
```

### Labels, Titles, and Legends

```python
ax.set_title("Title", fontsize=14)
ax.set_xlabel("X Label", fontsize=12)
ax.set_ylabel("Y Label", fontsize=12)
ax.legend(loc="upper right", fontsize=10)
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)
```

### Grid and Ticks

```python
ax.grid(True, alpha=0.3)
ax.tick_params(axis="both", labelsize=10)
```

## Plotting Directly from pandas

pandas DataFrames and Series have a built-in `.plot()` method that wraps Matplotlib, making quick exploratory plots convenient.

```python
import pandas as pd

df = pd.DataFrame({
    "A": rng.normal(0, 1, 200),
    "B": rng.normal(1, 2, 200)
})

# Histogram of all columns
df.plot.hist(bins=30, alpha=0.5, edgecolor="black")

# Scatter plot
df.plot.scatter(x="A", y="B", alpha=0.5)

# Box plot
df.plot.box()

plt.show()
```

## Saving Figures

```python
fig.savefig("figure.png", dpi=150, bbox_inches="tight")
fig.savefig("figure.pdf", bbox_inches="tight")
fig.savefig("figure.svg", bbox_inches="tight")
```

The `bbox_inches="tight"` argument trims excess whitespace around the figure.

## Statistical Plot Recipes

The following patterns recur throughout the book.

### Empirical CDF

```python
def plot_ecdf(data, ax, **kwargs):
    """Plot the empirical CDF of a 1-D array."""
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.step(sorted_data, ecdf, where="post", **kwargs)
    ax.set_ylabel("ECDF")

fig, ax = plt.subplots()
sample = rng.normal(0, 1, 300)
plot_ecdf(sample, ax, label="sample")
ax.legend()
plt.show()
```

### Q-Q Plot (Manual)

```python
from scipy.stats import norm

sample = np.sort(rng.normal(0, 1, 200))
theoretical = norm.ppf(np.linspace(0.005, 0.995, len(sample)))

fig, ax = plt.subplots()
ax.scatter(theoretical, sample, s=10, alpha=0.6)
lims = [min(theoretical.min(), sample.min()), max(theoretical.max(), sample.max())]
ax.plot(lims, lims, "r--", lw=1)
ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Sample Quantiles")
ax.set_title("Q-Q Plot")
plt.show()
```

### Confidence Interval Visualization

```python
means = [2.3, 3.1, 4.5]
ci_lower = [1.8, 2.5, 3.9]
ci_upper = [2.8, 3.7, 5.1]
labels = ["A", "B", "C"]

fig, ax = plt.subplots()
y_pos = range(len(means))
ax.errorbar(means, y_pos,
            xerr=[[m - lo for m, lo in zip(means, ci_lower)],
                  [hi - m for m, hi in zip(means, ci_upper)]],
            fmt="o", capsize=4)
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel("Estimate")
ax.set_title("Confidence Intervals")
plt.show()
```

## Summary

| Plot Type | When to Use | Key Function |
|---|---|---|
| Line plot | Trends, time series, function curves | `ax.plot()` |
| Scatter plot | Bivariate relationships | `ax.scatter()` |
| Histogram | Distribution of a single variable | `ax.hist()` |
| Bar chart | Categorical comparisons | `ax.bar()` |
| Box plot | Five-number summary and outliers | `ax.boxplot()` |
| Subplots | Side-by-side comparisons | `plt.subplots(nrows, ncols)` |
| ECDF | Non-parametric distribution view | Custom `step` function |
| Q-Q plot | Normality assessment | `scatter` + reference line |
