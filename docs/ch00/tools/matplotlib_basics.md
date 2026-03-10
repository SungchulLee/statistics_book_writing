# Basic Visualization with Matplotlib

Matplotlib is Python's foundational plotting library, producing publication-quality static figures. This page covers the core plotting patterns used throughout the book.

## Definition

Every Matplotlib plot lives inside a **Figure** containing one or more **Axes** objects (individual plots). The recommended creation pattern is `fig, ax = plt.subplots()` for a single plot or `fig, axes = plt.subplots(nrows, ncols)` for grids.

## Explanation

**Core plot types** for statistics:

- **Histogram** (`ax.hist`): Distribution of a single variable. Use `density=True` to overlay a theoretical PDF.
- **Scatter** (`ax.scatter`): Bivariate relationships, regression diagnostics.
- **Line** (`ax.plot`): Time series, function curves, CDFs.
- **Box** (`ax.boxplot`): Five-number summary and outliers across groups.
- **Bar** (`ax.bar`): Categorical comparisons.

**Customization essentials**: `ax.set_xlabel/ylabel/title` for labels, `ax.legend()` for legends, `ax.grid(True, alpha=0.3)` for gridlines, `fig.tight_layout()` to prevent overlap. Save with `fig.savefig("name.png", dpi=150, bbox_inches="tight")`.

**pandas integration**: DataFrames have `.plot.hist()`, `.plot.scatter()`, `.plot.box()` methods wrapping Matplotlib for quick exploration.

## Examples

```python
import numpy as np
from scipy.stats import norm

# Statistical recipe: histogram with density overlay
rng = np.random.default_rng(42)
data = rng.normal(loc=50, scale=10, size=500)

# Verify histogram statistics (no plotting)
print(f"Sample mean: {data.mean():.2f}")
print(f"Sample std:  {data.std(ddof=1):.2f}")
print(f"Theoretical: mean=50, std=10")

# Q-Q plot computation (no plotting)
sorted_data = np.sort(data)
n = len(sorted_data)
theoretical_q = norm.ppf((np.arange(1, n + 1) - 0.5) / n, loc=50, scale=10)
correlation = np.corrcoef(theoretical_q, sorted_data)[0, 1]
print(f"Q-Q correlation: {correlation:.6f} (1.0 = perfect normality)")
```
