# Basic Visualization with Matplotlib

Matplotlib is the foundational plotting library in Python and produces the publication-quality figures used throughout this book. Its object-oriented API gives precise control over every figure element — axes, ticks, gridlines, annotations — which matters because every plot in a statistics text serves a specific argument and should be tailored, not auto-generated.

## Definition

Every Matplotlib plot lives inside a **`Figure`** containing one or more **`Axes`** objects (each is an individual plot panel). The recommended pattern for creating them is

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4))
# ... draw on ax ...
plt.show()
```

For grid layouts:

```python
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
ax = axes[0, 1]    # row 0, column 1
```

`Figure` holds the canvas, title, and overall layout; each `Axes` is its own coordinate system with labels, ticks, and artists.

## Explanation

### The two APIs

Matplotlib has two interfaces:

1. **Pyplot (stateful)** — `plt.plot(x, y)`, `plt.title(...)`. Operates on a global "current axes" maintained behind the scenes. Convenient for one-off plots; ambiguous for multi-panel figures.
2. **Object-oriented** — `ax.plot(x, y)`, `ax.set_title(...)`. Explicit about which `Axes` is being modified. Preferred for any figure with more than one panel.

The book uses the object-oriented form exclusively. The pyplot form is reserved for `plt.subplots`, `plt.show`, and `plt.savefig`.

### Core plot types for statistics

| Plot | Method | Used for |
|---|---|---|
| Histogram | `ax.hist(x, bins=...)` | Distribution of one variable; overlay theoretical PDFs with `density=True` |
| Scatter | `ax.scatter(x, y)` | Bivariate relationship; regression diagnostics |
| Line | `ax.plot(x, y)` | Time series, CDFs, smooth function curves |
| Boxplot | `ax.boxplot(x_list)` | Five-number summary and outliers across groups |
| Bar | `ax.bar(categories, heights)` | Categorical comparisons |
| Q-Q plot | `scipy.stats.probplot(x, plot=ax)` | Normality diagnostics |

### Customization essentials

```python
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Title")
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)
ax.grid(True, alpha=0.3)
ax.legend(loc="best", frameon=False)
fig.tight_layout()
fig.savefig("figure.png", dpi=150, bbox_inches="tight")
```

`tight_layout` prevents axis labels from being clipped or overlapping in multi-panel figures. `dpi=150` is sufficient for screen and most print purposes; `dpi=300` is overkill except for camera-ready publication.

### pandas integration

DataFrames carry their own plotting methods that wrap Matplotlib:

```python
df["x"].plot.hist(bins=30)
df.plot.scatter(x="x", y="y", ax=ax)
df.boxplot(column="value", by="group", ax=ax)
```

These are convenient for quick exploration. For final figures, falling back to direct Matplotlib calls gives finer control.

## Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

rng = np.random.default_rng(42)
data = rng.normal(loc=50, scale=10, size=500)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: histogram with theoretical density overlay
axes[0].hist(data, bins=30, density=True, alpha=0.5,
             edgecolor="black", label="Sample")
x = np.linspace(data.min(), data.max(), 200)
axes[0].plot(x, norm.pdf(x, loc=50, scale=10), "r-",
             lw=2, label="N(50, 10) PDF")
axes[0].set_xlabel("value"); axes[0].set_ylabel("density")
axes[0].set_title("Histogram with density overlay")
axes[0].legend()

# Right: Q-Q plot
from scipy.stats import probplot
probplot(data, dist="norm", plot=axes[1])
axes[1].set_title("Q-Q plot vs. Normal")

fig.tight_layout()
plt.show()
```

The histogram tests the fit visually; the Q-Q plot tests it analytically by showing departures from linearity. A statistical figure is rarely complete without one of these.

## Exercises

**Exercise 1.**
Write code using the object-oriented API to plot a histogram of 500 standard normal samples with 30 bins. Include axis labels, a title, and a vertical line at the sample mean.

??? success "Solution to Exercise 1"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    data = rng.standard_normal(500)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(data.mean(), color="red", lw=2, label=f"mean = {data.mean():.3f}")
    ax.set_xlabel("z")
    ax.set_ylabel("frequency")
    ax.set_title("500 standard normal samples")
    ax.legend()
    plt.show()
    ```

    `axvline` draws a vertical reference line at a fixed $x$-coordinate, useful for showing a mean, median, threshold, or critical value.

---

**Exercise 2.**
Create a two-panel figure: left, a scatter of 100 random $(x, y)$ pairs; right, the curve $y = \sin(x)$ on $[0, 2\pi]$. Title each panel and share a single $y$-axis label "value".

??? success "Solution to Exercise 2"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    x_s, y_s = rng.uniform(0, 10, 100), rng.uniform(-1, 1, 100)
    x_l = np.linspace(0, 2 * np.pi, 200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax1.scatter(x_s, y_s, alpha=0.6)
    ax1.set_title("Random scatter")
    ax1.set_xlabel("x")
    ax2.plot(x_l, np.sin(x_l), color="blue")
    ax2.set_title("sin(x)")
    ax2.set_xlabel("x")
    fig.supylabel("value")
    fig.tight_layout()
    plt.show()
    ```

    `sharey=True` ties the two y-axes; `fig.supylabel` adds a single y-label spanning the figure.

---

**Exercise 3.**
Why is `ax.plot()` preferred over `plt.plot()` for multi-panel figures? Give one concrete example where the pyplot form silently plots to the wrong subplot.

??? success "Solution to Exercise 3"
    `plt.plot()` targets the "current" Axes, which is global state. In a notebook cell that creates two figures, or one figure with several subplots, the "current" Axes is whichever was most recently created or activated — usually the **last** subplot, not the one the author meant.

    ```python
    fig, axes = plt.subplots(1, 2)
    axes[0].set_title("Panel A")        # explicit — correct
    plt.plot([1, 2, 3])                 # silently lands on axes[1] because it was created last
    ```

    The object-oriented form `axes[0].plot([1, 2, 3])` makes the target explicit and avoids the ambiguity entirely.

---

**Exercise 4.**
Plot a normalized histogram of 1000 samples from $N(5, 4)$ (mean 5, variance 4) and overlay the true density. Verify visually that the empirical and theoretical curves agree.

??? success "Solution to Exercise 4"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    rng = np.random.default_rng(42)
    data = rng.normal(loc=5, scale=2, size=1000)   # scale = sqrt(variance)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=30, density=True, alpha=0.5, edgecolor="black", label="Histogram")
    x = np.linspace(data.min(), data.max(), 200)
    ax.plot(x, norm.pdf(x, loc=5, scale=2), "r-", lw=2, label="N(5, 4) PDF")
    ax.set_xlabel("x"); ax.set_ylabel("density")
    ax.set_title("Histogram with true density overlay")
    ax.legend()
    plt.show()
    ```

    `density=True` rescales the histogram bars so the total area equals 1, making them directly comparable to a PDF. The width of bars is determined by `bins`; too few bins hide structure, too many invent it.

---

**Exercise 5.**
Construct a residual plot: fit a least-squares line to 50 noisy points from $y = 1 + 2x + \varepsilon$, then plot the residuals versus the fitted values with a horizontal reference at zero. What pattern would indicate a violated linearity assumption?

??? success "Solution to Exercise 5"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 50)
    y = 1 + 2 * x + rng.standard_normal(50)

    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    y_hat = X @ beta
    resid = y - y_hat

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_hat, resid, alpha=0.7)
    ax.axhline(0, color="red", lw=1)
    ax.set_xlabel("fitted value")
    ax.set_ylabel("residual")
    ax.set_title("Residuals vs. fitted")
    plt.show()
    ```

    A well-specified linear model produces residuals scattered randomly around zero with no visible trend. A **curved** (U- or arch-shaped) pattern in residuals vs. fitted signals a missing nonlinear term; a **funnel** signals non-constant variance (heteroscedasticity). Chapter 13 develops these diagnostics formally.

---

**Exercise 6.**
Save the figure from Exercise 5 to disk as a PNG at 200 DPI with no padding. What does the `bbox_inches="tight"` argument do, and when does it matter?

??? success "Solution to Exercise 6"
    ```python
    fig.savefig("residuals.png", dpi=200, bbox_inches="tight")
    ```

    `dpi=200` controls the rasterization resolution. `bbox_inches="tight"` recomputes the bounding box of the figure to exclude empty whitespace at the margins. It matters most when the figure is embedded in another document (LaTeX, Word, slides): without it, savefig writes the full canvas including any unused space, and the embedded image has unsightly white borders. The pair `dpi=200, bbox_inches="tight"` is the recommended default for figures included in this book.
