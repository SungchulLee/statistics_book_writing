# Histogram and Density Plots


Graphical methods offer a visual approach to assessing whether a dataset follows a normal distribution. While these methods are not formal statistical tests, they provide insights that are useful in understanding data distribution.

## Overview

A **histogram** is a graphical representation of a dataset's distribution. It divides the data into bins and shows how frequently data points fall into each bin. When the data is normally distributed, the histogram should approximate the familiar bell-shaped curve. A **density plot** is similar but provides a smooth curve representing the distribution.

## Normal Samples with Normal PDF

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_histogram_with_density(data, figsize=(12, 3)):
    """
    The histogram will show the frequency of data points,
    while the **kernel density estimate (KDE)** line will smooth the histogram
    to give a clearer idea of the data distribution.

    Parameters:
    - data (array-like): The input dataset to plot.
    - figsize (tuple): The size of the plot (width, height).

    Returns:
    - None: Displays the plot.
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the histogram with the density curve (KDE)
    _, bins, _ = ax.hist(data, bins=20, density=True, alpha=0.5, label="Data Histogram")

    mu = data.mean()
    sigma = data.std()
    pdf = stats.norm(loc=mu, scale=sigma).pdf(bins)

    ax.plot(bins, pdf, "--r", label="Normal PDF")

    # Customize the appearance: remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set plot title and labels
    ax.set_title('Histogram with Density Plot')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()

    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    sample_data = np.random.normal(loc=0, scale=1, size=1000)
    plot_histogram_with_density(sample_data)
```

When the data is drawn from a normal distribution, the histogram closely matches the overlaid normal PDF curve.

## Exponential Samples with Normal PDF

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_histogram_with_density(data, figsize=(12, 3)):
    """
    Plot histogram with a fitted normal PDF overlay.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _, bins, _ = ax.hist(data, bins=20, density=True, alpha=0.5, label="Data Histogram")

    mu = data.mean()
    sigma = data.std()
    pdf = stats.norm(loc=mu, scale=sigma).pdf(bins)
    ax.plot(bins, pdf, "--r", label="Normal PDF")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title('Histogram with Density Plot')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    sample_data = np.random.exponential(scale=1, size=1000)
    plot_histogram_with_density(sample_data)
```

For exponential data, the histogram is strongly right-skewed and clearly does not match the symmetric normal PDF curve.

## Chi-Square Samples with Normal PDF

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_histogram_with_density(data, figsize=(12, 3)):
    """
    Plot histogram with a fitted normal PDF overlay.
    """
    fig, ax = plt.subplots(figsize=figsize)
    _, bins, _ = ax.hist(data, bins=20, density=True, alpha=0.5, label="Data Histogram")

    mu = data.mean()
    sigma = data.std()
    pdf = stats.norm(loc=mu, scale=sigma).pdf(bins)
    ax.plot(bins, pdf, "--r", label="Normal PDF")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title('Histogram with Density Plot')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    sample_data = np.random.chisquare(df=10, size=1000)
    plot_histogram_with_density(sample_data)
```

Chi-square data with moderate degrees of freedom is moderately right-skewed. The normal PDF provides a rough but imperfect fit, illustrating the importance of formal tests beyond visual inspection.


## Exercises

**Exercise 1.**
Explain how the choice of bin width affects the appearance of a histogram. What happens with too few bins? Too many?

??? success "Solution to Exercise 1"
    **Too few bins (wide):** The histogram is oversmoothed. Important features like bimodality, skewness, or gaps in the data are hidden. The distribution appears simpler than it is.

    **Too many bins (narrow):** The histogram is undersmoothed. Random noise creates a jagged, spiky appearance that obscures the underlying shape. Each bin contains few observations, making the heights unreliable.

    **Guidelines for bin width:** Common rules include Sturges' rule ($k = 1 + \log_2 n$), the Freedman-Diaconis rule ($h = 2 \times \text{IQR} \times n^{-1/3}$), and Scott's rule ($h = 3.49s \times n^{-1/3}$). The Freedman-Diaconis rule is most robust to outliers.

---

**Exercise 2.**
What is the difference between a histogram and a kernel density estimate (KDE)? State one advantage of each.

??? success "Solution to Exercise 2"
    A **histogram** bins the data into intervals and counts observations per bin. It is discontinuous (step function) and depends on bin edges.

    A **KDE** places a smooth kernel (e.g., Gaussian) at each data point and sums them, producing a smooth continuous estimate of the density.

    **Histogram advantage:** Simpler to interpret, directly shows counts/frequencies, and makes gaps in the data visible.

    **KDE advantage:** Smooth and continuous, does not depend on arbitrary bin edges, and better represents the true density shape. It avoids the binning artifacts that can create misleading peaks or valleys.

---

**Exercise 3.**
When overlaying a normal density curve on a histogram for normality assessment, what must you ensure about the histogram's y-axis?

??? success "Solution to Exercise 3"
    The histogram must be **normalized** so that its total area equals 1 (density scale), matching the property of a probability density function. This is typically achieved by setting `density=True` in matplotlib or using relative frequency bins with appropriate bin widths.

    If the histogram shows raw counts (frequency), the normal density curve (which integrates to 1) will be on a completely different scale and the visual comparison is meaningless. After normalization, the height of each bar represents the estimated density, allowing direct visual comparison with the overlaid normal PDF.

---

**Exercise 4.**
Create a Python code snippet that generates a histogram with a KDE overlay for a dataset of 500 observations.

??? success "Solution to Exercise 4"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    rng = np.random.default_rng(42)
    data = rng.normal(5, 2, 500)

    fig, ax = plt.subplots()
    ax.hist(data, bins=30, density=True, alpha=0.5, edgecolor="black", label="Histogram")

    kde = gaussian_kde(data)
    x_grid = np.linspace(data.min() - 1, data.max() + 1, 200)
    ax.plot(x_grid, kde(x_grid), "r-", lw=2, label="KDE")

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    plt.show()
    ```
