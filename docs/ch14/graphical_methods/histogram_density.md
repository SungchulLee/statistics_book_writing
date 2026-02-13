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
