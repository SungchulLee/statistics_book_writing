# Q-Q Plots (Quantile-Quantile Plots)

## Overview

A **Q-Q plot** compares the quantiles of the dataset with the quantiles of a theoretical normal distribution. If the data is normally distributed, the points on the Q-Q plot should fall along a straight diagonal line. Deviations from this line indicate departures from normality.

## Q-Q Plot with Normal Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_qq_with_custom_spines(data, dist="norm", sparams=(), figsize=(12, 3)):
    """
    Generates a Q-Q plot to assess if the data follows the specified distribution.
    Adjusts the spines for a cleaner visual appearance.

    Parameters:
    - data (array-like): The input dataset to plot.
    - dist (str): The theoretical distribution to compare against (default: "norm").
    - sparams (tuple): Shape parameters for the specified distribution.
    - figsize (tuple): The size of the plot (width, height).

    Returns:
    - None: Displays the Q-Q plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    stats.probplot(data, dist=dist, sparams=sparams, plot=ax)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title('Q-Q Plot')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Ordered Values')
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    sample_data = np.random.normal(loc=0, scale=1, size=1000)
    plot_qq_with_custom_spines(sample_data, dist="norm")
```

When the data is normally distributed, the points lie closely along the diagonal reference line.

## Q-Q Plot with Exponential Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_qq_with_custom_spines(data, dist="norm", sparams=(), figsize=(12, 3)):
    """
    Generates a Q-Q plot to assess if the data follows the specified distribution.
    """
    fig, ax = plt.subplots(figsize=figsize)
    stats.probplot(data, dist=dist, sparams=sparams, plot=ax)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title('Q-Q Plot')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Ordered Values')
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    sample_data = np.random.exponential(scale=1, size=1000)
    plot_qq_with_custom_spines(sample_data, dist="expon")
```

When comparing exponential data against its own theoretical distribution, the points align well. However, comparing exponential data against a normal distribution would show strong curvature, revealing the departure from normality.

## Q-Q Plot with Chi-Square Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_qq_with_custom_spines(data, dist="norm", sparams=(), figsize=(12, 3)):
    """
    Generates a Q-Q plot to assess if the data follows the specified distribution.
    """
    fig, ax = plt.subplots(figsize=figsize)
    stats.probplot(data, dist=dist, sparams=sparams, plot=ax)

    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title('Q-Q Plot')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Ordered Values')
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    sample_data = np.random.chisquare(df=10, size=1000)
    plot_qq_with_custom_spines(sample_data, dist="chi2", sparams=(10,))
```

When comparing chi-square data against its own theoretical distribution (with matching degrees of freedom), the Q-Q plot shows a good fit. Comparing against a normal Q-Q plot would reveal right-skew through upward curvature in the tails.
