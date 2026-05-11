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


## Exercises

**Exercise 1.**
Describe the construction of a Q-Q plot for assessing normality. What do the x-axis and y-axis represent?

??? success "Solution to Exercise 1"
    To construct a normal Q-Q plot:

    1. Sort the data: $x_{(1)} \leq x_{(2)} \leq \dots \leq x_{(n)}$.
    2. Compute theoretical quantiles: $q_i = \mathcal{N}^{-1}((i - 0.5)/n)$, where $\mathcal{N}^{-1}$ is the standard normal quantile function.
    3. Plot the points $(q_i, x_{(i)})$.

    The **x-axis** shows the theoretical normal quantiles (what the data should look like if normal). The **y-axis** shows the actual ordered data values. If the data are normal, the points fall approximately on a straight line with slope $\sigma$ and intercept $\mu$.

---

**Exercise 2.**
On a Q-Q plot, describe the pattern you would see for (a) right-skewed data, (b) heavy-tailed data, and (c) light-tailed data.

??? success "Solution to Exercise 2"
    **(a) Right-skewed:** The Q-Q plot curves upward at the right end (upper quantiles are larger than expected) and may curve slightly downward at the left end. The overall shape is concave-up.

    **(b) Heavy-tailed (leptokurtic):** Both tails deviate from the line -- the left tail curves below the line and the right tail curves above it, forming an S-shape. Extreme values are more extreme than the normal predicts.

    **(c) Light-tailed (platykurtic):** The opposite S-shape -- the left tail curves above the line and the right tail curves below it. Extreme values are less extreme than normal.

---

**Exercise 3.**
A Q-Q plot shows points lying almost exactly on a line in the center but with 3 points far above the line at the upper right. What does this suggest?

??? success "Solution to Exercise 3"
    The central linearity suggests the bulk of the data is approximately normal. The 3 points far above the line at the upper right are **outliers** -- they are much larger than what a normal distribution would predict.

    This pattern is common in data with a few contaminating observations from a different process (e.g., data entry errors, measurement anomalies, or genuinely rare events from a heavy-tailed distribution).

    Action: investigate the outliers for data quality issues. If they are valid observations, consider robust methods (trimmed mean, M-estimators) or acknowledge that the normal model fits the core distribution but not the tails.

---

**Exercise 4.**
Explain the difference between a Q-Q plot and a P-P (probability-probability) plot. When is each preferred?

??? success "Solution to Exercise 4"
    A **Q-Q plot** compares quantiles: it plots ordered data against theoretical quantiles. It is sensitive to departures in the tails (tail quantiles are spread far apart on the axis).

    A **P-P plot** compares cumulative probabilities: it plots $F_n(x_{(i)})$ against $F_0(x_{(i)})$. Points cluster near (0,0) and (1,1), with the most resolution in the center of the distribution.

    **Q-Q plots are preferred** for assessing normality because tail behavior is critical for inference, and Q-Q plots magnify tail departures. **P-P plots are preferred** when the center of the distribution matters more (e.g., calibration assessment) or when comparing distributions with different tail behavior.
