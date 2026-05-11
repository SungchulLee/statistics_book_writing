# Boxplots and Their Interpretation


## Overview

A **boxplot** summarizes the distribution by showing the median, quartiles, and potential outliers. While not explicitly designed for normality testing, boxplots can hint at skewness and whether the data is symmetric, which are characteristics of normally distributed data.

The boxplot in a normally distributed dataset will be symmetric, and the whiskers (indicating the data range) will be roughly the same length on both sides.

## Box Plot with Normal Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def plot_horizontal_boxplot(data, figsize=(12, 1)):
    """
    Generates a horizontal boxplot for the given data and customizes the appearance
    by removing unnecessary spines.

    Parameters:
    - data (array-like): The input dataset to plot.
    - figsize (tuple): The size of the plot (width, height).

    Returns:
    - None: Displays the horizontal boxplot.
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=data, orient='h', ax=ax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_title('Horizontal Boxplot')
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    sample_data = np.random.normal(loc=0, scale=1, size=1000)
    plot_horizontal_boxplot(sample_data)
```

For normally distributed data, the boxplot is symmetric: the median line is centered in the box, and the whiskers extend approximately equally on both sides.

## Box Plot with Exponential Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def plot_horizontal_boxplot(data, figsize=(12, 1)):
    """
    Generates a horizontal boxplot for the given data.
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=data, orient='h', ax=ax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_title('Horizontal Boxplot')
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    sample_data = np.random.exponential(scale=1, size=1000)
    plot_horizontal_boxplot(sample_data)
```

For exponential data, the boxplot is clearly asymmetric: the right whisker extends much farther than the left, and multiple outliers appear on the right side, indicating strong positive skew.

## Box Plot with Chi-Square Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def plot_horizontal_boxplot(data, figsize=(12, 1)):
    """
    Generates a horizontal boxplot for the given data.
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=data, orient='h', ax=ax)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_title('Horizontal Boxplot')
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    sample_data = np.random.chisquare(df=10, size=1000)
    plot_horizontal_boxplot(sample_data)
```

Chi-square data with 10 degrees of freedom shows moderate right-skew in the boxplot: the median is shifted left within the box, and the right whisker is longer than the left.

## Limitations of Graphical Methods

While graphical methods are helpful for visually assessing normality, they are subjective and rely on interpretation. Small deviations from normality might not be noticeable, and different users may interpret the same plot differently. Moreover, graphical methods are less effective for small sample sizes, where the variability in the data can obscure patterns.


## Exercises

**Exercise 1.**
A boxplot shows the median at 50, Q1 at 35, Q3 at 65, and two outlier points at 5 and 120. Compute the IQR and the whisker boundaries using the 1.5*IQR rule.

??? success "Solution to Exercise 1"
    The IQR is $Q_3 - Q_1 = 65 - 35 = 30$.

    The whisker boundaries are:

    - Lower fence: $Q_1 - 1.5 \times \text{IQR} = 35 - 45 = -10$
    - Upper fence: $Q_3 + 1.5 \times \text{IQR} = 65 + 45 = 110$

    The lower whisker extends to the smallest data point above $-10$ (not to $-10$ itself). The upper whisker extends to the largest data point below 110.

    Points at 5 (above $-10$, so within whiskers) and 120 (above 110, so marked as an outlier) are handled differently. Only 120 is an outlier by the 1.5*IQR rule. The point at 5 would be at or near the lower whisker end.

---

**Exercise 2.**
Compare side-by-side boxplots as a tool for comparing distributions across groups versus using ANOVA. What can boxplots show that ANOVA cannot?

??? success "Solution to Exercise 2"
    Boxplots show the full distributional shape: median, spread (IQR), symmetry, tail behavior, and outliers. ANOVA provides only a single test of whether means differ.

    Boxplots can reveal: (1) differences in medians versus means, (2) unequal variances across groups (different box heights), (3) skewness (asymmetric whiskers), (4) outliers in specific groups, and (5) whether differences are practically meaningful (overlapping boxes suggest small effects).

    ANOVA cannot show any of these features -- it reduces the comparison to a single p-value. The combination of boxplots (visual) and ANOVA (formal test) is ideal.

---

**Exercise 3.**
Explain how boxplots can be used to assess the normality and homoscedasticity assumptions of ANOVA.

??? success "Solution to Exercise 3"
    **Normality:** In each group's boxplot, check for symmetry. The median should be roughly centered in the box (Q1-Q3), and the whiskers should be approximately equal in length. Strongly asymmetric boxes or many outliers suggest non-normality.

    **Homoscedasticity:** Compare the box heights (IQR) across groups. If all boxes have similar heights, the variances are approximately equal. If one group's box is much taller than others, the equal-variance assumption may be violated.

    These are quick visual checks, not formal tests, but they catch the most obvious violations before running ANOVA.

---

**Exercise 4.**
Why can a boxplot be misleading for bimodal distributions? What alternative plot would better reveal bimodality?

??? success "Solution to Exercise 4"
    A boxplot summarizes the distribution through five numbers (min, Q1, median, Q3, max) plus outliers. A bimodal distribution (two distinct clusters) might produce a boxplot that looks like a single symmetric distribution with a wide IQR, completely hiding the two modes.

    For example, a mixture of $N(0,1)$ and $N(5,1)$ would show a boxplot centered around 2.5 with a large IQR, indistinguishable from a single wide distribution.

    **Better alternatives:** violin plots (overlay a KDE on the boxplot, revealing multimodality), histograms, or strip/swarm plots (showing individual data points alongside the box).
