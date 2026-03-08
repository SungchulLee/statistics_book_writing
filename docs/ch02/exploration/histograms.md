# Histograms and Density Plots

## Overview

A **histogram** is one of the most fundamental tools in exploratory data analysis. It divides the range of a continuous variable into equal-width intervals (bins) and displays the count or density of observations falling into each bin as rectangular bars. When normalized so that the total area equals one, the histogram approximates a **density plot**—a smooth curve estimating the underlying probability density function (PDF).

$$
\text{Histogram height (density)} = \frac{\text{count in bin}}{\text{total count} \times \text{bin width}}
$$

Histograms reveal distributional features at a glance: center, spread, skewness, modality, gaps, and outliers.

## Basic Histogram with Density Overlay

The following example draws 10,000 samples from a normal distribution, plots a histogram with `density=True`, and overlays the fitted normal PDF.

```python
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

samples = 10_000
x = stats.norm(loc=5, scale=10).rvs(samples)

fig, ax = plt.subplots(figsize=(12, 3))
_, bins, _ = ax.hist(x, bins=100, density=True)

x_mean = x.mean()
x_std = x.std(ddof=1)
pdf = stats.norm(loc=x_mean, scale=x_std).pdf(bins)

ax.plot(bins, pdf, 'r-', linewidth=2)
plt.show()
```

**Key points:**

- `density=True` normalizes the histogram so the total area equals 1, making the y-axis represent probability density rather than raw counts.
- The red curve is the PDF of a normal distribution fitted to the sample mean and standard deviation.
- With 10,000 samples and 100 bins, the histogram closely tracks the theoretical density.

## Histogram of Real-World Data: Income Distribution

Income data is a classic example of a right-skewed distribution where the histogram shape carries important interpretive meaning.

```python
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def plot_loan_income_distribution():
    url = 'https://raw.githubusercontent.com/gedeck/practical-statistics-for-data-scientists/master/data/loans_income.csv'
    df = pd.read_csv(url)

    mean_income = df['x'].mean()
    std_dev_income = df['x'].std()

    fig, ax = plt.subplots(figsize=(15, 4))
    _, bins, _ = ax.hist(df['x'], bins=30, density=True,
                         color='skyblue', label='Income histogram')

    norm_pdf = stats.norm(loc=mean_income, scale=std_dev_income).pdf(bins)
    ax.plot(bins, norm_pdf, "--r", label='Normal distribution')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Loan Income Distribution with Normal Fit')
    ax.set_xlabel('Income')
    ax.set_ylabel('Density')
    ax.legend()
    plt.show()

if __name__ == "__main__":
    plot_loan_income_distribution()
```

The mismatch between the histogram and the normal curve reveals right skewness—a long tail of high-income earners pulls the fitted normal to the right.

## Multi-Panel Histograms: Housing Data

When a dataset contains many numerical features, a grid of histograms provides a rapid overview of all variables simultaneously.

```python
import matplotlib.pyplot as plt
import os
import pandas as pd
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()
df = load_housing_data()

fig, axes = plt.subplots(3, 3, figsize=(12, 9))
df.hist(bins=50, ax=axes)

for ax in axes.reshape((-1,)):
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.show()
```

## Histograms for Categorical-Adjacent Data: Titanic

Even for datasets mixing categorical and numerical variables, histograms help visualize the distribution of each column.

```python
import matplotlib.pyplot as plt
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url, index_col='PassengerId')

fig, axes = plt.subplots(1, 5, figsize=(12, 3))
titles = ("Sex", "Survived", "Age", "Pclass", "Age")

for ax, title in zip(axes, titles):
    ax.hist(df[title], density=True, edgecolor='black', alpha=0.7)
    ax.set_title(title)

plt.tight_layout()
plt.show()
```

## Customized Histogram: Distribution Table to Density Histogram

When data arrives as a frequency table with unequal bin widths, the bar heights must be adjusted so that each bar's **area** (not height) represents the percentage.

$$
\text{height}_i = \frac{\text{percent}_i}{\text{width}_i}
$$

| Income Level (\$) | Percent |
|---|---|
| 0 – 1,000 | 1 |
| 1,000 – 2,000 | 2 |
| 2,000 – 3,000 | 3 |
| 3,000 – 4,000 | 4 |
| 4,000 – 5,000 | 5 |
| 5,000 – 6,000 | 5 |
| 6,000 – 7,000 | 5 |
| 7,000 – 10,000 | 15 |
| 10,000 – 15,000 | 26 |
| 15,000 – 25,000 | 26 |
| 25,000 – 50,000 | 8 |

```python
import matplotlib.pyplot as plt

def compute_bins_widths_heights():
    bins = [0, 1_000, 2_000, 3_000, 4_000, 5_000,
            6_000, 7_000, 10_000, 15_000, 25_000, 50_000]
    widths = [right - left for left, right in zip(bins[:-1], bins[1:])]
    percents = [1, 2, 3, 4, 5, 5, 5, 15, 26, 26, 8]
    heights = [p / w for w, p in zip(widths, percents)]
    return bins, widths, heights

def draw_line(start, end, ax):
    ax.plot([start[0], end[0]], [start[1], end[1]], '-k')

def draw_box(x_left, x_right, height, ax):
    draw_line([x_left, 0], [x_right, 0], ax)
    draw_line([x_right, 0], [x_right, height], ax)
    draw_line([x_right, height], [x_left, height], ax)
    draw_line([x_left, height], [x_left, 0], ax)

def main():
    bins, widths, heights = compute_bins_widths_heights()
    fig, ax = plt.subplots(figsize=(12, 3))
    for x_left, x_right, height in zip(bins[:-1], bins[1:], heights):
        draw_box(x_left, x_right, height, ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position("zero")
    plt.show()

if __name__ == "__main__":
    main()
```

## Choosing the Number of Bins

The number of bins profoundly affects interpretation. Too few bins over-smooth and hide structure; too many create noise. Common guidelines include Sturges' rule ($k = 1 + \log_2 n$), the square-root rule ($k = \lceil\sqrt{n}\rceil$), and the Freedman–Diaconis rule which uses the IQR to set bin width. Matplotlib's `bins='auto'` applies a data-adaptive strategy.

## Summary

Histograms and density plots are the first line of exploration for any continuous variable. They expose the shape of the distribution—symmetric or skewed, unimodal or multimodal, heavy-tailed or light-tailed—guiding every subsequent modeling and inference decision.
