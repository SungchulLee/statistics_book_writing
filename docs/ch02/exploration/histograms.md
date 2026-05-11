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

## Exercises

**Exercise 1.**
A researcher collects 20 exam scores: $55, 62, 67, 70, 71, 73, 74, 75, 76, 78, 80, 81, 83, 85, 87, 88, 90, 92, 95, 98$.

**(a)** Using Sturges' rule, how many bins should the histogram have?
**(b)** Apply 4 equal-width bins spanning the range. Specify bin edges and counts.
**(c)** Explain why too few bins hide structure and too many bins fabricate it.

??? success "Solution to Exercise 1"
    (a) Sturges' rule: $k = \lceil \log_2 n \rceil + 1$. With $n = 20$, $k = \lceil 4.32 \rceil + 1 = 6$.

    (b) Range $= 43$; bin width $= 43/4 = 10.75$:

    - $[55, 65.75)$: 2 values (55, 62)
    - $[65.75, 76.5)$: 7 values (67, 70, 71, 73, 74, 75, 76)
    - $[76.5, 87.25)$: 6 values (78, 80, 81, 83, 85, 87)
    - $[87.25, 98]$: 5 values (88, 90, 92, 95, 98)

    (c) Too few bins merge distinct features — a bimodal distribution can look unimodal if both modes land in the same bin. Too many bins create sampling-noise peaks and valleys that do not reflect the true density. The "right" number balances bias (from over-smoothing) and variance (from noisy bins), which is the same trade-off behind nonparametric density estimation.

---

**Exercise 2.**
Compare **Sturges' rule** ($k = 1 + \log_2 n$), the **square-root rule** ($k = \lceil \sqrt n \rceil$), and the **Freedman–Diaconis rule** (bin width $h = 2 \cdot \mathrm{IQR}/n^{1/3}$). When does each fail?

??? success "Solution to Exercise 2"
    **Sturges:** assumes approximately normal data and undercounts bins for large $n$ — only 10 bins at $n = 1024$, which over-smooths for big samples. Fails for skewed or heavy-tailed data.

    **Square-root:** simple and reasonable for moderate $n$ but ignores the data's spread. Tends to over-bin sparse data and under-bin dense data.

    **Freedman–Diaconis:** uses the IQR (robust to outliers) and scales as $n^{-1/3}$ (the asymptotically optimal rate for histogram MISE). Generally the best default. Fails when the IQR is zero or very small (e.g., heavily discrete data) — in that case fall back to Scott's rule using SD.

    Modern practice: use Matplotlib's `bins='auto'` which combines Freedman–Diaconis with Sturges, picking the larger of the two.

---

**Exercise 3.**
Show that with `density=True`, the total area under the histogram equals 1. Why is this normalization required to compare a histogram with a theoretical PDF?

??? success "Solution to Exercise 3"
    Let bins have widths $w_1, \ldots, w_k$ and counts $c_1, \ldots, c_k$ with $\sum c_i = n$. With density normalization, the height of bin $i$ is $h_i = c_i / (n w_i)$. The total area:

    $$
    \sum_i w_i \cdot h_i = \sum_i w_i \cdot \frac{c_i}{n w_i} = \frac{1}{n}\sum_i c_i = 1
    $$

    Any PDF $f$ satisfies $\int f(x)\,dx = 1$. Without normalization, the histogram heights would be in counts (totaling $n$, not 1), and direct overlay with $f$ would mismatch by a factor of $n \cdot w$. Density normalization places both on the same scale (probability per unit $x$), enabling direct visual comparison and goodness-of-fit assessment.

---

**Exercise 4.**
A histogram of 1000 i.i.d. samples from $N(0, 1)$ uses 30 equal-width bins on $[-4, 4]$. (a) Estimate the expected count in the bin containing zero. (b) Estimate the standard deviation of that count.

??? success "Solution to Exercise 4"
    Bin width $w = 8/30 \approx 0.267$. The bin containing zero is $[-w/2, w/2] = [-0.133, 0.133]$.

    (a) Probability a single observation falls in this bin: $P(-0.133 < Z < 0.133) \approx 2 \cdot 0.133 \cdot \phi(0) \approx 2 \cdot 0.133 \cdot 0.399 \approx 0.106$. Expected count $\approx 1000 \times 0.106 = 106$.

    (b) The count is binomial: $\mathrm{Var} = np(1-p) = 1000 \cdot 0.106 \cdot 0.894 \approx 95$, so SD $\approx 9.7$.

    The relative noise (SD/mean) is $\approx 9\%$ in this central bin — small enough that the histogram tracks the density faithfully. In a tail bin where $p \approx 0.001$, expected count is only 1 and SD is also $\approx 1$ — the relative noise is 100%, which is why histogram tails look ragged and density estimates need different treatment in the tails.

---

**Exercise 5.**
The **kernel density estimate (KDE)** smooths the histogram by replacing each observation with a kernel function $K_h(x - x_i)$. Write the KDE formula. Why is KDE generally preferred over histograms for visualization?

??? success "Solution to Exercise 5"
    The KDE is

    $$
    \hat f(x) = \frac{1}{n h} \sum_{i=1}^n K\!\left(\frac{x - x_i}{h}\right)
    $$

    where $K$ is a kernel function (usually Gaussian: $K(u) = \frac{1}{\sqrt{2\pi}}e^{-u^2/2}$) integrating to 1, and $h > 0$ is the bandwidth.

    **Advantages over histograms:**

    - **Smoothness**: KDE produces a continuous curve, easier to read and compare across plots.
    - **No bin-edge artifacts**: the histogram's appearance changes discontinuously as bin edges shift; KDE is invariant to such shifts.
    - **Better convergence rate** for smooth densities: optimal $O(n^{-4/5})$ MISE for Gaussian kernel vs. $O(n^{-2/3})$ for histograms.
    - **Adaptive bandwidth methods** (Silverman's rule, plug-in selectors) automate the smoothness choice.

    **Disadvantages:** KDE can over-smooth (hide modes) or under-smooth (create spurious peaks). It can also produce non-zero density in implausible regions (e.g., negative values for income data). Boundary-corrected KDEs address the latter.

---

**Exercise 6.**
Sketch what histograms of the following look like and identify which distributional feature each reveals: (a) heights of adult humans; (b) annual household income; (c) age at death in a developed country; (d) the digit-sum of phone numbers.

??? success "Solution to Exercise 6"
    (a) **Heights of adults**: roughly symmetric, bell-shaped, possibly slightly bimodal (males and females have distinct modes). Reveals approximate normality conditional on sex, mixture structure unconditionally.

    (b) **Annual household income**: strongly right-skewed with a long upper tail. Mean $\gg$ median. Often fitted by a lognormal or Pareto distribution. Reveals economic inequality through the heavy upper tail.

    (c) **Age at death** (developed country): bimodal — a small peak near 0 (infant mortality) and a large peak in the 70s–80s. Reveals competing causes of mortality (early-life vs. age-related). With improving health care, the infant peak has shrunk; the old-age peak has shifted right.

    (d) **Digit-sum of phone numbers**: approximately bell-shaped (CLT in action). The digit-sum is a sum of nearly independent uniform digits, so its distribution approaches normal. Reveals the central limit theorem in everyday data.

    Together these illustrate that histogram shape encodes *qualitative* information that summary statistics alone miss. Always plot first; summarize second.
