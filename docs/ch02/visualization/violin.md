# Violin Plots

## Overview

A **violin plot** combines a box plot with a kernel density estimate (KDE) on each side, showing the full distribution shape alongside summary statistics. Where a box plot reduces the distribution to five numbers plus outliers, a violin plot reveals multimodality, skewness, and density variations that box plots hide.

## Violin Plot vs. Box Plot

The key advantage of violin plots over box plots is the ability to show the **probability density** of the data at different values. This makes them particularly useful for:

- Detecting bimodal or multimodal distributions that a box plot would miss.
- Comparing distribution shapes across groups when the differences are subtle.
- Communicating the full distributional story to an audience.

## Basic Violin Plot

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

# Create a bimodal distribution that a box plot would obscure
data_1 = np.concatenate([np.random.normal(0, 1, 500),
                          np.random.normal(5, 1, 500)])
data_2 = np.random.normal(2.5, 2, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Violin plot reveals bimodality
ax1.violinplot([data_1, data_2], showmeans=True, showmedians=True)
ax1.set_title("Violin Plot")
ax1.set_xticks([1, 2])
ax1.set_xticklabels(["Bimodal", "Unimodal"])

# Box plot hides the bimodality
ax2.boxplot([data_1, data_2], labels=["Bimodal", "Unimodal"])
ax2.set_title("Box Plot")

plt.tight_layout()
plt.show()
```

## Violin Plot with Seaborn

Seaborn provides a more polished violin plot with built-in grouping:

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

fig, ax = plt.subplots(figsize=(10, 4))
sns.violinplot(data=df, x="Pclass", y="Age", hue="Sex",
               split=True, ax=ax)
ax.set_title("Age Distribution by Class and Sex (Titanic)")
plt.show()
```

The `split=True` option places the two hue categories on opposite sides of each violin, enabling direct visual comparison within each class.

## When to Use Violin Plots

Violin plots are most valuable when comparing the shapes of distributions across groups, especially when the distributions may be non-normal or multimodal. For simple comparisons where only the median and IQR matter, box plots remain more concise and easier to read.

## Summary

Violin plots extend box plots by adding density information, making them ideal for revealing distributional details such as multimodality and asymmetry. They are particularly effective in group comparisons where distribution shape—not just summary statistics—drives the analysis.

## Exercises

**Exercise 1.**
Two plant-growth experiments give: **Treatment 1**: $\{5, 6, 6, 7, 7, 7, 8, 8, 9\}$; **Treatment 2**: $\{3, 5, 7, 7, 7, 7, 7, 9, 11\}$. (a) Five-number summaries. (b) Would boxplots look similar? (c) How do violin plots differ?

??? success "Solution to Exercise 1"
    (a) Both datasets:

    | | T1 | T2 |
    |---|---|---|
    | Min | 5 | 3 |
    | $Q_1$ | 6 | 6 |
    | Median | 7 | 7 |
    | $Q_3$ | 8 | 8 |
    | Max | 9 | 11 |

    (b) Boxes are identical; only the whisker lengths differ. The two boxplots look very similar.

    (c) Violin plots reveal Treatment 2's sharp spike at 7 (five of nine values equal 7), while Treatment 1 has roughly uniform density across $[5, 9]$. The two distributions have nearly identical centers and spread but very different shapes. The boxplot is blind to this; the violin plot makes it immediately visible.

---

**Exercise 2.**
The violin plot's density is computed by **kernel density estimation**. Write the KDE formula and discuss how the bandwidth $h$ affects the violin plot's appearance.

??? success "Solution to Exercise 2"
    KDE formula:

    $$
    \hat f(x) = \frac{1}{n h}\sum_{i=1}^n K\!\left(\frac{x - x_i}{h}\right)
    $$

    where $K$ is a kernel (typically Gaussian) and $h > 0$ is the bandwidth.

    **Bandwidth effect on the violin:**

    - **Small $h$**: density estimate becomes spiky. Each data point creates a narrow bump. The violin shows individual observations rather than the underlying density. Can over-fit sampling noise.
    - **Large $h$**: density oversmoothed. Modes blur together; bimodal distributions look unimodal. Distortion that hides exactly the features the violin is supposed to show.
    - **Optimal $h$ (e.g., Silverman, Scott, plug-in selectors):** balances bias and variance.

    Most plotting libraries (matplotlib, seaborn) apply Scott's rule by default. Tweak `bw_method` if violins look too jagged (decrease) or over-smoothed (increase).

---

**Exercise 3.**
**Half-violin (split-violin) plots** show two groups on opposite sides of a single vertical axis. When is this presentation preferred to side-by-side full violins?

??? success "Solution to Exercise 3"
    Split-violin plots are preferred when:

    - **Direct paired comparison** is the primary message — e.g., comparing male and female age distributions within each passenger class.
    - The two distributions are expected to differ subtly. Placing them on opposite sides of a shared axis makes small differences in shape, location, or spread visually obvious.
    - **Space is limited**: a single split-violin takes half the horizontal space of two side-by-side full violins.

    Avoid split-violins when:

    - There are more than two groups.
    - The two groups have very different sample sizes (the densities are normalized, hiding the imbalance).
    - The shapes are very different — the "opposite halves" might be misleading because the eye reads them as symmetric.

---

**Exercise 4.**
A violin plot of medical-trial outcome data is **truncated** at a hard physical lower bound (e.g., zero for non-negative quantities). What artifact does the KDE introduce, and how can it be corrected?

??? success "Solution to Exercise 4"
    A standard KDE places kernel mass *symmetrically* around each observation. Near a hard boundary, this places probability mass *below the boundary* — for data bounded at zero, the KDE assigns nonzero density to negative values that are physically impossible.

    **Visual artifact:** the violin appears to extend below zero, suggesting the data could be negative. The density just above zero is also under-estimated because the boundary blocks the kernel mass that would normally come from the symmetric extension.

    **Corrections:**

    - **Reflection method**: reflect the data across the boundary, fit KDE on the doubled dataset, then truncate at the boundary and double the density above it.
    - **Beta KDE** for $[0, 1]$ bounded data, or **gamma / log-normal KDE** for $[0, \infty)$ data — kernels with appropriate support.
    - **Transformation**: take $\log(x + 1)$ for non-negative data, fit KDE on the transformed scale, plot on the original scale via change-of-variables.

    Most plotting libraries let you clip the violin at a stated boundary, but the underlying density estimate may still be biased near the boundary. Always interpret near-boundary parts of a violin with caution.

---

**Exercise 5.**
Why is the *width* of a violin sometimes *normalized* across groups (each violin has the same maximum width) and sometimes *not normalized* (width reflects sample size)? When is each appropriate?

??? success "Solution to Exercise 5"
    **Normalized (each violin max width = 1):** emphasizes shape comparison. Each group's distribution shape is shown at full visual size regardless of how many observations the group contains. Suitable when sample sizes differ but you want to compare shapes directly.

    **Not normalized (width $\propto n$):** preserves the relative importance of each group. A group with 1000 observations appears much wider than one with 10 observations, signaling that the small group's density estimate is less reliable.

    **When each is appropriate:**

    - Use **normalized** when sample sizes are comparable, or when the message is purely about shape (e.g., comparing income distributions across countries with different population sizes — the country with 30M people shouldn't visually dominate one with 3M).
    - Use **un-normalized** when sample size differences are themselves part of the story (e.g., comparing the 1000-respondent treatment arm with the 50-respondent control arm — the difference in certainty matters).

    Many libraries default to **scale="area"** (un-normalized) but offer **scale="width"** (normalized). Always be aware of which mode you're using and label appropriately.

---

**Exercise 6.**
The violin plot's strength is showing distribution shape; its weakness is being unfamiliar to most audiences. What is a reasonable communication strategy when presenting violin plots to a general (non-statistician) audience?

??? success "Solution to Exercise 6"
    Several strategies that compound:

    - **Annotate the median** with a horizontal line and label "median". Most viewers immediately understand the median.
    - **Annotate $Q_1$ and $Q_3$** with lines or shading at the same locations as a boxplot's box. This grounds the violin in the more-familiar boxplot semantics.
    - **Overlay the actual data points** (using `inner='points'` or `'sticks'`) for small samples. Direct visibility of the data points reassures the audience that you haven't smoothed away anything.
    - **Show the violin alongside a boxplot for the same data** the first time you use it. Explain: "the box plot tells you the median and IQR; the violin tells you where the density is concentrated."
    - **Use only when shape information matters.** A box plot suffices when only the median and IQR are interesting; reserve violins for cases where the audience needs to see multimodality, skew, or shape differences.

    The goal: violins should *add* information without adding cognitive load. If your audience would benefit more from a labeled bar chart, use that instead.
