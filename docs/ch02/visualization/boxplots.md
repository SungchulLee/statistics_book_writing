# Boxplots

## Overview

A **box plot** (or box-and-whisker plot) is a standardized way of displaying the distribution of data based on the five-number summary: minimum, first quartile ($Q_1$), median ($Q_2$), third quartile ($Q_3$), and maximum. It provides a compact visual summary of center, spread, skewness, and outliers simultaneously.

## Anatomy of a Box Plot

The components of a box plot are:

- **Box:** Spans from $Q_1$ to $Q_3$, covering the interquartile range (IQR = $Q_3 - Q_1$). The length of the box represents the middle 50% of the data.
- **Median line:** A line inside the box at $Q_2$.
- **Whiskers:** Extend from the box to the most extreme data points within $1.5 \times \text{IQR}$ of $Q_1$ and $Q_3$.
- **Outliers:** Individual points plotted beyond the whiskers.

## Basic Box Plot: Titanic Passenger Ages

```python
import matplotlib.pyplot as plt
import pandas as pd

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url, index_col='PassengerId')

fig, ax = plt.subplots(figsize=(5, 3))
df['Age'].plot(kind='box', ax=ax, vert=False)
ax.set_title("Horizontal Boxplot of Passenger Ages on Titanic")
ax.set_xlabel("Age")
ax.spines[["top", "left", "right"]].set_visible(False)
plt.show()
```

## Detecting Skewness from Box Plots

Box plots provide a quick diagnostic for distribution shape:

$$
\begin{array}{lll}
\text{Left\_Box} > \text{Right\_Box} &\Rightarrow& \text{Left-skewed} \\
\text{Left\_Box} < \text{Right\_Box} &\Rightarrow& \text{Right-skewed} \\
\text{Boxes equal, Left\_Whisker} > \text{Right\_Whisker} &\Rightarrow& \text{Left-skewed} \\
\text{Boxes equal, Left\_Whisker} < \text{Right\_Whisker} &\Rightarrow& \text{Right-skewed} \\
\text{Both equal} &\Rightarrow& \text{Symmetric} \\
\end{array}
$$

## Paired Histogram and Box Plot

Displaying a histogram alongside a box plot makes the connection between shape and summary statistics explicit:

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

np.random.seed(0)
main_data = stats.norm().rvs(1_000)
right_1 = stats.norm(loc=2).rvs(200)
right_2 = stats.norm(loc=4).rvs(100)
combined = np.concatenate((main_data, right_1, right_2))

fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(12, 6))

ax_hist.hist(combined, density=True, bins=30)
ax_hist.set_title('Histogram of Right-Skewed Data')

ax_box.boxplot(combined, vert=False)
ax_box.set_title('Boxplot of Right-Skewed Data')

plt.tight_layout()
plt.show()
```

## Comparative Box Plots

Box plots are most powerful when used to compare distributions across groups:

```python
import numpy as np
import matplotlib.pyplot as plt

data_a = np.array([1, 2, 0, 0, 0, 1, 3, 1, 2, 1, 2, 4, 5, -1, -2, 0, 8])
data_b = np.array([1, 2, 0, 0, 0, 1, 3, 1, 2, 1, 2, 4, 5, -1, -2, 0, -8]) * 0.5
data_c = np.array([1, 2, 0, 0, 0, 1, 3, 1, 2, 1, 2, 4, 5, -1, -2, 0, 10, -7]) * 0.25

fig, ax = plt.subplots()
ax.boxplot([data_a, data_b, data_c],
           labels=["$10^4$", "$5 \\cdot 10^4$", "$10^5$"])
ax.plot([0, 1, 2, 3, 4], [1, 1, 1, 1, 1],
        label="FIM Delta", linestyle="--", color="r", alpha=0.7)
ax.legend()
ax.set_ylim(-10.0, 10.0)
ax.set_xlabel('Number of Samples')
ax.set_ylabel('MC Delta')
plt.show()
```

## Summary

Box plots are a compact, information-rich visualization that reveal center (median), spread (IQR and whisker length), skewness (box and whisker asymmetry), and outliers (individual points) all in a single graphic. They are especially effective for comparing distributions across groups or conditions.

## Exercises

**Exercise 1.**
A dataset has the five-number summary: Min $= 10$, $Q_1 = 25$, Median $= 35$, $Q_3 = 50$, Max $= 90$. Compute the IQR and the fence values. Are there any outliers according to the $1.5 \times \text{IQR}$ rule?

??? success "Solution to Exercise 1"
    The IQR is:

    $$
    \text{IQR} = Q_3 - Q_1 = 50 - 25 = 25
    $$

    The fences are:

    $$
    \text{Lower fence} = Q_1 - 1.5 \times \text{IQR} = 25 - 37.5 = -12.5
    $$

    $$
    \text{Upper fence} = Q_3 + 1.5 \times \text{IQR} = 50 + 37.5 = 87.5
    $$

    The minimum (10) is above the lower fence ($-12.5$), so there is no lower outlier. However, the maximum (90) exceeds the upper fence (87.5), so **90 is an outlier**. In the boxplot, the upper whisker would extend to the largest value at or below 87.5, and the point at 90 would appear as an individual outlier marker.

---

**Exercise 2.**
Two boxplots are shown side by side. Boxplot A has a short box with long whiskers, while Boxplot B has a long box with short whiskers. Both have the same range. Compare the two distributions in terms of where the data is concentrated.

??? success "Solution to Exercise 2"
    **Boxplot A** (short box, long whiskers): The middle 50% of the data is tightly concentrated around the median, but the tails extend far. This suggests a distribution that is **peaked near the center** with observations spread thinly in the tails — a leptokurtic or heavy-tailed shape.

    **Boxplot B** (long box, short whiskers): The middle 50% of the data is widely spread, but there are no extreme values far from the quartiles. This suggests a distribution where data is more **uniformly spread** across a range, resembling a platykurtic or uniform-like shape.

    Even though both have the same overall range, the distributions are fundamentally different: A concentrates observations near the median with a few far-flung values, while B spreads observations more evenly.

---

**Exercise 3.**
Describe how a boxplot of a perfectly symmetric distribution would look. What specific features would indicate perfect symmetry?

??? success "Solution to Exercise 3"
    In a perfectly symmetric distribution:

    - The **median line** would be exactly in the center of the box, meaning $Q_2 - Q_1 = Q_3 - Q_2$.
    - The **whiskers** would have equal length on both sides: the distance from $Q_1$ to the lower whisker endpoint equals the distance from $Q_3$ to the upper whisker endpoint.
    - If any **outliers** exist, they would appear symmetrically on both sides (same number and approximately equidistant from the box on each side).

    The normal distribution is a classic example: a boxplot of a large sample from $N(\mu, \sigma^2)$ would show these symmetric features.

---

**Exercise 4.**
A boxplot of exam scores for Class X shows the median at 75, $Q_1 = 65$, $Q_3 = 85$, a lower whisker at 40, and an upper whisker at 100 (no outliers). A boxplot for Class Y shows the median at 75, $Q_1 = 70$, $Q_3 = 80$, a lower whisker at 55, and an upper whisker at 95 (no outliers). Compare the two classes.

??? success "Solution to Exercise 4"
    Both classes have the same median (75), so the "typical" student performance is similar. However, they differ substantially in spread:

    - **Class X** has $\text{IQR} = 85 - 65 = 20$ and range $= 100 - 40 = 60$. The scores are widely dispersed, indicating high variability in student performance.
    - **Class Y** has $\text{IQR} = 80 - 70 = 10$ and range $= 95 - 55 = 40$. The scores are more tightly clustered around the median.

    Class Y is more homogeneous: most students scored between 70 and 80. Class X has a wider spread, suggesting a mix of very strong and very weak performers. An instructor seeing these boxplots might investigate why Class X has such high variability — perhaps different preparation levels or a mix of student backgrounds.

---

**Exercise 5.**
Tukey's original boxplot specification places the whiskers at the most extreme data point within $1.5 \cdot \mathrm{IQR}$ of the box. Why is this preferable to placing them at the min and max?

??? success "Solution to Exercise 5"
    Whiskers at the min and max have two problems:

    - **No outlier signal**: if the maximum is an outlier, the whisker extends to it and is indistinguishable from a long upper whisker on a non-outlier distribution. The viewer cannot tell whether the data has a long tail or a single extreme observation.
    - **Sensitivity to a single point**: a single very extreme observation pulls the whisker to it, distorting the visual scale of the entire plot. Other features (the box, smaller variations) become compressed against the axis.

    Tukey's choice separates the typical range (whiskers to the most extreme *non-outlier*) from individual outliers (plotted as separate points beyond the whiskers). This visualization decision encodes a model: "the bulk of the data is described by the box and whiskers; anything outside is suspicious or interesting and deserves individual attention." For approximately normal data, roughly 0.7% of observations fall beyond the $1.5\,\mathrm{IQR}$ fences, so a few flagged points are expected from clean data — the rate that Tukey calibrated to.

---

**Exercise 6.**
**Notched boxplots** add a "notch" around the median at $\pm 1.57 \cdot \mathrm{IQR}/\sqrt{n}$. What does the notch represent, and how is it used for visual hypothesis testing across groups?

??? success "Solution to Exercise 6"
    The notch represents an approximate **95% confidence interval for the median** based on McGill, Tukey, and Larsen (1978): notch half-width $\approx 1.57 \cdot \mathrm{IQR}/\sqrt{n}$.

    **Use for visual comparison:** when two notched box plots are drawn side by side, **non-overlapping notches indicate a statistically significant difference between the medians** (at roughly the 5% level). Overlapping notches suggest no significant difference. This provides a quick visual analog of a Mann–Whitney test or median test without computing $p$-values.

    **Caveats:**

    - The 1.57 constant comes from a normal-approximation argument and is approximate for small samples or non-normal data.
    - Notches can extend beyond the box (above $Q_3$ or below $Q_1$) for small samples; this looks visually odd but indicates the median is poorly determined.
    - The visual rule has higher Type I error than a formal test in some configurations. Use for exploration; back up with a formal test when stakes are high.

    Notched plots are produced in matplotlib via `boxplot(notch=True)` and in seaborn via `sns.boxplot(notch=True)`. They are particularly useful in publications comparing many groups simultaneously, where formal pairwise tests would multiply the number of comparisons.
