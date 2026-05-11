# Outliers and Leverage

## Overview

**Outliers** are data points that significantly differ from other observations in a dataset. They may be unusually high or low and can arise due to variability in the data, errors in data collection, or they may indicate special cases that deserve further investigation. Detecting and understanding outliers is crucial because they can distort statistical analyses such as the mean, variance, and regression models.

---

## 1. Types of Outliers

**Univariate Outliers:** Unusual with respect to a single variable. For example, in a dataset of student heights, an individual who is extremely short or tall compared to the others.

**Multivariate Outliers:** Appear normal when each variable is considered separately, but unusual patterns emerge when the relationship between multiple variables is examined.

## 2. Causes of Outliers

- **Measurement Error:** Mistakes in data entry, instrument errors, or inaccuracies during measurement.
- **Experimental Error:** Anomalous conditions during data collection.
- **Natural Variation:** Inherent variability in the system being studied.
- **Sampling Error:** Rare cases included in the dataset or insufficient sample size.

## 3. Effects of Outliers

**Impact on Central Tendency:** Outliers pull the mean toward extreme values, making it an inaccurate representation. For example, a CEO's salary in a small sample of salaries can skew the mean upward significantly.

**Impact on Variability:** Outliers inflate variance and standard deviation, as these measures are sensitive to extreme values.

**Impact on Statistical Models:** Outliers can have a disproportionate influence on regression models, potentially leading to misleading or biased coefficients that reduce generalizability.

---

## 4. Identifying Outliers

### Box Plot Method

Data points beyond $1.5 \times \text{IQR}$ from $Q_1$ or $Q_3$ are flagged as outliers:

- Lower fence: $Q_1 - 1.5 \times \text{IQR}$
- Upper fence: $Q_3 + 1.5 \times \text{IQR}$

### Z-Score Method

The Z-score measures how many standard deviations a data point is from the mean. Points with $|Z| > 3$ are typically considered outliers:

$$
Z = \frac{x - \mu}{\sigma}
$$

### IQR Method

Values below $Q_1 - 1.5 \times \text{IQR}$ or above $Q_3 + 1.5 \times \text{IQR}$ are classified as outliers.

### Scatterplot (Multivariate)

In multivariate data, scatterplots can reveal points that deviate significantly from the overall pattern or trend.

### Cook's Distance (Regression)

Cook's Distance identifies influential data points that have a large impact on regression model predictions. High values indicate potential outliers with leverage.

---

## 5. Five-Number Summary

The five-number summary provides a concise description that naturally highlights potential outliers through the box plot:

$$
\text{Min} \quad Q_1 \quad \text{Median} \quad Q_3 \quad \text{Max}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.array([1, 2, 0, 0, 0, 1, 3, 1, 2, 1, 2, 4, 5, -1, -2, 0, 8])

quantiles = {"Min": 0, "Q1": 0.25, "Median": 0.5, "Q3": 0.75, "Max": 1}

for label, q in quantiles.items():
    print(f"{label:6} : {np.quantile(data, q)}")

fig, ax = plt.subplots(figsize=(2, 3))
ax.boxplot(data)
ax.set_title("Boxplot of Data")
plt.show()
```

### Comparative Box Plots

Box plots are particularly effective when comparing distributions across groups or conditions:

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

---

## 6. Handling Outliers

**Investigate the Source:** Confirm whether outliers are erroneous before taking action. If confirmed as errors, correct or remove them.

**Transform the Data:** Log or square-root transformations can reduce the influence of outliers by compressing the scale.

**Use Robust Statistical Methods:** The median, IQR, and robust regression techniques (e.g., Lasso, Ridge) are less sensitive to outliers.

**Trimming or Winsorizing:** Trimming removes extreme data points. Winsorizing replaces outliers with the nearest non-outlier value.

**Keep the Outliers:** Sometimes outliers represent rare but important cases (e.g., extreme market events in finance) and should be retained for further investigation.

---

## 7. Real-Life Examples

**Income Distribution:** Extreme outliers such as tech billionaire incomes drastically increase the mean, making the median a more representative measure.

**Stock Market Analysis:** Large market movements during crises (e.g., 2008 financial crisis) appear as outliers in historical price data.

**Medical Studies:** Patients with unique drug responses may be outliers that reveal important information about subgroup effects.

## Summary

Outliers deserve careful attention rather than automatic removal. Understanding their source—whether error, natural variation, or a genuinely rare event—determines the appropriate response. The combination of visual tools (box plots, scatter plots) and numerical methods (Z-scores, IQR fences, Cook's Distance) provides a robust framework for outlier detection and management.

## Exercises

**Exercise 1.**
For the dataset $4, 7, 8, 12, 14, 15, 16, 18, 19, 22, 25, 55$: (a) find $Q_1, Q_2, Q_3$; (b) compute the IQR; (c) apply the $1.5 \times \mathrm{IQR}$ rule and identify outliers; (d) describe the boxplot.

??? success "Solution to Exercise 1"
    (a) Lower half $\{4, 7, 8, 12, 14, 15\}$ → $Q_1 = (8 + 12)/2 = 10$. Full median $(15 + 16)/2 = 15.5$. Upper half $\{16, 18, 19, 22, 25, 55\}$ → $Q_3 = (19 + 22)/2 = 20.5$.

    (b) $\mathrm{IQR} = 20.5 - 10 = 10.5$.

    (c) Lower fence $= 10 - 1.5 \times 10.5 = -5.75$; upper fence $= 20.5 + 15.75 = 36.25$. Only $55 > 36.25$, so it is flagged as an outlier.

    (d) Box from 10 to 20.5 with a line at 15.5. Lower whisker reaches 4; upper whisker stops at 25 (the largest non-outlier). The point 55 appears as an isolated dot beyond the upper whisker. The right-side tail reveals positive skew.

---

**Exercise 2.**
**Why is $1.5$ the multiplier in the boxplot fence rule?** Derive what this number corresponds to under the normal distribution.

??? success "Solution to Exercise 2"
    For a standard normal: $Q_1 \approx -0.6745$, $Q_3 \approx 0.6745$, $\mathrm{IQR} \approx 1.349$. The upper fence is

    $$
    Q_3 + 1.5 \cdot \mathrm{IQR} \approx 0.6745 + 2.024 \approx 2.698
    $$

    The tail probability beyond this point is $P(Z > 2.698) \approx 0.0035$. With both tails, about 0.7% of normal data lies outside the fences.

    Tukey chose 1.5 (heuristically — there is no formal derivation) so that for normal data, **roughly 1 in 100 observations** is flagged. This produces a small but non-zero rate of "outliers" in clean normal data — useful for highlighting truly unusual values without overwhelming the analyst.

    Larger samples generate more flagged points in absolute terms even when the data is purely normal. For very large $n$, alternatives like $3 \times \mathrm{IQR}$ (Tukey's "far out") or distribution-aware tests (Grubbs', Dixon's) are sometimes preferred.

---

**Exercise 3.**
The **Z-score method** flags points with $|Z| > 3$. Under a normal distribution, what fraction of data is flagged? Why does this rule fail in the presence of multiple outliers?

??? success "Solution to Exercise 3"
    Under normality, $P(|Z| > 3) \approx 0.0027$ — about 0.27% of clean normal data is flagged.

    **Failure mechanism (masking):** if multiple outliers are present, they inflate the sample mean and SD. A point that would be 5 SDs from the *true* mean might be only 2 SDs from the *contaminated* sample mean — failing to be flagged. The outliers protect each other.

    **Fix:** use *robust* estimators of location and scale instead. The **modified Z-score** uses the median and MAD:

    $$
    M_i = 0.6745 \cdot \frac{x_i - \mathrm{median}(x)}{\mathrm{MAD}}
    $$

    Flag $|M_i| > 3.5$ (Iglewicz and Hoaglin 1993). Because MAD has breakdown 50%, masking is much harder to engineer.

---

**Exercise 4.**
Distinguish three categories of outliers: (a) **error outliers**, (b) **mixture outliers**, (c) **influential outliers in regression**. For each, give an example and a recommended action.

??? success "Solution to Exercise 4"
    **(a) Error outliers** — data-entry mistakes, instrument failures, miscoded values. Example: a height recorded as 7.2 m instead of 72 in (1.83 m). *Action:* investigate and correct or remove. Document the decision.

    **(b) Mixture outliers** — genuine observations from a different population than most of the data. Example: a wholesale customer in a dataset of retail transactions. *Action:* either model the mixture explicitly (mixture models, robust regression with heavy-tailed errors) or exclude with a clear rule and reported sensitivity.

    **(c) Influential outliers in regression** — points whose removal substantially changes fitted coefficients. Example: a single high-leverage point at extreme $x$. *Action:* compute **Cook's distance** and **DFBETAS** to quantify influence. If influential, refit without the point and report both estimates; if the conclusions disagree, the data is too sensitive to that point and additional samples are needed.

    The danger of conflating these categories: removing "outliers" indiscriminately can delete genuinely informative observations (mixture or influential) while keeping error outliers if their values happen to be near the bulk. *Investigate before removing.*

---

**Exercise 5.**
**Cook's distance** for observation $i$ in a regression with $p$ parameters is

$$
D_i = \frac{(\hat y - \hat y_{(i)})^T (\hat y - \hat y_{(i)})}{p \, s^2} = \frac{e_i^2}{p \, s^2} \cdot \frac{h_{ii}}{(1 - h_{ii})^2}
$$

where $h_{ii}$ is the leverage and $e_i$ the residual. Why is Cook's distance more informative than either residual or leverage alone?

??? success "Solution to Exercise 5"
    Cook's distance combines two things that each must be true for an observation to be influential:

    - **Large residual** ($e_i^2$ large) — the point is poorly fit by the model.
    - **Large leverage** ($h_{ii}/(1 - h_{ii})^2$ large) — the point's $x$-value is far from the mean of $x$'s, so the model has to "stretch" to fit it.

    A point with high residual but low leverage (extreme $y$ at typical $x$) is anomalous but does not drag the regression line — the abundance of other typical points anchors the slope. A point with high leverage but low residual (extreme $x$ that happens to be fit perfectly) is supported by the model — it's a powerful but consistent point.

    Only points with **both** large residual AND large leverage actually change the fitted coefficients when removed. Cook's distance is constructed to detect exactly this combination. Conventional threshold: investigate $D_i > 4/n$ or $D_i > 1$.

---

**Exercise 6.**
**Winsorization** at the 5%/95% level replaces values below the 5th percentile with the 5th-percentile value and values above the 95th with the 95th-percentile value. Compare this with **trimming** (deleting the extremes) and with **leaving outliers alone**. When is each appropriate?

??? success "Solution to Exercise 6"
    **Trimming**: discard observations below the 5th and above the 95th percentile. Result: $n$ shrinks. Useful when extreme values are clearly errors or contamination. The standard error of the resulting estimator can be smaller (less noise) but the sample size shrinks.

    **Winsorization**: replace extremes with the cut-off values. Result: $n$ unchanged but the data is squashed at the tails. Useful when extreme values are believed real but you want them to have bounded influence in a non-robust analysis (e.g., a sample mean computation). The squashed values retain partial influence on quantile-based statistics but not on tail-sensitive statistics like mean and variance.

    **Leave alone**: most appropriate when the analysis uses robust statistics (median, MAD, M-estimators) that are insensitive to extremes anyway, or when the extremes are the very phenomenon of interest (financial crisis returns, drug super-responders).

    **Recommendation:** never silently apply any of these. Always (1) plot the data to see whether the extremes look like errors or genuine signal; (2) report results both with and without the extremes; (3) when in doubt, prefer robust methods that don't require deciding upfront which points are "real."
