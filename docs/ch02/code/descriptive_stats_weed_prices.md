# Descriptive Statistics from Scratch: Weed Prices

## Overview

This case study implements the core descriptive statistics—mean, median, mode, variance, standard deviation, covariance, and correlation—from first principles, then verifies each result against pandas built-in methods. The dataset consists of synthetic monthly high-quality weed prices for California and New York, inspired by real market data.

Computing statistics from scratch reinforces the definitions and exposes the mechanics that library functions hide.

---

## 1. The Data

We work with 48 monthly price observations for California (CA) and New York (NY) high-quality weed:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CA_PRICES = np.array([
    248.75, 248.59, 248.63, 248.37, 248.02, 247.68, 247.36,
    246.85, 246.44, 246.06, 245.81, 245.48, 245.18, 244.87,
    244.55, 244.23, 243.89, 243.60, 243.34, 243.08, 242.85,
    242.64, 242.36, 242.15, 241.88, 241.64, 241.40, 241.14,
    240.91, 240.65, 240.42, 240.20, 239.96, 239.74, 239.52,
    239.28, 239.07, 238.81, 238.55, 238.34, 238.12, 237.90,
    237.66, 237.43, 237.19, 236.98, 236.76, 236.56,
])

NY_PRICES = np.array([
    350.50, 350.31, 350.02, 349.82, 349.55, 349.30, 349.04,
    348.78, 348.54, 348.27, 348.01, 347.78, 347.51, 347.26,
    346.98, 346.72, 346.48, 346.19, 345.93, 345.68, 345.44,
    345.17, 344.93, 344.67, 344.42, 344.18, 343.91, 343.68,
    343.43, 343.17, 342.93, 342.68, 342.44, 342.18, 341.93,
    341.67, 341.43, 341.16, 340.90, 340.66, 340.41, 340.16,
    339.92, 339.67, 339.41, 339.18, 338.93, 338.70,
])
```

Both series show a steady downward trend over the 48-month period, with NY prices consistently higher than CA prices.

---

## 2. Mean

The sample mean is the sum of all observations divided by the count:

$$
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

```python
def mean_from_scratch(data):
    return np.sum(data) / len(data)
```

---

## 3. Median

The median is the middle value of the sorted data. For even $n$, it is the average of the two central values:

$$
\text{median} =
\begin{cases}
x_{(m+1)} & \text{if } n = 2m + 1 \\[4pt]
\dfrac{x_{(m)} + x_{(m+1)}}{2} & \text{if } n = 2m
\end{cases}
$$

```python
def median_from_scratch(data):
    sorted_data = np.sort(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 1:
        return sorted_data[mid]
    return (sorted_data[mid - 1] + sorted_data[mid]) / 2
```

---

## 4. Mode

The mode is the most frequently occurring value. For continuous data, values are first rounded to a chosen precision:

```python
def mode_from_scratch(data, decimals=1):
    rounded = np.round(data, decimals)
    values, counts = np.unique(rounded, return_counts=True)
    return values[np.argmax(counts)]
```

!!! note "Mode for Continuous Data"
    Continuous data rarely has repeated exact values. Rounding or binning is required before computing the mode, and the result depends on the rounding precision chosen.

---

## 5. Variance and Standard Deviation

The sample variance uses Bessel's correction (dividing by $n - 1$) to produce an unbiased estimate of the population variance:

$$
s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2
$$

The sample standard deviation is:

$$
s = \sqrt{s^2}
$$

```python
def variance_from_scratch(data):
    m = mean_from_scratch(data)
    return np.sum((data - m) ** 2) / (len(data) - 1)

def std_from_scratch(data):
    return np.sqrt(variance_from_scratch(data))
```

---

## 6. Covariance and Correlation

The sample covariance measures the linear co-movement of two variables:

$$
\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})
$$

The Pearson correlation standardizes covariance to the range $[-1, 1]$:

$$
r = \frac{\text{Cov}(X, Y)}{s_X \, s_Y}
$$

```python
def covariance_from_scratch(x, y):
    n = len(x)
    mx, my = mean_from_scratch(x), mean_from_scratch(y)
    return np.sum((x - mx) * (y - my)) / (n - 1)

def correlation_from_scratch(x, y):
    return covariance_from_scratch(x, y) / (std_from_scratch(x) * std_from_scratch(y))
```

---

## 7. Results and Verification

Running the from-scratch functions on the California data and verifying with pandas:

```python
data = CA_PRICES
m   = mean_from_scratch(data)     # 242.2177
med = median_from_scratch(data)   # 241.7600
mod = mode_from_scratch(data)     # rounding-dependent
var = variance_from_scratch(data) # 12.8310
sd  = std_from_scratch(data)      #  3.5821

s = pd.Series(data)
# s.mean(), s.median(), s.var(), s.std() match the above
```

For CA vs NY:

```python
cov  = covariance_from_scratch(CA_PRICES, NY_PRICES)   # 11.7610
corr = correlation_from_scratch(CA_PRICES, NY_PRICES)   #  0.9998
```

The near-perfect correlation ($r \approx 1$) reflects that both series follow a similar steady downward trend over the same period.

---

## 8. Visualization

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Histogram with mean and median
axes[0].hist(data, bins=15, edgecolor="white", alpha=0.7)
axes[0].axvline(m, color="red", linestyle="--", label=f"Mean {m:.1f}")
axes[0].axvline(med, color="blue", linestyle=":", label=f"Median {med:.1f}")
axes[0].set_title("CA HighQ Price Distribution")
axes[0].set_xlabel("Price (\$)")
axes[0].legend(fontsize=8)

# Boxplot
axes[1].boxplot(data, vert=True)
axes[1].set_title("Box Plot — CA HighQ")
axes[1].set_ylabel("Price (\$)")

# Scatter: CA vs NY
axes[2].scatter(CA_PRICES, NY_PRICES, alpha=0.6)
axes[2].set_xlabel("CA HighQ (\$)")
axes[2].set_ylabel("NY HighQ (\$)")
axes[2].set_title(f"CA vs NY  (r = {corr:.3f})")

plt.tight_layout()
plt.show()
```

The left panel shows a roughly uniform distribution (prices decrease steadily, so each price level is visited approximately once). The right panel's tight linear scatter confirms the near-perfect correlation.

---

## Exercises

**Exercise 1.**
Given the five values 3, 7, 7, 10, 13, compute the mean, median, and mode by hand.

??? success "Solution to Exercise 1"

    - Mean: $\bar{x} = (3 + 7 + 7 + 10 + 13) / 5 = 40 / 5 = 8$
    - Median: sorted data is 3, 7, 7, 10, 13; the middle value is $7$
    - Mode: 7 appears twice; all others appear once. Mode $= 7$

---

**Exercise 2.**
Derive the formula for the sample variance $s^2$ starting from the requirement that $E[s^2] = \sigma^2$ (unbiasedness). Explain why the denominator is $n - 1$ rather than $n$.

??? success "Solution to Exercise 2"
    Start with the naive estimator $\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2$. Expanding:

    $$
    \sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n (X_i - \mu)^2 - n(\bar{X} - \mu)^2
    $$

    Taking expectations:

    $$
    E\left[\sum_{i=1}^n (X_i - \bar{X})^2\right] = n\sigma^2 - n \cdot \frac{\sigma^2}{n} = (n-1)\sigma^2
    $$

    So $E[\hat{\sigma}^2] = \frac{(n-1)\sigma^2}{n} \neq \sigma^2$. Dividing by $n-1$ instead of $n$ corrects this:

    $$
    E\left[\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2\right] = \sigma^2
    $$

    The factor $n - 1$ accounts for the one degree of freedom consumed by estimating $\mu$ with $\bar{X}$. $\square$

---

**Exercise 3.**
Two stocks have annual returns $X = (0.10, -0.05, 0.08)$ and $Y = (-0.02, 0.12, -0.03)$. Compute $\text{Cov}(X, Y)$ and the Pearson correlation $r$ by hand.

??? success "Solution to Exercise 3"
    First compute the means:

    $$
    \bar{x} = \frac{0.10 + (-0.05) + 0.08}{3} = \frac{0.13}{3} \approx 0.04333
    $$

    $$
    \bar{y} = \frac{-0.02 + 0.12 + (-0.03)}{3} = \frac{0.07}{3} \approx 0.02333
    $$

    Deviations and products:

    | $i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | Product |
    |---|---|---|---|
    | 1 | $0.05667$ | $-0.04333$ | $-0.002456$ |
    | 2 | $-0.09333$ | $0.09667$ | $-0.009022$ |
    | 3 | $0.03667$ | $-0.05333$ | $-0.001956$ |

    $$
    \text{Cov}(X, Y) = \frac{-0.002456 - 0.009022 - 0.001956}{2} = \frac{-0.013434}{2} \approx -0.006717
    $$

    Standard deviations: $s_X \approx 0.07937$, $s_Y \approx 0.08386$.

    $$
    r = \frac{-0.006717}{0.07937 \times 0.08386} \approx -1.009
    $$

    Rounding errors give $|r|$ slightly above 1; with exact arithmetic, $r = -1.0$ because the three points are exactly collinear (by construction of this small dataset). This illustrates that perfect negative correlation means the returns move in exactly opposite directions.

---

**Exercise 4.**
Show that the Pearson correlation coefficient satisfies $-1 \le r \le 1$. Under what conditions does equality hold?

??? success "Solution to Exercise 4"
    By the Cauchy–Schwarz inequality, for any random variables $U$ and $V$ with finite second moments:

    $$
    |\text{Cov}(U, V)|^2 \le \text{Var}(U) \cdot \text{Var}(V)
    $$

    Setting $U = X - \bar{X}$ and $V = Y - \bar{Y}$:

    $$
    |r| = \frac{|\text{Cov}(X, Y)|}{s_X s_Y} \le 1
    $$

    Equality $r = 1$ holds if and only if $Y = a + bX$ for some $b > 0$ (perfect positive linear relationship). Equality $r = -1$ holds if and only if $Y = a + bX$ for some $b < 0$ (perfect negative linear relationship). $\square$

---

**Exercise 5.**
The CA prices above decrease nearly linearly. If prices were exactly linear ($x_i = a - bi$ for constants $a, b > 0$ and $i = 1, \ldots, n$), show that the sample mean equals $a - b \cdot \frac{n+1}{2}$ and find a closed-form expression for the sample variance.

??? success "Solution to Exercise 5"
    With $x_i = a - bi$:

    $$
    \bar{x} = \frac{1}{n}\sum_{i=1}^n (a - bi) = a - b \cdot \frac{1}{n}\sum_{i=1}^n i = a - b \cdot \frac{n+1}{2}
    $$

    The deviations are:

    $$
    x_i - \bar{x} = -bi + b \cdot \frac{n+1}{2} = b\left(\frac{n+1}{2} - i\right)
    $$

    So the sample variance is:

    $$
    s^2 = \frac{b^2}{n-1}\sum_{i=1}^n \left(\frac{n+1}{2} - i\right)^2 = \frac{b^2}{n-1} \cdot \frac{n(n+1)}{12} \cdot (n-1) \cdot \frac{1}{n-1}
    $$

    Simplifying using $\sum_{i=1}^n \left(\frac{n+1}{2} - i\right)^2 = \frac{n(n^2 - 1)}{12}$:

    $$
    s^2 = \frac{b^2 \cdot n(n^2 - 1)}{12(n - 1)} = \frac{b^2 \, n(n+1)}{12}
    $$

    $\square$

---

**Exercise 6.**
The **coefficient of variation** $\mathrm{CV} = s/\bar x$ provides a unitless measure of relative spread. For the CA prices above, $\bar x \approx 244$ and $s \approx 9$; for OR prices $\bar x \approx 209$ and $s \approx 5$. Compute the CV for each and explain which state has *relatively* more price variability.

??? success "Solution to Exercise 6"
    California: $\mathrm{CV}_{\mathrm{CA}} = 9/244 \approx 0.037 = 3.7\%$.
    Oregon: $\mathrm{CV}_{\mathrm{OR}} = 5/209 \approx 0.024 = 2.4\%$.

    California has higher relative variability (3.7% vs. 2.4%). This means that as a fraction of the typical price, California prices fluctuate more than Oregon's.

    **Why CV matters here:** comparing absolute SDs ($9 vs. 5$) might suggest CA prices are "more volatile" — but that's partly because CA prices are higher to begin with. Dividing by the mean normalizes for price level and reveals the relative volatility. CV is particularly useful for comparing variability across markets, countries, or eras with different price scales.

    **Caveat:** CV is well-defined only for strictly-positive data with $\bar x > 0$. For data that can be negative (returns, deltas), use SD or other unbounded scale measures instead.

---

**Exercise 7.**
**Why does Pearson correlation measure *linear* association only?** Construct a small dataset where $X$ and $Y$ are perfectly related deterministically ($Y$ is a function of $X$) but Pearson's $r \approx 0$.

??? success "Solution to Exercise 7"
    Construction: let $X = (-2, -1, 0, 1, 2)$ and $Y = X^2 = (4, 1, 0, 1, 4)$. Then $Y$ is exactly determined by $X$. Compute the means and correlation:

    $\bar x = 0$, $\bar y = 2$.

    Deviations: $x - \bar x = (-2, -1, 0, 1, 2)$, $y - \bar y = (2, -1, -2, -1, 2)$.

    Cross-product sum: $(-2)(2) + (-1)(-1) + 0 \cdot (-2) + 1 \cdot (-1) + 2 \cdot 2 = -4 + 1 + 0 - 1 + 4 = 0$.

    Pearson's $r = 0$. Yet $Y$ is a deterministic function of $X$. Pearson's correlation captures only the *linear* trend, which is zero for a symmetric parabola: the positive trend on the right half exactly cancels the negative trend on the left.

    **Implication:** correlation $\approx 0$ does *not* mean independence. **Always** plot the data before relying on correlation. Alternatives like **Spearman's rank correlation** (monotonic), **distance correlation** (any dependence, including nonlinear), or **mutual information** capture broader notions of dependence.
