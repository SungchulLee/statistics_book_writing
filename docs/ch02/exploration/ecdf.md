# ECDF and Quantiles

## Overview

The **empirical cumulative distribution function (ECDF)** and **quantiles** provide complementary views of a distribution that avoid the bin-width sensitivity of histograms. The ECDF maps every data value to the proportion of observations at or below that value, producing a step function that converges to the true CDF as the sample size grows. Quantiles invert this relationship, answering: "At what value does a given fraction of the data fall below?"

## The Empirical CDF

For a sample $x_1, x_2, \ldots, x_n$, the ECDF is defined as

$$
\hat{F}(t) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(x_i \le t)
$$

where $\mathbf{1}(\cdot)$ is the indicator function. Key properties:

- $\hat{F}$ is a non-decreasing step function ranging from 0 to 1.
- Each jump has height $1/n$ (or multiples for tied values).
- By the Glivenko–Cantelli theorem, $\hat{F}$ converges uniformly to the true CDF $F$ almost surely.

### ECDF vs. Theoretical CDF

Comparing the ECDF to a parametric CDF is a powerful diagnostic for assessing distributional assumptions.

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

np.random.seed(1)
x = 4 + np.random.normal(0, 1.5, 100)

loc = x.mean()
scale = x.std()

x.sort()
cdf = stats.norm(loc=loc, scale=scale).cdf(x)

fig, ax = plt.subplots(figsize=(12, 3))
ax.ecdf(x, ls="-", c="r", label="Empirical CDF")
ax.plot(x, cdf, "-b", label="Theoretical CDF")
ax.legend()
plt.show()
```

When the empirical and theoretical curves closely overlap, the parametric model is a good fit. Systematic departures indicate skewness, heavy tails, or multimodality.

### CDF and PDF Side by Side

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

loc = 1
scale = 2
normal = stats.norm(loc=loc, scale=scale)

x = np.linspace(loc - 3 * scale, loc + 3 * scale, 1_000)
pdf = normal.pdf(x)
cdf = normal.cdf(x)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, pdf, "-b", label="PDF")
ax.plot(x, cdf, "-r", label="CDF")
ax.legend()
plt.show()
```

The PDF shows where density is concentrated; the CDF shows cumulative probability. Together they give a complete picture of the distribution.

## Quantiles, Percentiles, and Quartiles

### Percentiles

The $p$-th **percentile** $P_p$ is the value below which $p\%$ of the data falls. Reading a cumulative relative frequency graph at height $p/100$ on the y-axis and projecting horizontally to the curve gives the percentile on the x-axis.

### Quartiles

The three quartiles divide the data into four equal parts:

$$
\begin{array}{llll}
\text{First Quartile} & Q_1 &=& P_{25} \\
\text{Second Quartile} & Q_2 &=& P_{50} \\
\text{Third Quartile} & Q_3 &=& P_{75} \\
\end{array}
$$

### Deciles

$$
\begin{array}{llll}
D_1 = P_{10}, \quad D_2 = P_{20}, \quad \ldots, \quad D_9 = P_{90}
\end{array}
$$

### Relationship to Median

$$
\text{Median} = Q_2 = D_5 = P_{50}
$$

## Computing Quantiles in Python

Three common approaches yield identical results:

```python
import pandas as pd
import numpy as np
from scipy import stats

data = {'x': [4, 4, 6, 7, 10, 11, 12, 14, 15]}
df = pd.DataFrame(data)

# pandas: q in [0, 1]
print(f"{df.x.quantile(0.75) = }")

# numpy: q in [0, 100]
print(f"{np.percentile(df.x.values, 75) = }")

# scipy: q in [0, 100]
print(f"{stats.scoreatpercentile(df.x.values, 75) = }")
```

## Example: Sugar Content in Starbucks Drinks

Nutritionists measured sugar content (in grams) for 32 Starbucks drinks. Using the cumulative relative frequency graph:

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 55, 5)
y = [0, 0.1, 0.1, 0.2, 0.3, 0.5, 0.6, 0.6, 0.8, 0.9, 1.0]

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, y, '-o')
ax.set_xlabel("Sugar Content (g)")
ax.set_ylabel("Cumulative Relative Frequency")
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.grid()
plt.show()
```

**Questions and answers:**

1. A coffee with 15 grams of sugar is at approximately the **20th percentile**.
2. The **median** (50th percentile) is approximately **25 grams**.
3. $Q_1 \approx 17.5$ g, $Q_3 \approx 38.5$ g, so $\text{IQR} = Q_3 - Q_1 \approx 21$ g.

## The Five-Number Summary

The five-number summary captures the key quantiles of a distribution:

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

## Q-Q Plots: Quantile-Quantile Comparison

A **Q-Q plot** compares the quantiles of observed data against the quantiles of a theoretical distribution. If the data follows the reference distribution, the points lie along the diagonal reference line.

### Q-Q Plot Against Normal Distribution

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_qq(data, dist="norm", sparams=(), figsize=(12, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    stats.probplot(data, dist=dist, sparams=sparams, plot=ax)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title('Q-Q Plot')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Ordered Values')
    plt.show()

np.random.seed(0)
sample_data = np.random.normal(loc=0, scale=1, size=1000)
plot_qq(sample_data, dist="norm")
```

### Q-Q Plot Against Exponential Distribution

```python
np.random.seed(0)
sample_data = np.random.exponential(scale=1, size=1000)
plot_qq(sample_data, dist="expon")
```

### Q-Q Plot Against Chi-Square Distribution

```python
np.random.seed(0)
sample_data = np.random.chisquare(df=10, size=1000)
plot_qq(sample_data, dist="chi2", sparams=(10,))
```

### Diagnostic Use: Chi-Square Data Against Normal Q-Q Plot

When chi-square data is plotted against a normal reference, the systematic curvature reveals right skewness—confirming that the normal model is inappropriate.

```python
np.random.seed(0)
sample_data = np.random.chisquare(df=10, size=1000)
plot_qq(sample_data, dist="norm")  # Systematic departure from the line
```

## Summary

The ECDF and quantiles provide bin-free, exact representations of empirical distributions. The ECDF is ideal for comparing distributions or assessing goodness of fit, while quantiles and the five-number summary offer concise numerical summaries. Q-Q plots extend these ideas into a powerful visual diagnostic for checking distributional assumptions.
