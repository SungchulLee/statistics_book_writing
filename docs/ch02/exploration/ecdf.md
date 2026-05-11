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

## Exercises

**Exercise 1.**
For the dataset $\{2, 5, 5, 7, 10\}$: (a) write the ECDF $\hat F(x)$ as a piecewise function; (b) compute $\hat F(5)$ and $\hat F(6)$; (c) determine the 50th percentile (median).

??? success "Solution to Exercise 1"
    (a) With $n = 5$:

    $$
    \hat F(x) = \begin{cases} 0 & x < 2 \\ 1/5 & 2 \le x < 5 \\ 3/5 & 5 \le x < 7 \\ 4/5 & 7 \le x < 10 \\ 1 & x \ge 10 \end{cases}
    $$

    Jumps of $1/n$ at each unique value; the jump at 5 is $2/5$ because 5 occurs twice.

    (b) $\hat F(5) = 3/5 = 0.6$ (three values $\le 5$); $\hat F(6) = 3/5 = 0.6$ (no value strictly between 5 and 7).

    (c) The 50th percentile is the smallest $x$ with $\hat F(x) \ge 0.5$: that's $x = 5$.

---

**Exercise 2.**
State the **Glivenko–Cantelli theorem** and interpret what it tells us about using the ECDF as an estimator of the true CDF.

??? success "Solution to Exercise 2"
    Let $X_1, X_2, \ldots$ be i.i.d. with CDF $F$, and let $\hat F_n$ be the empirical CDF. The Glivenko–Cantelli theorem states

    $$
    \sup_x |\hat F_n(x) - F(x)| \xrightarrow{\text{a.s.}} 0 \quad \text{as } n \to \infty
    $$

    The convergence is *uniform* over all $x$ — not just pointwise. This is what allows the ECDF to serve as a general-purpose distribution estimator: any continuity statistic of $F$ (median, IQR, skewness, etc.) can be consistently estimated by the corresponding plug-in statistic of $\hat F_n$.

    The **Dvoretzky–Kiefer–Wolfowitz (DKW) inequality** quantifies the rate: $P(\sup_x |\hat F_n - F| > \varepsilon) \le 2 e^{-2n\varepsilon^2}$. With $n = 100$, the worst-case discrepancy is $\le 0.1$ with probability $\ge 0.96$ — fast for a non-parametric estimator.

---

**Exercise 3.**
Compare the **ECDF** and the **histogram** as distributional summaries. List two advantages of each.

??? success "Solution to Exercise 3"
    **ECDF advantages:** (1) bin-free — no arbitrary bin-width choice; (2) uses every observation exactly; (3) converges uniformly at parametric rate $O(1/\sqrt{n})$; (4) easy to compare two distributions (overlay two ECDFs or compute K-S distance).

    **Histogram advantages:** (1) more intuitive visual for typical readers — "where does most of the data live?"; (2) emphasizes density (peaks, modes, gaps) that the ECDF flattens out across the y-axis range $[0, 1]$; (3) reveals multimodality immediately; (4) standard in publications and dashboards.

    In practice: use the **ECDF** for goodness-of-fit and distribution comparison; use the **histogram** (or kernel density estimate) for visual communication of shape.

---

**Exercise 4.**
**Quantile interpretation.** A standardized test reports that a student is at the 85th percentile. State precisely what this means. Discuss the difference between this and "scoring 85% on the test."

??? success "Solution to Exercise 4"
    Being at the 85th percentile means **85% of test takers scored at or below this student's score** (with 15% scoring higher). The percentile is a *rank-based* measure relative to the reference population.

    "Scoring 85% on the test" is an *absolute* performance measure — the fraction of questions answered correctly. The two are unrelated:

    - A student scoring 85% on a very hard test might be at the 99th percentile (most others scored worse).
    - A student scoring 85% on a very easy test might be at the 30th percentile (most others scored even better).

    Percentile ranks are commonly used in standardized testing because they are invariant to the difficulty of the specific test version. Comparisons across years or test forms use percentiles, not raw scores, after re-norming.

---

**Exercise 5.**
The **Kolmogorov–Smirnov statistic** $D_n = \sup_x |\hat F_n(x) - F_0(x)|$ tests whether data came from a specified distribution $F_0$. Why is this a natural test statistic, and how does its null distribution depend on $F_0$?

??? success "Solution to Exercise 5"
    **Natural choice:** Glivenko–Cantelli guarantees $D_n \to 0$ under the null ($F = F_0$). Under the alternative ($F \ne F_0$), $D_n$ converges to $\sup_x |F(x) - F_0(x)| > 0$. So $D_n$ separates the null from the alternative for any continuous alternative.

    **Null distribution:** for a fully specified $F_0$, $D_n$ has a distribution that depends only on $n$, not on $F_0$ itself. This is the **distribution-free** property of K-S: $F_0(X)$ is uniformly distributed on $[0, 1]$ under the null, so $D_n$ effectively measures discrepancy from the uniform distribution, regardless of the original $F_0$. This makes critical values usable for any continuous reference distribution.

    **Caveat:** if $F_0$ has unknown parameters estimated from the same data (e.g., testing normality with $\hat\mu, \hat\sigma$ estimated from the sample), the test is no longer distribution-free. The **Lilliefors test** or **Shapiro-Wilk** is appropriate in that setting.

---

**Exercise 6.**
A **Q-Q plot** compares quantiles of the data against quantiles of a reference distribution. Interpret the following patterns: (a) points fall on a straight line; (b) S-shaped curve; (c) systematic curvature that's concave-up; (d) heavy departures in the tails only.

??? success "Solution to Exercise 6"
    (a) **Straight line** (matching the reference): the data are well-approximated by the reference distribution (typically the standard normal, possibly after centering and scaling). Slope = SD of the data, intercept = mean.

    (b) **S-shape** (rising slowly, then quickly, then slowly): the data has **lighter tails** than the reference — fewer extreme values. The middle is steeper than the reference's middle. Indicates a light-tailed distribution (e.g., uniform or truncated).

    (c) **Concave-up curvature** (steeper at the top right): the data has **right skew** — the upper tail is longer than the reference. Common for income, count data, lognormal, or exponential samples plotted against a normal reference.

    (d) **Tail departures only**: the bulk of the data is well-modeled, but extreme observations don't match. Could indicate either heavy tails (more extremes than the reference, e.g., $t$ distribution) or outliers from a contaminating process. Distinguish by looking at multiple samples or running a robust analysis on the bulk only.

    The Q-Q plot is more informative than a single goodness-of-fit $p$-value because it shows *where* the model fails, not just *whether* it fails.
