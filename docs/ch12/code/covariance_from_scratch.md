# Covariance from Scratch

## Overview

This page builds the covariance and Pearson correlation coefficient from first principles, step by step. Using synthetic price data for two states that share a common macroeconomic trend, we compute each quantity by hand, verify against library implementations, and illustrate why correlated time series do not imply causation.

---

## Sample Covariance

For paired observations $(x_1, y_1), \ldots, (x_n, y_n)$, the **sample covariance** is defined as

$$
\text{Cov}(X, Y) = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})
$$

where $\bar{x}$ and $\bar{y}$ are the sample means. The denominator $n - 1$ (Bessel's correction) yields an unbiased estimator of the population covariance.

Each term $(x_i - \bar{x})(y_i - \bar{y})$ is called a **deviation product**:

- **Positive** when $x_i$ and $y_i$ deviate from their means in the same direction.
- **Negative** when they deviate in opposite directions.

If positive products dominate, the covariance is positive, indicating that the variables tend to move together.

---

## Pearson Correlation from Covariance

The Pearson correlation coefficient normalizes the covariance by the product of the standard deviations:

$$
r = \frac{\text{Cov}(X, Y)}{s_X \, s_Y}
$$

where $s_X = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2}$ is the sample standard deviation (and similarly for $s_Y$). This normalization ensures $-1 \le r \le 1$.

---

## Step-by-Step Implementation

```python
import numpy as np


def covariance_step_by_step(x, y):
    """Compute sample covariance and return intermediate deviations."""
    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    x_dev = x - x_mean
    y_dev = y - y_mean
    cov = np.sum(x_dev * y_dev) / (n - 1)
    return cov, x_dev, y_dev


def pearson_r_step_by_step(x, y):
    """Compute Pearson r from first principles."""
    cov, _, _ = covariance_step_by_step(x, y)
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    return cov / (sx * sy)
```

---

## Generating the Data

We simulate weekly prices for two states (CA and NY) that share a common downward trend but have independent noise:

$$
\text{CA}_t = 248 + \text{trend}_t + \varepsilon_t^{(\text{CA})}, \qquad
\text{NY}_t = 350 + 0.8\,\text{trend}_t + \varepsilon_t^{(\text{NY})}
$$

where $\text{trend}_t$ is a linear decrease from 0 to $-12$ over 48 weeks, $\varepsilon_t^{(\text{CA})} \sim \mathcal{N}(0, 0.5^2)$, and $\varepsilon_t^{(\text{NY})} \sim \mathcal{N}(0, 0.6^2)$.

```python
np.random.seed(42)
WEEKS = 48
trend = np.linspace(0, -12, WEEKS)

CA = 248.0 + trend + np.random.normal(0, 0.5, WEEKS)
NY = 350.0 + trend * 0.8 + np.random.normal(0, 0.6, WEEKS)
```

---

## Computing and Verifying

```python
import pandas as pd

cov, x_dev, y_dev = covariance_step_by_step(CA, NY)
r = pearson_r_step_by_step(CA, NY)

print(f"CA mean     = {CA.mean():.4f}")
print(f"NY mean     = {NY.mean():.4f}")
print(f"Covariance  = {cov:.4f}")
print(f"Pearson r   = {r:.4f}")

# Verify against library functions
df = pd.DataFrame({"CA": CA, "NY": NY})
print(f"pandas cov  = {df['CA'].cov(df['NY']):.4f}")
print(f"pandas corr = {df['CA'].corr(df['NY']):.4f}")
print(f"numpy corr  = {np.corrcoef(CA, NY)[0, 1]:.4f}")
```

All three methods should produce identical results, confirming our from-scratch implementation.

---

## Visualization

Three panels tell the complete story:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel 1: Scatter plot with regression line
axes[0].scatter(CA, NY, alpha=0.6, edgecolors="grey")
z = np.polyfit(CA, NY, 1)
axes[0].plot(np.sort(CA), np.polyval(z, np.sort(CA)),
             color="red", linewidth=2)
axes[0].set_xlabel("CA Price (\\$)")
axes[0].set_ylabel("NY Price (\\$)")
axes[0].set_title(f"Scatter (r = {r:.3f})")

# Panel 2: Deviation products
products = x_dev * y_dev
colours = ["steelblue" if p > 0 else "salmon" for p in products]
axes[1].bar(range(WEEKS), products, color=colours, edgecolor="white")
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_xlabel("Week")
axes[1].set_ylabel("$(x - \\bar{x})(y - \\bar{y})$")
axes[1].set_title("Deviation Products")

# Panel 3: Time series
weeks = np.arange(WEEKS)
axes[2].plot(weeks, CA, label="CA", marker="o", markersize=3)
axes[2].plot(weeks, NY, label="NY", marker="s", markersize=3)
axes[2].set_xlabel("Week")
axes[2].set_ylabel("Price (\\$)")
axes[2].set_title("Common Trend")
axes[2].legend()

plt.tight_layout()
plt.show()
```

---

## Interpretation

The strong positive correlation between CA and NY prices ($r$ near 1) arises entirely from the shared downward trend -- the confounding variable. Neither state's price *causes* the other. This is a textbook illustration of the principle that **correlation does not imply causation**.

The deviation product bar chart shows that nearly all products are positive (blue), which is why the covariance -- and hence $r$ -- is strongly positive. In the time series panel, both series decline together, driven by the common trend rather than by any causal link between them.

---

## Exercises

**Exercise 1.**
Compute the covariance and Pearson $r$ by hand for the dataset $(1, 2), (2, 4), (3, 5), (4, 4), (5, 5)$. Show all intermediate steps including the deviation products.

??? success "Solution to Exercise 1"

    The sample means are $\bar{x} = 3$ and $\bar{y} = 4$.

    | $i$ | $x_i$ | $y_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | $(x_i - \bar{x})(y_i - \bar{y})$ |
    |:---:|:---:|:---:|:---:|:---:|:---:|
    | 1 | 1 | 2 | $-2$ | $-2$ | 4 |
    | 2 | 2 | 4 | $-1$ | 0 | 0 |
    | 3 | 3 | 5 | 0 | 1 | 0 |
    | 4 | 4 | 4 | 1 | 0 | 0 |
    | 5 | 5 | 5 | 2 | 1 | 2 |

    $$
    \text{Cov}(X, Y) = \frac{4 + 0 + 0 + 0 + 2}{5 - 1} = \frac{6}{4} = 1.5
    $$

    $$
    s_X = \sqrt{\frac{4 + 1 + 0 + 1 + 4}{4}} = \sqrt{2.5} \approx 1.5811
    $$

    $$
    s_Y = \sqrt{\frac{4 + 0 + 1 + 0 + 1}{4}} = \sqrt{1.5} \approx 1.2247
    $$

    $$
    r = \frac{1.5}{1.5811 \times 1.2247} \approx \frac{1.5}{1.9365} \approx 0.7746
    $$

    $\square$

---

**Exercise 2.**
Prove that the sample covariance $\frac{1}{n-1}\sum(x_i - \bar{x})(y_i - \bar{y})$ is an unbiased estimator of the population covariance $\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$.

??? success "Solution to Exercise 2"

    Let $(X_1, Y_1), \ldots, (X_n, Y_n)$ be i.i.d. with $\mathbb{E}[X] = \mu_X$, $\mathbb{E}[Y] = \mu_Y$, and $\text{Cov}(X, Y) = \sigma_{XY}$.

    Expand the sum:

    $$
    \sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y}) = \sum_{i=1}^n X_i Y_i - n\bar{X}\bar{Y}
    $$

    Taking expectations:

    $$
    \mathbb{E}\!\left[\sum_{i=1}^n X_i Y_i\right] = n(\sigma_{XY} + \mu_X \mu_Y)
    $$

    $$
    \mathbb{E}[n\bar{X}\bar{Y}] = n\!\left(\frac{\sigma_{XY}}{n} + \mu_X \mu_Y\right) = \sigma_{XY} + n\mu_X \mu_Y
    $$

    Therefore:

    $$
    \mathbb{E}\!\left[\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})\right] = n\sigma_{XY} + n\mu_X\mu_Y - \sigma_{XY} - n\mu_X\mu_Y = (n-1)\sigma_{XY}
    $$

    Dividing by $n - 1$:

    $$
    \mathbb{E}\!\left[\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})\right] = \sigma_{XY}
    $$

    This confirms that the sample covariance with the $n - 1$ denominator is unbiased. $\square$

---

**Exercise 3.**
Show that $\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\,\mathbb{E}[Y]$. Use this identity to prove that if $X$ and $Y$ are independent, then $\text{Cov}(X, Y) = 0$.

??? success "Solution to Exercise 3"

    Starting from the definition:

    $$
    \text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]
    $$

    Expanding:

    $$
    = \mathbb{E}[XY - \mu_Y X - \mu_X Y + \mu_X \mu_Y]
    $$

    $$
    = \mathbb{E}[XY] - \mu_Y \mathbb{E}[X] - \mu_X \mathbb{E}[Y] + \mu_X \mu_Y
    $$

    $$
    = \mathbb{E}[XY] - \mu_X \mu_Y - \mu_X \mu_Y + \mu_X \mu_Y = \mathbb{E}[XY] - \mathbb{E}[X]\,\mathbb{E}[Y]
    $$

    If $X \perp Y$, then $\mathbb{E}[XY] = \mathbb{E}[X]\,\mathbb{E}[Y]$ (by independence), so:

    $$
    \text{Cov}(X, Y) = \mathbb{E}[X]\,\mathbb{E}[Y] - \mathbb{E}[X]\,\mathbb{E}[Y] = 0
    $$

    Note: the converse is false in general. Zero covariance does not imply independence (e.g., $X \sim \mathcal{N}(0,1)$ and $Y = X^2$ have $\text{Cov}(X, Y) = 0$ but are clearly dependent). $\square$

---

**Exercise 4.**
Modify the simulation so that CA and NY prices have independent trends (no shared component). Recompute the covariance and $r$. How does removing the common trend affect the results?

??? success "Solution to Exercise 4"

    ```python
    import numpy as np

    np.random.seed(42)
    WEEKS = 48
    trend_CA = np.linspace(0, -12, WEEKS)
    trend_NY = np.linspace(0, -8, WEEKS)  # independent trend

    # Add independent noise to make trends truly separate
    CA = 248.0 + trend_CA + np.random.normal(0, 3, WEEKS)
    NY = 350.0 + trend_NY + np.random.normal(0, 3, WEEKS)

    cov = np.cov(CA, NY)[0, 1]
    r = np.corrcoef(CA, NY)[0, 1]
    print(f"Covariance = {cov:.4f}")
    print(f"Pearson r  = {r:.4f}")
    ```

    With larger independent noise and separate trends, the correlation drops significantly. The original high correlation was driven by the *shared* trend. When trends are independent, the only source of covariation is random noise, and the correlation is expected to be near zero (though not exactly zero in a finite sample). This further confirms that the original correlation was a confounding artifact. $\square$

---

**Exercise 5.**
Prove the bilinearity property of covariance: for constants $a, b, c, d$ and random variables $X, Y, W$,

$$
\text{Cov}(aX + bY,\; cW + d) = ac\,\text{Cov}(X, W) + bc\,\text{Cov}(Y, W)
$$

??? success "Solution to Exercise 5"

    Using the identity $\text{Cov}(U, V) = \mathbb{E}[UV] - \mathbb{E}[U]\,\mathbb{E}[V]$:

    $$
    \text{Cov}(aX + bY,\; cW + d)
    $$

    $$
    = \mathbb{E}[(aX + bY)(cW + d)] - \mathbb{E}[aX + bY]\,\mathbb{E}[cW + d]
    $$

    Expanding the first term:

    $$
    = ac\,\mathbb{E}[XW] + ad\,\mathbb{E}[X] + bc\,\mathbb{E}[YW] + bd\,\mathbb{E}[Y]
    $$

    Expanding the second term:

    $$

    - (a\,\mathbb{E}[X] + b\,\mathbb{E}[Y])(c\,\mathbb{E}[W] + d)
    $$

    $$
    = -ac\,\mathbb{E}[X]\mathbb{E}[W] - ad\,\mathbb{E}[X] - bc\,\mathbb{E}[Y]\mathbb{E}[W] - bd\,\mathbb{E}[Y]
    $$

    Combining:

    $$
    = ac(\mathbb{E}[XW] - \mathbb{E}[X]\mathbb{E}[W]) + bc(\mathbb{E}[YW] - \mathbb{E}[Y]\mathbb{E}[W])
    $$

    $$
    = ac\,\text{Cov}(X, W) + bc\,\text{Cov}(Y, W)
    $$

    Note that the constant $d$ drops out entirely, reflecting the fact that adding a constant to a random variable does not change its covariance with anything. $\square$
