# Confidence Intervals for Simple OLS Estimators

## Overview

Given the sampling distributions derived in the [previous section](sampling_dist_simple.md), we can construct confidence intervals for the slope, the expected response, and individual predictions in simple linear regression. Each confidence interval takes the standard form: **point estimate $\pm$ critical value $\times$ standard error**.

---

## Confidence Interval Formulas

### Slope

$$
\hat{\beta}_1 \pm t_{n-2}(0.975)\; s\sqrt{\frac{1}{\sum_{i=1}^n(x_i-\bar{x})^2}}
$$

This interval quantifies the uncertainty in the estimated rate of change of $y$ with respect to $x$. If the interval excludes zero, we have evidence at the 5% significance level that $x$ has a linear effect on $y$.

### Response Expectation (Mean Response at $x_0$)

$$
(\hat{\beta}_0+\hat{\beta}_1 x_0) \pm t_{n-2}(0.975)\; s\sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n(x_i-\bar{x})^2}}
$$

This confidence interval captures the true mean of $y$ at a specific value $x_0$. The interval is narrowest when $x_0 = \bar{x}$ and widens as $x_0$ moves away from the center of the data, producing the characteristic "bowtie" shape when plotted across all $x_0$ values.

### Response (Prediction Interval at $x_0$)

$$
(\hat{\beta}_0+\hat{\beta}_1 x_0) \pm t_{n-2}(0.975)\; s\sqrt{1+\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n(x_i-\bar{x})^2}}
$$

This prediction interval captures where a **new individual observation** at $x_0$ is likely to fall. It is always wider than the confidence interval for the mean response because it includes the irreducible noise term $\sigma^2$ (the leading 1 under the square root).

### Key Distinction

The confidence interval for the mean response and the prediction interval for an individual response share the same center $\hat{\beta}_0 + \hat{\beta}_1 x_0$, but differ in width. The mean response interval shrinks toward zero width as $n \to \infty$ (estimation uncertainty vanishes), while the prediction interval converges to $\hat{y}_0 \pm t \cdot s$ (irreducible noise remains).

!!! info "Reference"
    [Khan Academy: Inference for Slope](https://www.khanacademy.org/math/ap-statistics/inference-slope-linear-regression/inference-slope/v/intro-inference-slope)

---

## Example: Study Hours and Caffeine Consumption

### Problem

Musa investigates the correlation between studying hours and caffeine consumption among 20 students at his school. After performing a least-squares regression, he obtains the following output:

|  | Coef | SE Coef | T | P |
|:---|---:|---:|---:|---:|
| Constant | 2.544 | 0.134 | 18.955 | 0.000 |
| Caffeine | 0.164 | 0.057 | 2.862 | 0.010 |

$S = 1.532$, $R^2 = 60.0\%$

**Task**: Determine the 95% confidence interval for the slope of the least-squares regression line.

### Solution

From the regression output we extract:

- $\hat{\beta}_1 = 0.164$ (the estimated slope)
- $\text{SE}(\hat{\beta}_1) = 0.057$ (the standard error of the slope)
- $n = 20$, so $\text{df} = n - 2 = 18$

The 95% confidence interval is:

$$
\hat{\beta}_1 \pm t_{18}(0.975) \times \text{SE}(\hat{\beta}_1) = 0.164 \pm 2.1009 \times 0.057
$$

$$
= 0.164 \pm 0.1198 = (0.0442,\; 0.2838)
$$

**Interpretation**: We are 95% confident that the true slope relating caffeine consumption to study hours lies between 0.044 and 0.284. Since this interval does not contain zero, there is statistically significant evidence of a positive linear relationship.

!!! info "Reference"
    [Khan Academy: Confidence Interval for Slope](https://www.khanacademy.org/math/ap-statistics/inference-slope-linear-regression/inference-slope/v/confidence-interval-slope)

### Python Implementation

```python
from scipy import stats

def main():
    beta_1_hat = 0.164
    n = 20
    df = n - 2
    confidence_level = 0.95
    alpha = 1 - confidence_level
    t_star = stats.t(df).ppf(1 - alpha / 2)
    standard_error = 0.057
    margin_of_error = t_star * standard_error
    print(f"{confidence_level:.0%} confidence interval of the slope")
    print(f"{beta_1_hat:.4f} ± {margin_of_error:.4f}")

if __name__ == "__main__":
    main()
```

**Output**:
```
95% confidence interval of the slope
0.1640 ± 0.1198
```

---

## Visualization: Confidence and Prediction Bands

The following example generates synthetic regression data and plots both the 95% confidence interval for the mean response (inner band) and the 95% prediction interval for individual observations (outer band).

### Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
```

### Data Generation

```python
def generate_data(n, sigma, seed=0):
    """
    Generate synthetic linear regression data.

    Parameters
    ----------
    n : int
        Number of observations.
    sigma : float
        Standard deviation of the noise term.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    x, y : ndarray of shape (n, 1)
        Predictor and response arrays.
    """
    np.random.seed(seed)
    x = np.random.randn(n, 1)
    y = 1 + 2 * x + sigma * np.random.randn(n, 1)
    return x, y
```

### Regression Estimation

```python
def estimate_regression_line(x, y):
    """
    Estimate slope and intercept via the correlation formula.

    Returns
    -------
    y_hat : ndarray
        Fitted values.
    beta_hat : float
        Estimated slope.
    y_bar, x_bar : float
        Sample means.
    """
    x_bar = x.mean()
    y_bar = y.mean()
    s_x = x.std(ddof=1)
    s_y = y.std(ddof=1)
    r = np.corrcoef(np.concatenate([x, y], axis=1), rowvar=False)[1, 0]
    beta_hat = r * s_y / s_x
    y_hat = beta_hat * (x - x_bar) + y_bar
    return y_hat, beta_hat, y_bar, x_bar
```

### Residual Variance

```python
def calculate_residual_variance(y, y_hat, n):
    """
    Compute the unbiased residual variance s² and standard deviation s.
    """
    s_square = np.sum((y - y_hat) ** 2) / (n - 2)
    s = np.sqrt(s_square)
    return s_square, s
```

### Confidence and Prediction Intervals

```python
def confidence_intervals(x, y_hat, beta_hat, x_bar, y_bar, n, s):
    """
    Compute 95% confidence intervals for E[y|x] and for y|x.

    Returns
    -------
    x0 : ndarray
        Grid of x values for plotting.
    lower, upper : ndarray
        Bounds for the mean response interval.
    lower2, upper2 : ndarray
        Bounds for the prediction interval.
    """
    x0 = np.linspace(x.min(), x.max(), 20)
    y0_hat = beta_hat * (x0 - x_bar) + y_bar
    t_val = stats.t(n - 2).ppf(0.975)

    # Confidence interval for E[y | x = x0]
    margin = t_val * s * np.sqrt(
        (1 / n) + (x0 - x_bar) ** 2 / np.sum((x - x_bar) ** 2)
    )
    lower = y0_hat - margin
    upper = y0_hat + margin

    # Prediction interval for y | x = x0
    margin2 = t_val * s * np.sqrt(
        1 + (1 / n) + (x0 - x_bar) ** 2 / np.sum((x - x_bar) ** 2)
    )
    lower2 = y0_hat - margin2
    upper2 = y0_hat + margin2

    return x0, lower, upper, lower2, upper2
```

### Plotting

```python
def plot_intervals(x, y, y_hat, x0, lower, upper, lower2, upper2):
    """Plot confidence and prediction bands side by side."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))

    # Left: Confidence interval for E[y]
    ax0.plot(x, y, 'o', alpha=0.5)
    ax0.plot(x, y_hat, '--b', label='Fitted line')
    ax0.plot(x0, upper, '--r', label='95% CI bounds')
    ax0.plot(x0, lower, '--r')
    ax0.set_title('95% Confidence Interval for $E[y]$')
    ax0.legend()

    # Right: Prediction interval for y
    ax1.plot(x, y, 'o', alpha=0.5)
    ax1.plot(x, y_hat, '--b', label='Fitted line')
    ax1.plot(x0, upper2, '--r', label='95% PI bounds')
    ax1.plot(x0, lower2, '--r')
    ax1.set_title('95% Prediction Interval for $y$')
    ax1.legend()

    plt.tight_layout()
    plt.show()
```

### Full Example

```python
# Parameters
n = 100
sigma = 3

# Generate data (true model: y = 1 + 2x + noise)
x, y = generate_data(n, sigma)

# Fit regression
y_hat, beta_hat, y_bar, x_bar = estimate_regression_line(x, y)

# Residual variance
s_square, s = calculate_residual_variance(y, y_hat, n)
print(f"True σ²: {sigma**2}")
print(f"Estimated s²: {s_square:.4f}")

# Compute intervals
x0, lower, upper, lower2, upper2 = confidence_intervals(
    x, y_hat, beta_hat, x_bar, y_bar, n, s
)

# Plot
plot_intervals(x, y, y_hat, x0, lower, upper, lower2, upper2)
```

The left panel shows the confidence band for the mean response—notice its characteristic "bowtie" shape, narrowest at $\bar{x}$. The right panel shows the wider prediction band that accounts for individual observation variability.
