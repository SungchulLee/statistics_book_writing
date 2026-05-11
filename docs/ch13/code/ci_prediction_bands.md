# Confidence Interval and Prediction Bands

## Overview

This page explains and implements confidence intervals for the mean response $E[y \mid x]$ and prediction intervals for a new observation $y \mid x$ in simple linear regression. We generate synthetic data from a known model, fit OLS, and visualize both bands to illustrate how uncertainty about the regression line differs from uncertainty about individual predictions.

## Mathematical Background

Consider the simple linear regression model

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \qquad \varepsilon_i \overset{\text{iid}}{\sim} N(0, \sigma^2).
$$

Given a new predictor value $x_0$, the fitted value is $\hat{y}_0 = \hat{\beta}_0 + \hat{\beta}_1 x_0$. There are two types of intervals:

**Confidence interval for $E[y \mid x_0]$** (mean response):

$$
\hat{y}_0 \pm t^*_{n-2} \cdot s \sqrt{\frac{1}{n} + \frac{(x_0 - \bar{x})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}}
$$

**Prediction interval for a new $y$ at $x_0$**:

$$
\hat{y}_0 \pm t^*_{n-2} \cdot s \sqrt{1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{\sum_{i=1}^n (x_i - \bar{x})^2}}
$$

The prediction interval is always wider because it accounts for both the uncertainty in estimating the mean and the irreducible noise $\sigma^2$.

## Code

### Data Generation

```python
import numpy as np

def generate_data(n, sigma, seed=0):
    np.random.seed(seed)
    x = np.random.randn(n, 1)
    y = 1 + 2 * x + sigma * np.random.randn(n, 1)
    return x, y
```

### Regression Line Estimation

```python
def estimate_regression_line(x, y):
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
    s_square = np.sum((y - y_hat) ** 2) / (n - 2)
    s = np.sqrt(s_square)
    return s_square, s
```

### Confidence and Prediction Intervals

```python
from scipy import stats

def confidence_intervals(x, y_hat, beta_hat, x_bar, y_bar, n, s):
    x0 = np.linspace(x.min(), x.max(), 20)
    y0_hat = beta_hat * (x0 - x_bar) + y_bar
    t_val = stats.t(n - 2).ppf(0.975)
    ss_x = np.sum((x - x_bar) ** 2)

    # CI for E[y | x = x0]
    margin = t_val * s * np.sqrt((1 / n) + (x0 - x_bar) ** 2 / ss_x)
    lower = y0_hat - margin
    upper = y0_hat + margin

    # PI for y | x = x0
    margin2 = t_val * s * np.sqrt(1 + (1 / n) + (x0 - x_bar) ** 2 / ss_x)
    lower2 = y0_hat - margin2
    upper2 = y0_hat + margin2

    return x0, lower, upper, lower2, upper2
```

## Interpretation

- **Confidence bands** (CI) quantify uncertainty about where the true regression line lies. They are narrowest at $x = \bar{x}$ and widen as $x_0$ moves away from the center of the data.
- **Prediction bands** (PI) quantify uncertainty about where a single new observation will fall. They include the additional $+1$ under the square root, making them always wider than the confidence bands.
- As $n \to \infty$, the confidence band shrinks to zero width (the line is estimated perfectly), but the prediction band converges to $\hat{y}_0 \pm t^* \cdot s$, reflecting irreducible noise.
- The "bow-tie" shape of both bands reflects the principle of leverage: predictions are most reliable near the center of the observed predictor values.

## Exercises

**Exercise 1.** Using the synthetic data with $n = 100$ and $\sigma = 3$, compute the 90% confidence interval for $E[y \mid x_0 = 0]$ and compare it with the 95% interval.

??? success "Solution to Exercise 1"

    At $x_0 = 0$, which is approximately $\bar{x}$ for standard normal predictors:

    ```python
    t_90 = stats.t(n - 2).ppf(0.95)  # ~1.66
    t_95 = stats.t(n - 2).ppf(0.975) # ~1.98

    margin_90 = t_90 * s * np.sqrt(1 / n)
    margin_95 = t_95 * s * np.sqrt(1 / n)
    ```

    The 90% interval is narrower by the ratio $t_{0.95}/t_{0.975} \approx 1.66/1.98 \approx 0.84$. $\square$

---

**Exercise 2.** Explain algebraically why the prediction interval is always wider than the confidence interval for any $x_0$.

??? success "Solution to Exercise 2"

    The squared half-width of the CI is proportional to

    $$
    \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{xx}},
    $$

    while for the PI it is

    $$
    1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{xx}}.
    $$

    The PI adds a term of $1$ (representing $\mathrm{Var}(\varepsilon_{\text{new}})/s^2 = 1$ after standardization), so the quantity under the square root is strictly larger for the PI. Since the critical value and $s$ are the same, the PI is always wider. $\square$

---

**Exercise 3.** Modify the code to plot the bands for a model with $n = 30$ and $\sigma = 1$. How do the band widths compare to the original $n = 100$, $\sigma = 3$ case?

??? success "Solution to Exercise 3"

    ```python
    x30, y30 = generate_data(30, 1)
    y_hat30, beta30, ybar30, xbar30 = estimate_regression_line(x30, y30)
    _, s30 = calculate_residual_variance(y30, y_hat30, 30)
    ```

    The CI width depends on $s/\sqrt{n}$. With $n=30$ and $\sigma=1$, $s \approx 1$ and $s/\sqrt{30} \approx 0.18$, versus the original $s \approx 3$ and $s/\sqrt{100} = 0.3$. The new CI is narrower. The PI width is dominated by $s$, so with $\sigma=1$ the PI is much narrower than with $\sigma=3$. $\square$

---

**Exercise 4.** Derive the variance of the prediction error $\hat{y}_0 - y_{\text{new}}$ where $y_{\text{new}} = \beta_0 + \beta_1 x_0 + \varepsilon_{\text{new}}$ and $\varepsilon_{\text{new}}$ is independent of the training data.

??? success "Solution to Exercise 4"

    The prediction error is $\hat{y}_0 - y_{\text{new}} = (\hat{y}_0 - E[y \mid x_0]) - \varepsilon_{\text{new}}$. Since $\hat{y}_0$ depends only on training data and $\varepsilon_{\text{new}}$ is independent:

    $$
    \mathrm{Var}(\hat{y}_0 - y_{\text{new}}) = \mathrm{Var}(\hat{y}_0) + \mathrm{Var}(\varepsilon_{\text{new}}) = \sigma^2\!\left(\frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{xx}}\right) + \sigma^2.
    $$

    Factoring out $\sigma^2$:

    $$
    \mathrm{Var}(\hat{y}_0 - y_{\text{new}}) = \sigma^2\!\left(1 + \frac{1}{n} + \frac{(x_0 - \bar{x})^2}{S_{xx}}\right).
    $$

    This is exactly the expression under the square root in the prediction interval formula (after replacing $\sigma$ with $s$). $\square$

---

**Exercise 5.** Show that at $x_0 = \bar{x}$, the confidence interval for the mean simplifies to $\bar{y} \pm t^*_{n-2} \cdot s / \sqrt{n}$. Compare this to the confidence interval for a population mean from introductory statistics.

??? success "Solution to Exercise 5"

    At $x_0 = \bar{x}$, the term $(x_0 - \bar{x})^2/S_{xx} = 0$, so the CI becomes

    $$
    \hat{y}_0 \pm t^*_{n-2} \cdot s \sqrt{\frac{1}{n}} = \bar{y} \pm \frac{t^*_{n-2} \cdot s}{\sqrt{n}},
    $$

    since $\hat{y}_0 = \hat{\beta}_0 + \hat{\beta}_1 \bar{x} = \bar{y}$. This has the same form as the CI for a population mean $\bar{y} \pm t^*_{n-1} \cdot s/\sqrt{n}$, except the degrees of freedom are $n-2$ rather than $n-1$ because we estimate two parameters in regression. $\square$
