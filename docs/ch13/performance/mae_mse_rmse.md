# Mean Absolute, Mean Squared, and Root Mean Squared Error

R-squared tells us the proportion of variance explained, but it does not tell us how large the prediction errors are in the original units of $Y$. Error metrics such as MAE, MSE, and RMSE quantify the magnitude of prediction errors directly, providing complementary information about model performance.

---

## 1. Residuals

All error metrics are built from the **residuals**, defined as the difference between the observed and fitted values:

$$
e_i = y_i - \hat{y}_i, \quad i = 1, \ldots, n
$$

A good model produces residuals that are small in magnitude and show no systematic pattern. The metrics below summarize these residuals into a single number.

---

## 2. Mean Absolute Error

The **mean absolute error (MAE)** is the average of the absolute residuals:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

MAE is measured in the same units as $Y$, making it directly interpretable. If $\text{MAE} = 3.2$ and $Y$ is measured in dollars, then the average prediction is off by \$3.20.

### Properties

- **Robust to outliers**: MAE treats all errors linearly, so a single large error does not disproportionately inflate the metric.
- **Not differentiable at zero**: The absolute value function has a corner at zero, which complicates optimization. This is why least squares (which minimizes MSE) is more common in parameter estimation.
- **Median connection**: The value that minimizes $\sum |y_i - c|$ over $c$ is the sample median, just as the value that minimizes $\sum (y_i - c)^2$ is the sample mean.

---

## 3. Mean Squared Error

The **mean squared error (MSE)** is the average of the squared residuals:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

MSE is the most common loss function in regression because it leads to closed-form solutions through the normal equations. However, its units are the square of the original units of $Y$, which makes direct interpretation difficult.

### Properties

- **Sensitive to outliers**: Squaring the residuals amplifies large errors. A single observation with $|e_i| = 10$ contributes 100 to the sum, while ten observations with $|e_i| = 1$ contribute only 10 total.
- **Differentiable**: MSE is smooth everywhere, making it well-suited for gradient-based optimization.
- **Decomposition**: For a random variable framework, MSE decomposes into bias squared plus variance: $\text{MSE}(\hat{\theta}) = \text{Bias}^2(\hat{\theta}) + \text{Var}(\hat{\theta})$.

!!! note "MSE vs SSE"
    MSE and SSE differ only by a scaling factor: $\text{MSE} = \text{SSE} / n$. In some texts, the denominator is $n - p - 1$ instead of $n$ to produce an unbiased estimate of $\sigma^2$. This unbiased version is often denoted $s^2$ or $\hat{\sigma}^2$.

---

## 4. Root Mean Squared Error

The **root mean squared error (RMSE)** is the square root of MSE:

$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

RMSE restores the original units of $Y$, combining the mathematical convenience of MSE with the interpretability of MAE.

### Properties

- **Same units as $Y$**: Like MAE, RMSE is directly interpretable in the scale of the response.
- **Always at least as large as MAE**: By Jensen's inequality (since square root is concave), $\text{RMSE} \geq \text{MAE}$, with equality only when all residuals have the same absolute value.
- **More sensitive to outliers than MAE**: RMSE inherits the outlier sensitivity of MSE because large squared errors contribute more before the square root is taken.

---

## 5. Comparing MAE and RMSE

The relationship between MAE and RMSE reveals information about the distribution of errors:

$$
\text{MAE} \leq \text{RMSE} \leq \sqrt{n} \cdot \text{MAE}
$$

The lower bound is achieved when all absolute residuals are equal. The upper bound is achieved when a single residual accounts for all the error. When RMSE is much larger than MAE, it indicates that the error distribution has large outliers or heavy tails.

| Property | MAE | RMSE |
|---|---|---|
| Units | Same as $Y$ | Same as $Y$ |
| Outlier sensitivity | Low | High |
| Differentiability | No (at zero) | Yes |
| Optimization | Leads to median regression | Leads to OLS regression |
| Interpretation | Average absolute error | Standard deviation of errors |

!!! tip "When to prefer MAE over RMSE"
    Use MAE when outliers are expected and should not dominate the evaluation (e.g., real estate price prediction where a few luxury homes create extreme errors). Use RMSE when large errors are particularly undesirable and the model should be penalized more heavily for them (e.g., predicting structural loads where underestimation is dangerous).

---

## 6. Numerical Example

Consider a model with $n = 5$ observations:

| $i$ | $y_i$ | $\hat{y}_i$ | $e_i$ | $|e_i|$ | $e_i^2$ |
|-----|--------|--------------|--------|----------|---------|
| 1   | 10     | 11           | $-1$   | 1        | 1       |
| 2   | 20     | 19           | 1      | 1        | 1       |
| 3   | 30     | 28           | 2      | 2        | 4       |
| 4   | 40     | 41           | $-1$   | 1        | 1       |
| 5   | 50     | 44           | 6      | 6        | 36      |

Computing each metric:

$$
\text{MAE} = \frac{1 + 1 + 2 + 1 + 6}{5} = \frac{11}{5} = 2.2
$$

$$
\text{MSE} = \frac{1 + 1 + 4 + 1 + 36}{5} = \frac{43}{5} = 8.6
$$

$$
\text{RMSE} = \sqrt{8.6} \approx 2.93
$$

Notice that RMSE ($2.93$) is substantially larger than MAE ($2.2$). This gap is driven by the single large error $e_5 = 6$, which contributes $36/43 \approx 84\%$ of MSE but only $6/11 \approx 55\%$ of the total absolute error. This example illustrates how RMSE disproportionately reflects the influence of outlying residuals.

## Exercises

**Exercise 1.**
Given actual values $y = (3, 5, 2, 8)$ and predictions $\hat{y} = (2.5, 5.5, 1.5, 7)$, compute the MAE, MSE, and RMSE.

??? success "Solution to Exercise 1"
    Errors: $e = (0.5, -0.5, 0.5, 1.0)$.

    $$
    \text{MAE} = \frac{1}{4}(|0.5| + |-0.5| + |0.5| + |1.0|) = \frac{2.5}{4} = 0.625
    $$

    $$
    \text{MSE} = \frac{1}{4}(0.25 + 0.25 + 0.25 + 1.0) = \frac{1.75}{4} = 0.4375
    $$

    $$
    \text{RMSE} = \sqrt{0.4375} \approx 0.6614
    $$

---

**Exercise 2.**
Explain why MSE penalizes large errors more heavily than MAE. Give a practical scenario where MAE is preferred.

??? success "Solution to Exercise 2"
    MSE squares each error, so an error of 10 contributes $10^2 = 100$ to the sum, while an error of 1 contributes only $1$. This means a single large error dominates the MSE. MAE uses absolute values, so the same errors contribute 10 and 1 respectively -- a 10:1 ratio instead of 100:1.

    **MAE is preferred when:** outliers are expected and should not disproportionately influence model evaluation. For example, in real estate price prediction, a few luxury homes with large prediction errors should not dominate the assessment of model quality. MAE evaluates the "typical" error magnitude, while MSE evaluates the "worst-case-penalized" error.

---

**Exercise 3.**
Show that the value $c$ that minimizes $\sum(y_i - c)^2$ is the mean $\bar{y}$, while the value that minimizes $\sum|y_i - c|$ is the median.

??? success "Solution to Exercise 3"
    **MSE minimizer (mean):** Differentiate $f(c) = \sum(y_i - c)^2$ with respect to $c$:

    $$
    f'(c) = -2\sum(y_i - c) = -2(n\bar{y} - nc) = 0 \implies c = \bar{y}
    $$

    **MAE minimizer (median):** The function $g(c) = \sum|y_i - c|$ is piecewise linear and convex. Its derivative is $g'(c) = -\#\{y_i > c\} + \#\{y_i < c\}$. Setting $g'(c) = 0$ requires equal numbers of observations above and below $c$, which defines the median.

    This connection explains why models optimized for MSE (e.g., OLS) predict the conditional mean, while models optimized for MAE (e.g., quantile regression at $\tau = 0.5$) predict the conditional median. $\square$

---

**Exercise 4.**
Why is RMSE preferred over MSE for reporting model performance? What is its unit?

??? success "Solution to Exercise 4"
    RMSE $= \sqrt{\text{MSE}}$ is preferred because it has the **same units** as the response variable $y$, making it directly interpretable. If $y$ is measured in dollars, RMSE is in dollars and represents the "typical" prediction error magnitude. MSE is in dollars-squared, which is unintuitive.

    RMSE also has a statistical interpretation: for a model with normally distributed errors, approximately 68% of predictions fall within $\pm$ RMSE of the actual value, and about 95% fall within $\pm 2 \cdot$ RMSE.

    However, RMSE shares MSE's sensitivity to outliers (since it is a monotonic transformation of MSE). For robustness, report both RMSE and MAE.
