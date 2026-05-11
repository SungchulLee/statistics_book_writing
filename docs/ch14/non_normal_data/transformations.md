# Transformations to Achieve Normality


When data deviates significantly from normality, certain statistical methods that rely on normality assumptions may no longer be appropriate. One common approach is to apply a transformation to make the data more normal.

## Common Transformations

Popular transformations include:

**Log Transformation**: Suitable for positively skewed data.

$$
X' = \log(X)
$$

**Square Root Transformation**: Also used for right-skewed data, particularly when there are small values.

$$
X' = \sqrt{X}
$$

**Box-Cox Transformation**: A more flexible transformation that finds an optimal power parameter $\lambda$ to transform the data.

$$
X' = \frac{X^\lambda - 1}{\lambda}, \quad \lambda \neq 0
$$

## Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox

# Generate positively skewed data
skewed_data = np.random.exponential(scale=2, size=1000)

# Log transformation
log_transformed_data = np.log(skewed_data + 1)  # Adding 1 to avoid log(0)

# Box-Cox transformation
boxcox_transformed_data, best_lambda = boxcox(skewed_data + 1)

# Plot the original and transformed data
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
axs[0].hist(skewed_data, bins=30)
axs[0].set_title('Original Data')

axs[1].hist(log_transformed_data, bins=30)
axs[1].set_title('Log Transformed Data')

axs[2].hist(boxcox_transformed_data, bins=30)
axs[2].set_title(f'Box-Cox Transformed Data (λ={best_lambda:.2f})')

plt.show()
```

Both log and Box-Cox transformations are applied to skewed data. These transformations often make data more symmetric and closer to normality, making it suitable for parametric tests.


## Exercises

**Exercise 1.**
A variable $Y$ has a right-skewed distribution with all positive values. Apply the log transformation $Y' = \log(Y)$ and explain why this often reduces skewness.

??? success "Solution to Exercise 1"
    The log function is concave: it compresses large values more than small values. For right-skewed data, the long right tail (large values) is compressed substantially while the left side (small values near zero) is stretched. This pulls the right tail inward toward the center, reducing the asymmetry.

    Formally, if $Y$ is log-normally distributed ($\log Y \sim N(\mu, \sigma^2)$), the log transformation produces perfectly normal data. Even for approximately log-normal data (common for incomes, stock prices, biological measurements), the log transformation substantially improves normality.

    Caveat: the log transformation is undefined for $Y \leq 0$. For data with zeros, use $\log(Y + c)$ for a small constant $c$.

---

**Exercise 2.**
The Box-Cox transformation family is $Y^{(\lambda)} = (Y^\lambda - 1)/\lambda$ for $\lambda \neq 0$ and $\log(Y)$ for $\lambda = 0$. What transformations do $\lambda = 1$, $\lambda = 0.5$, and $\lambda = -1$ correspond to?

??? success "Solution to Exercise 2"

    - $\lambda = 1$: $Y^{(1)} = (Y - 1)/1 = Y - 1$ (linear shift, essentially no transformation).
    - $\lambda = 0.5$: $Y^{(0.5)} = (\sqrt{Y} - 1)/0.5 = 2(\sqrt{Y} - 1)$ (square root transformation, up to linear rescaling).
    - $\lambda = 0$: $Y^{(0)} = \log(Y)$ (log transformation).
    - $\lambda = -1$: $Y^{(-1)} = (1/Y - 1)/(-1) = 1 - 1/Y$ (reciprocal transformation, up to linear rescaling).

    The optimal $\lambda$ is chosen by maximum likelihood: find the $\lambda$ that makes the transformed data most closely normal. This is typically done using `scipy.stats.boxcox` in Python.

---

**Exercise 3.**
After applying a log transformation to right-skewed data, the regression coefficients have a different interpretation. Explain how to interpret $\hat{\beta}_1$ in the model $\log(Y) = \beta_0 + \beta_1 X + \varepsilon$.

??? success "Solution to Exercise 3"
    In the log-linear model, exponentiating gives $Y = e^{\beta_0 + \beta_1 X + \varepsilon}$. A one-unit increase in $X$ multiplies the expected value of $Y$ by $e^{\beta_1}$:

    $$
    \frac{E[Y \mid X+1]}{E[Y \mid X]} \approx e^{\beta_1} \approx 1 + \beta_1 \text{ (for small } \beta_1\text{)}
    $$

    So $\beta_1 \approx $ the proportional (percentage) change in $Y$ per unit change in $X$. For example, $\beta_1 = 0.05$ means a 1-unit increase in $X$ is associated with approximately a 5% increase in $Y$.

    This multiplicative interpretation is natural for many applications (incomes, prices, biological growth).

---

**Exercise 4.**
List two situations where transforming the data is not recommended even if normality is violated.

??? success "Solution to Exercise 4"

    1. **When the original scale is scientifically meaningful:** If the research question is about differences in the original units (e.g., "Does the treatment reduce blood pressure by at least 10 mmHg?"), transforming the data changes the interpretation. A 10-unit difference on the log scale is not the same as 10 mmHg. The analyst should use methods valid on the original scale (bootstrap, robust methods).

    2. **When the transformation creates interpretation difficulties:** Log-transforming a variable with many zeros (e.g., medical costs, insurance claims) requires ad hoc adjustments ($\log(Y+1)$), and back-transformed estimates are biased (the mean of $\log Y$ is not the log of the mean of $Y$). In these cases, a generalized linear model (Gamma GLM, Tweedie regression) is often preferable.
