# Sampling Distributions for General OLS Estimators

## Overview

This section extends the inferential results from simple linear regression to the **multiple linear regression** setting using matrix notation. We derive the sampling distributions of the OLS coefficient vector $\hat{\beta}$, the residual variance estimator $s^2$, and the $t$-statistic for testing individual coefficients.

---

## Setup: The Multiple Linear Regression Model

Consider the model in matrix form:

$$
\mathbf{y} = \mathbf{X}\beta + \varepsilon
$$

where:

- $\mathbf{y}$ is an $N \times 1$ vector of responses,
- $\mathbf{X}$ is an $N \times (p+1)$ design matrix (including an intercept column),
- $\beta$ is a $(p+1) \times 1$ vector of unknown coefficients,
- $\varepsilon \sim N(\mathbf{0}, \sigma^2 I_N)$ is the error vector.

The OLS estimator is:

$$
\hat{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

---

## 1. Sampling Distribution of $\hat{\beta}$

### Theorem

Under the assumptions of the linear regression model:

$$
\hat{\beta} \sim N\!\left(\beta,\; \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}\right)
$$

That is, $\hat{\beta}$ is normally distributed with:

- **Mean**: $E(\hat{\beta}) = \beta$ (unbiased),
- **Covariance**: $\text{Var}(\hat{\beta}) = \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}$.

### Proof

Substituting $\mathbf{y} = \mathbf{X}\beta + \varepsilon$ into the OLS formula:

$$
\hat{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T (\mathbf{X}\beta + \varepsilon) = \beta + (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \varepsilon
$$

**Unbiasedness**: Since $E(\varepsilon) = \mathbf{0}$:

$$
E(\hat{\beta}) = \beta + (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T E(\varepsilon) = \beta
$$

**Covariance**: Let $\mathbf{A} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$. Then $\hat{\beta} - \beta = \mathbf{A}\varepsilon$ and:

$$
\text{Var}(\hat{\beta}) = \mathbf{A}\,\text{Var}(\varepsilon)\,\mathbf{A}^T = \mathbf{A}(\sigma^2 I_N)\mathbf{A}^T = \sigma^2 \mathbf{A}\mathbf{A}^T
$$

Computing $\mathbf{A}\mathbf{A}^T$:

$$
\mathbf{A}\mathbf{A}^T = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1} = (\mathbf{X}^T\mathbf{X})^{-1}
$$

Therefore $\text{Var}(\hat{\beta}) = \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$.

**Normality**: Since $\hat{\beta} - \beta = \mathbf{A}\varepsilon$ is a linear transformation of the multivariate normal vector $\varepsilon$, $\hat{\beta}$ is itself multivariate normal. $\blacksquare$

### Implications

This result provides the foundation for all inference in multiple regression: confidence intervals for individual coefficients, joint confidence regions, hypothesis tests, and prediction intervals all follow from this distributional result.

---

## 2. Sampling Distribution of $s^2$

### Theorem

The residual variance estimator

$$
s^2 = \frac{1}{N - p - 1} \sum_{i=1}^N (y^{(i)} - \hat{y}^{(i)})^2
$$

has the sampling distribution:

$$
\frac{(N - p - 1)\,s^2}{\sigma^2} \sim \chi^2_{N-p-1}
$$

Equivalently:

$$
s^2 \sim \sigma^2 \frac{\chi^2_{N-p-1}}{N - p - 1}
$$

### Key Properties

**Unbiasedness**: $E(s^2) = \sigma^2$, so $s^2$ is an unbiased estimator of the error variance.

**Degrees of freedom**: The $N - p - 1$ degrees of freedom reflect the $N$ observations minus the $p + 1$ parameters estimated.

**Independence from $\hat{\beta}$**: Under the normality assumption, $s^2$ and $\hat{\beta}$ are statistically independent. This independence is essential for the validity of the $t$-tests.

### Proof

**Step 1: Express residuals via projection.** Define the hat matrix $\mathbf{P} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ and the residual-maker matrix $\mathbf{M} = I_N - \mathbf{P}$. The residual vector is:

$$
\mathbf{e} = \mathbf{y} - \mathbf{X}\hat{\beta} = (I_N - \mathbf{P})\mathbf{y} = \mathbf{M}\varepsilon
$$

where the last equality uses $\mathbf{M}\mathbf{X} = \mathbf{0}$.

**Step 2: Residual sum of squares as a quadratic form.**

$$
\text{RSS} = \mathbf{e}^T\mathbf{e} = \varepsilon^T \mathbf{M}^T \mathbf{M}\,\varepsilon = \varepsilon^T \mathbf{M}\,\varepsilon
$$

since $\mathbf{M}$ is symmetric and idempotent ($\mathbf{M}^2 = \mathbf{M}$).

**Step 3: Chi-squared distribution.** Since $\varepsilon \sim N(\mathbf{0}, \sigma^2 I_N)$ and $\mathbf{M}$ is a symmetric idempotent matrix with rank $\text{tr}(\mathbf{M}) = N - (p+1) = N - p - 1$:

$$
\frac{\varepsilon^T \mathbf{M}\,\varepsilon}{\sigma^2} = \frac{\text{RSS}}{\sigma^2} \sim \chi^2_{N-p-1}
$$

**Step 4: Conclusion.** Dividing by the degrees of freedom:

$$
s^2 = \frac{\text{RSS}}{N - p - 1} \sim \sigma^2\frac{\chi^2_{N-p-1}}{N - p - 1} \qquad \blacksquare
$$

---

## 3. T-Statistic for Individual Regression Coefficients

### Theorem

Under the null hypothesis $H_0: \beta_j = 0$, the test statistic:

$$
t_j = \frac{\hat{\beta}_j}{s\sqrt{v_j}} \sim t_{N-p-1}
$$

where $v_j = \left((\mathbf{X}^T\mathbf{X})^{-1}\right)_{jj}$ is the $j$-th diagonal element of $(\mathbf{X}^T\mathbf{X})^{-1}$.

### Interpretation

The $t$-statistic $t_j$ tests whether the $j$-th predictor contributes to the model **after accounting for all other predictors**. A large absolute value provides evidence that $\beta_j \neq 0$, i.e., that the $j$-th variable has a statistically significant partial effect on $y$.

The denominator $s\sqrt{v_j}$ is the **standard error** of $\hat{\beta}_j$:

$$
\text{SE}(\hat{\beta}_j) = s\sqrt{v_j}
$$

### Proof

**Step 1: Distribution of $\hat{\beta}_j$ under $H_0$.** From the sampling distribution of $\hat{\beta}$, each component satisfies:

$$
\hat{\beta}_j \sim N(\beta_j, \sigma^2 v_j)
$$

Under $H_0: \beta_j = 0$:

$$
\frac{\hat{\beta}_j}{\sigma\sqrt{v_j}} \sim N(0, 1)
$$

**Step 2: Independent chi-squared in the denominator.** From the distribution of $s^2$:

$$
\frac{(N-p-1)s^2}{\sigma^2} \sim \chi^2_{N-p-1}
$$

and this is independent of $\hat{\beta}_j$ (since $\hat{\beta}$ depends on $\mathbf{P}\varepsilon$ while $s^2$ depends on $\mathbf{M}\varepsilon$, and $\mathbf{PM} = \mathbf{0}$).

**Step 3: Form the $t$-ratio.** By the definition of the $t$-distribution:

$$
t_j = \frac{\hat{\beta}_j / (\sigma\sqrt{v_j})}{\sqrt{s^2/\sigma^2}} = \frac{\hat{\beta}_j}{s\sqrt{v_j}} \sim t_{N-p-1} \qquad \blacksquare
$$

### General Confidence Interval

For a $(1 - \alpha)$ confidence interval for $\beta_j$:

$$
\hat{\beta}_j \pm t_{N-p-1}(1 - \alpha/2)\; s\sqrt{v_j}
$$

---

## Example: Reproducing Linear Regression Output

### Problem

Using the Advertising dataset, reproduce the main regression output for the model $\text{Sales} \sim \text{TV} + \text{Radio} + \text{Newspaper}$ by computing coefficients, standard errors, $t$-statistics, $p$-values, and confidence intervals from scratch.

!!! info "Reference"
    - [Khan Academy: Using Least-Squares Regression Output](https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/least-squares-regression/v/using-least-squares-regression-output)
    - [Khan Academy: Interpreting Computer Regression Data](https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/assessing-fit-least-squares-regression/v/interpreting-computer-regression-data)

### Implementation

```python
import numpy as np
import pandas as pd
from scipy import stats

# Load the Advertising dataset
dataset_url = (
    'https://raw.githubusercontent.com/justmarkham/'
    'scikit-learn-videos/master/data/Advertising.csv'
)
advertising_data = pd.read_csv(dataset_url, usecols=[1, 2, 3, 4])

# Train/test split (70/30)
total_observations = advertising_data.shape[0]
test_set_ratio = 0.3
train_count = int(total_observations * (1 - test_set_ratio))
training_data = advertising_data.iloc[:train_count]

# Response vector y and design matrix X (with intercept column)
y = np.array(training_data.Sales).reshape(-1, 1)
n = y.shape[0]
X = np.concatenate(
    (np.ones((n, 1)), np.array(training_data.iloc[:, :-1])),
    axis=1
)
p_plus_1 = X.shape[1]  # number of parameters (including intercept)

# OLS coefficients: β̂ = (X'X)⁻¹X'y
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

# Predicted values and residual standard error
y_hat = X @ beta_hat
s = np.sqrt(np.sum((y - y_hat) ** 2) / (n - p_plus_1))

# Variance-covariance matrix: (X'X)⁻¹
cov_matrix = np.linalg.inv(X.T @ X)

# Print regression table
print("=" * 100)
print("\t\t    coef    std err \t     t      P>|t|"
      "     [0.025      0.975] ")
print("-" * 100)

variable_names = ["Intercept", "TV", "Radio", "Newspaper"]

for name, j in zip(variable_names, range(p_plus_1)):
    coef = beta_hat[j, 0]
    v_j = cov_matrix[j, j]
    se = s * np.sqrt(v_j)
    t_stat = coef / se
    p_val = 2 * stats.t(n - p_plus_1).sf(np.abs(t_stat))
    ci_lower = coef - stats.t(n - p_plus_1).ppf(0.975) * se
    ci_upper = coef + stats.t(n - p_plus_1).ppf(0.975) * se
    print(f"{name:10}    {coef:10.4f} {se:10.3f} "
          f"{t_stat:10.3f} {p_val:10.3f} "
          f"{ci_lower:10.3f} {ci_upper:10.3f}")

print("=" * 100)
```

### Understanding the Output

Each row of the regression table contains:

- **coef**: The OLS estimate $\hat{\beta}_j$.
- **std err**: The standard error $s\sqrt{v_j}$, where $v_j = ((\mathbf{X}^T\mathbf{X})^{-1})_{jj}$.
- **t**: The $t$-statistic $t_j = \hat{\beta}_j / (s\sqrt{v_j})$.
- **P>|t|**: The two-tailed $p$-value from $t_{N-p-1}$.
- **[0.025, 0.975]**: The 95% confidence interval $\hat{\beta}_j \pm t_{N-p-1}(0.975) \cdot s\sqrt{v_j}$.

A predictor is statistically significant at the 5% level when its $p$-value is less than 0.05, equivalently when its 95% confidence interval excludes zero.
