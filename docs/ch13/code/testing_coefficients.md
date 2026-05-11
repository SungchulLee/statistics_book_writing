# Testing Coefficients Examples

## Overview

This page demonstrates hypothesis testing for regression coefficients using statsmodels. Through three progressively complex examples, we show how to extract $t$-statistics, $p$-values, and confidence intervals from OLS output, and how to interpret statistical significance in the context of single-predictor and multiple-predictor regression models.

## Mathematical Background

For the linear model $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$, the hypothesis test for the $j$-th coefficient is

$$
H_0\colon \beta_j = 0 \quad \text{vs} \quad H_1\colon \beta_j \neq 0.
$$

The test statistic follows a $t$-distribution under $H_0$:

$$
t_j = \frac{\hat{\beta}_j}{\mathrm{SE}(\hat{\beta}_j)} \sim t_{n-k} \quad \text{under } H_0.
$$

The two-sided $p$-value is

$$
p = 2\,P(T_{n-k} > |t_j|).
$$

We reject $H_0$ at significance level $\alpha$ if $p < \alpha$, equivalently if $|t_j| > t^*_{n-k,\,\alpha/2}$, equivalently if the $(1-\alpha)$-level confidence interval for $\beta_j$ does not contain zero.

## Code

### Example 1: Two Predictors

```python
import numpy as np
import statsmodels.api as sm

np.random.seed(0)
X = np.random.rand(100, 2)
y = 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100)

X_const = sm.add_constant(X)
results = sm.OLS(y, X_const).fit()
print(results.summary())
```

### Example 2: Extracting p-values and Confidence Intervals

```python
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2.5 * X[:, 0] + np.random.randn(100)

X_const = sm.add_constant(X)
results = sm.OLS(y, X_const).fit()

p_values = results.pvalues
confidence_intervals = results.conf_int()

print("P-values:", p_values)
print("95% CI:\n", confidence_intervals)
```

### Example 3: Interpreting Multiple Predictors

```python
np.random.seed(42)
study_hours = np.random.rand(100) * 10
sleep_hours = np.random.rand(100) * 8
exam_scores = 5 + 2.5 * study_hours - 1.5 * sleep_hours + np.random.randn(100) * 2

X = np.column_stack((study_hours, sleep_hours))
X_const = sm.add_constant(X)
results = sm.OLS(exam_scores, X_const).fit()

for i, name in enumerate(["Intercept", "Study Hours", "Sleep Hours"]):
    pval = results.pvalues[i]
    ci = results.conf_int().iloc[i]
    status = "Significant" if pval < 0.05 else "Not Significant"
    print(f"{name}: coef={results.params[i]:.4f}, "
          f"p={pval:.4f} ({status}), "
          f"95% CI=({ci[0]:.4f}, {ci[1]:.4f})")
```

## Interpretation

- In Example 1, both predictors ($x_1$ and $x_2$) have true nonzero coefficients (3 and 5), so we expect both $p$-values to be small and both to be declared significant.
- In Example 2, the single predictor has a true coefficient of 2.5 with noise standard deviation 1. The $p$-value for the slope should be very small, confirming the linear relationship.
- In Example 3, "Study Hours" has a positive effect (coefficient $\approx 2.5$) and "Sleep Hours" has a negative effect (coefficient $\approx -1.5$) on exam scores. Both should be statistically significant. The confidence intervals provide a range of plausible effect sizes.
- A predictor being "not significant" ($p > 0.05$) does not prove the effect is zero; it means we lack sufficient evidence to reject $H_0$ at the chosen level. The confidence interval conveys the same information more informatively.

## Exercises

**Exercise 1.** In Example 1, compute the $t$-statistic for $\hat{\beta}_1$ manually from the coefficient and its standard error. Verify it matches the value from `results.tvalues[1]`.

??? success "Solution to Exercise 1"

    ```python
    t_manual = results.params[1] / results.bse[1]
    t_auto = results.tvalues[1]
    print(f"Manual t: {t_manual:.4f}")
    print(f"Auto t:   {t_auto:.4f}")
    print(f"Match: {np.isclose(t_manual, t_auto)}")
    ```

    The two values are identical because the summary table simply computes $t_j = \hat{\beta}_j / \mathrm{SE}(\hat{\beta}_j)$. $\square$

---

**Exercise 2.** Modify Example 2 to increase the noise standard deviation from 1 to 5. How does this affect the $p$-value and confidence interval for the slope?

??? success "Solution to Exercise 2"

    ```python
    y_noisy = 2.5 * X[:, 0] + 5 * np.random.randn(100)
    results_noisy = sm.OLS(y_noisy, X_const).fit()
    print(results_noisy.summary())
    ```

    With more noise, the residual standard error $s$ increases, inflating $\mathrm{SE}(\hat{\beta}_1)$. The $t$-statistic decreases, the $p$-value increases, and the confidence interval widens. With enough noise, the slope may no longer be statistically significant despite the true effect being nonzero. $\square$

---

**Exercise 3.** In Example 3, add a third predictor `caffeine = np.random.rand(100) * 5` that is unrelated to exam scores. What do you expect for its $p$-value, and why?

??? success "Solution to Exercise 3"

    Since caffeine is generated independently of exam scores, its true coefficient is zero. The OLS estimate $\hat{\beta}_{\text{caffeine}}$ should be close to zero with a large $p$-value (typically $> 0.05$). Occasionally, by chance alone (about 5% of the time at the $\alpha = 0.05$ level), the $p$-value may fall below 0.05, which is a Type I error. $\square$

---

**Exercise 4.** Explain the connection between the $F$-test for overall model significance and the individual $t$-tests. When can they give different conclusions?

??? success "Solution to Exercise 4"

    The $F$-test evaluates $H_0\colon \beta_1 = \beta_2 = \cdots = \beta_{k-1} = 0$ (all slopes are zero simultaneously), while each $t$-test evaluates a single coefficient. When predictors are uncorrelated, the individual $t$-tests are independent, and rejecting any $t$-test essentially implies the $F$-test rejects. When predictors are correlated, it is possible for the $F$-test to reject (the model is useful overall) while no individual $t$-test rejects (no single predictor is significant after adjusting for the others). This occurs with multicollinearity. $\square$

---

**Exercise 5.** Prove that for simple linear regression with one predictor, $t_1^2 = F$ where $F$ is the overall $F$-statistic. Under what condition does this equivalence break down?

??? success "Solution to Exercise 5"

    In simple regression, $k = 2$ (intercept + slope), and the $F$-statistic is

    $$
    F = \frac{\mathrm{ESS}/1}{\mathrm{RSS}/(n-2)}.
    $$

    The $t$-statistic for the slope is $t_1 = \hat{\beta}_1 / \mathrm{SE}(\hat{\beta}_1)$. One can show algebraically that $t_1^2 = \mathrm{ESS}/s^2 = F$. This equivalence holds exactly when there is a single predictor (one numerator degree of freedom). It breaks down in multiple regression where the $F$-test has $k-1 > 1$ numerator degrees of freedom and tests all slopes simultaneously. $\square$
