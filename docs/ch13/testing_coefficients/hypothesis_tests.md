# Hypothesis Tests for Regression Coefficients (t-tests)

In linear regression, a key objective is to evaluate the impact of each predictor on the dependent variable. Hypothesis tests for regression coefficients, typically conducted using **t-tests**, assess whether the predictors significantly contribute to the model.

---

## Formulation of the t-test

A t-test in linear regression evaluates the null hypothesis that a regression coefficient equals zero, indicating that the corresponding predictor variable has no effect on the dependent variable.

The hypotheses are:

- **Null Hypothesis ($H_0$):** $\beta_i = 0$ — the predictor has no effect.
- **Alternative Hypothesis ($H_1$):** $\beta_i \neq 0$ — the predictor has a significant effect.

The **t-statistic** is calculated for each coefficient:

$$
t = \frac{\hat{\beta}_i}{SE(\hat{\beta}_i)}
$$

where:

- $\hat{\beta}_i$ is the estimated coefficient for predictor $i$,
- $SE(\hat{\beta}_i)$ is the standard error of the estimated coefficient, representing the variability of the estimate.

The t-statistic measures how many standard errors the estimated coefficient is away from zero. A larger absolute value indicates stronger evidence against the null hypothesis.

---

## p-values in t-tests

The **p-value** is the probability of observing a t-statistic as extreme as (or more extreme than) the one computed, assuming the null hypothesis is true.

- **Low p-value (< 0.05):** The coefficient is significantly different from zero — reject $H_0$. The predictor has a significant impact on the dependent variable.
- **High p-value ($\geq$ 0.05):** We fail to reject $H_0$. The predictor might not significantly affect the dependent variable, and its inclusion may not improve predictive performance.

!!! example "Salary Prediction"
    In a linear regression model predicting salary based on years of experience, a t-test assesses whether "years of experience" has a significant effect on salary. If the p-value for its coefficient is 0.001, we conclude that years of experience is a significant predictor of salary.

---

## Confidence Intervals for Coefficients

While p-values provide a binary significance decision, **confidence intervals** (CIs) give a range of plausible values for the coefficient. A 95% CI is computed as:

$$
CI = \hat{\beta}_i \pm \left( t_{\alpha/2} \times SE(\hat{\beta}_i) \right)
$$

where $t_{\alpha/2}$ is the critical value from the t-distribution at the desired confidence level.

**Interpretation:**

- If the CI **includes 0**, the coefficient is not significantly different from zero — the predictor may not have a significant impact.
- If the CI **excludes 0**, the predictor is significant, providing stronger evidence that it contributes meaningfully to the model.

!!! example "Advertising Budget"
    For a predictor "advertising budget" in a regression model predicting sales, a 95% CI of $(0.03, 0.12)$ does not contain 0, so we conclude that advertising budget significantly influences sales.

---

## Interpreting Significance

The significance of a predictor is determined by the t-test outcome and its associated p-value.

**Statistically Significant Coefficient** ($p < 0.05$):

- The predictor explains some variability in the dependent variable and improves the model.
- Changes in this variable are associated with changes in the outcome.
- For example, if "education level" is significant in predicting job performance, education level is an important determinant.

**Statistically Insignificant Coefficient** ($p \geq 0.05$):

- There is not enough evidence to claim this predictor impacts the dependent variable.
- The variable may be removed from the model if it does not improve overall fit.

---

## The Role of Multicollinearity in t-tests

**Multicollinearity** occurs when two or more predictors are highly correlated, making it difficult to distinguish their individual effects. This can:

- Inflate standard errors of coefficients, reducing t-statistics.
- Inflate p-values, leading to incorrect conclusions about significance.
- Widen confidence intervals, making it harder to conclude significance.

**Addressing Multicollinearity:**

- **Variance Inflation Factor (VIF):** A VIF greater than 5 or 10 suggests multicollinearity. Corrective actions include removing or combining correlated predictors.
- **Principal Component Analysis (PCA):** Transforms predictors into uncorrelated components.
- **Ridge Regression:** Regularizes coefficients to reduce the impact of multicollinearity.

---

## Summary

Hypothesis tests for regression coefficients (t-tests) are critical for determining the significance of individual predictors. By evaluating p-values and confidence intervals, we determine which predictors significantly contribute to explaining variability in the dependent variable. It is also important to check for multicollinearity, which can obscure the true significance of predictors.
