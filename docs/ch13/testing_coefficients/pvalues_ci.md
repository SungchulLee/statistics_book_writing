# Understanding p-values and Confidence Intervals for Coefficients

In linear regression analysis, understanding **p-values** and **confidence intervals** is essential for interpreting the significance and precision of regression coefficients. These statistical measures provide insight into how strongly the predictors influence the dependent variable and the uncertainty associated with the estimated effects.

---

## p-values in Linear Regression

The p-value represents the probability that the observed relationship between a predictor and the dependent variable is due to random chance, assuming the null hypothesis is true.

The hypotheses are:

- **Null Hypothesis ($H_0$):** $\beta_i = 0$ — the predictor $X_i$ has no effect on the outcome $Y$.
- **Alternative Hypothesis ($H_1$):** $\beta_i \neq 0$ — the predictor $X_i$ has a significant effect on $Y$.

When we perform a t-test for each coefficient, the t-statistic computed from $\hat{\beta}_i$ and its standard error yields the p-value, which assesses statistical significance.

- **Low p-value (< 0.05):** Reject $H_0$. The predictor is statistically significant and contributes to explaining variation in the dependent variable.
- **High p-value ($\geq$ 0.05):** Fail to reject $H_0$. The predictor is not statistically significant and may not contribute much to the model.

---

## Example: p-value Interpretation

Consider a regression model predicting house prices based on the number of bedrooms, lot size, and age of the house:

| Predictor | p-value | Conclusion |
|-----------|---------|------------|
| Age of the house | 0.02 | Significant at 0.05 level — age is a significant predictor of house prices |
| Lot size | 0.25 | Not significant at 0.05 level — lot size may not strongly influence house prices in this model |

---

## Confidence Intervals for Coefficients

**Confidence intervals** provide a range of values within which we expect the true population coefficient to lie, based on the sample data. While p-values offer a binary decision, confidence intervals convey the precision of the estimate.

A **95% confidence interval** means that if we were to take 100 different samples and compute 100 confidence intervals, approximately 95 would contain the true coefficient value.

The formula for a 95% CI:

$$
CI = \hat{\beta}_i \pm \left( t_{\alpha/2} \times SE(\hat{\beta}_i) \right)
$$

where:

- $\hat{\beta}_i$ is the estimated regression coefficient,
- $SE(\hat{\beta}_i)$ is the standard error of the estimate,
- $t_{\alpha/2}$ is the critical value from the t-distribution at the 95% confidence level.

**Key interpretation rules:**

- If the CI **contains 0**, the predictor may not be significantly different from zero — its effect is uncertain.
- If the CI **excludes 0**, the predictor likely has a significant effect on the outcome.

---

## Example: Confidence Interval Interpretation

Consider a regression model predicting salary:

| Predictor | $\hat{\beta}$ | 95% CI | Interpretation |
|-----------|---------------|--------|----------------|
| Years of education | 2.5 | (1.2, 3.8) | Significant — CI excludes 0. The true effect lies between 1.2 and 3.8. |
| Industry experience | 0.55 | (−0.4, 1.5) | Uncertain — CI includes 0. May not be an important predictor. |

---

## Relationship Between p-values and Confidence Intervals

p-values and confidence intervals are closely related and generally lead to the same conclusion:

- If the **p-value < 0.05**, the corresponding 95% CI will **exclude 0** — the predictor is significant.
- If the **p-value $\geq$ 0.05**, the 95% CI will likely **include 0** — the predictor is not significant.

Both measures should be used together to understand not only statistical significance but also the practical significance and uncertainty of the predictor's effect.

!!! tip "Best Practice"
    Report both p-values and confidence intervals. The p-value tells you *whether* an effect exists; the confidence interval tells you *how large* the effect might be and how precisely it is estimated.

---

## Practical Interpretation

Suppose a regression output provides the following for the predictor "advertising budget":

| Metric | Value |
|--------|-------|
| Coefficient | 2.4 |
| p-value | 0.001 |
| 95% CI | (1.9, 2.9) |

**Interpretation:** Advertising budget has a significant positive impact on sales, with a very low probability (0.001) that the observed effect is due to chance. We are 95% confident that the true effect lies between 1.9 and 2.9 units of sales per unit increase in advertising budget.

---

## Summary

Understanding p-values and confidence intervals for regression coefficients is crucial for informed decision-making. p-values assess whether a predictor is statistically significant, while confidence intervals provide a measure of precision and uncertainty. Together, these tools guide researchers and analysts in evaluating the strength and reliability of the model's findings.
