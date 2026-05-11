# Normality Assumption


## Definition

The normality assumption asserts that the residuals of the regression model are normally distributed, with a mean of zero:

$$
\epsilon_i \sim N(0, \sigma^2)
$$

This means that when the residuals are plotted, they should form a bell-shaped (Gaussian) curve centered around zero.

## Importance

Normality of residuals is crucial because many inferential statistics in linear regression rely on this assumption:

- **t-tests for coefficients** — Testing whether individual regression coefficients are significantly different from zero requires normally distributed errors.
- **F-tests for overall significance** — The overall F-test for model significance assumes normal errors.
- **Confidence intervals** — The construction of confidence intervals for coefficients uses the t-distribution, which is derived under the normality assumption.
- **Prediction intervals** — Prediction intervals for new observations require normality to be valid.

**Important caveat:** The normality assumption is **not required** for unbiased estimation of regression coefficients. The OLS estimator $\hat{\beta}$ is unbiased regardless of the error distribution. However, without normality, the exact distributional results used for inference (t-tests, F-tests) are only asymptotically valid via the Central Limit Theorem.

## When Normality Matters Most

| Situation | Normality Importance |
|-----------|---------------------|
| Small sample size ($n < 30$) | Critical — CLT does not provide sufficient approximation |
| Large sample size ($n > 100$) | Less critical — CLT ensures approximate normality of test statistics |
| Constructing prediction intervals | Always important regardless of sample size |
| Heavy-tailed data | Important — outliers can strongly influence OLS estimates |

## Diagnostics

- **Q-Q Plot (Quantile-Quantile Plot):** Compares the quantiles of the residuals to the quantiles of a normal distribution. If the points fall approximately along a straight line, the residuals are likely normally distributed.
- **Shapiro-Wilk Test:** A formal statistical test for normality. A non-significant result suggests that the residuals are normally distributed.
- **Histogram of Residuals:** A simple visual check — the histogram should resemble a bell curve centered at zero.
- **Jarque-Bera Test:** Tests whether the skewness and kurtosis of the residuals match a normal distribution.

## Remedies for Non-Normality

- **Transformations:** Applying transformations to the dependent variable, such as log, square root, or Box-Cox transformations, can sometimes correct non-normality.
- **Robust Regression:** When normality cannot be achieved, robust regression techniques can provide valid results by down-weighting the influence of outliers.
- **Bootstrapping:** Bootstrap methods provide valid inference without relying on distributional assumptions.
- **Larger samples:** With sufficiently large samples, the Central Limit Theorem ensures that test statistics are approximately normally distributed even if the errors are not.

For detailed diagnostic methods, see [Checking Normality](checking_normality.md).
## Exercises

**Exercise 1.**
A researcher fits a regression model with $n = 15$ observations. The Q-Q plot of residuals shows substantial right skew. Explain why normality matters more in this setting than it would with $n = 500$.

??? success "Solution to Exercise 1"
    With only $n = 15$, the **Central Limit Theorem** does not provide a good approximation. The exact distributions of the $t$-statistics for individual coefficients and the $F$-statistic for overall significance depend on the normality of the errors. Right-skewed residuals mean the actual sampling distribution differs from the assumed $t$-distribution, producing inaccurate p-values and confidence intervals.

    With $n = 500$, the CLT ensures that the sampling distributions are approximately normal regardless of the error distribution, so moderate departures from normality have negligible impact on inference.

---

**Exercise 2.**
Explain why normality is not required for unbiased estimation of $\hat{\beta}$ but is required for exact validity of $t$-tests on the coefficients.

??? success "Solution to Exercise 2"
    The OLS estimator $\hat{\beta} = (X^T X)^{-1} X^T Y$ is a linear function of $Y$. Its expectation is:

    $$
    E[\hat{\beta}] = (X^T X)^{-1} X^T E[Y] = (X^T X)^{-1} X^T X \beta = \beta
    $$

    This derivation uses only $E[\varepsilon] = 0$, not normality. Hence $\hat{\beta}$ is unbiased regardless of the error distribution.

    The $t$-test statistic $t = \hat{\beta}_j / \text{SE}(\hat{\beta}_j)$ follows an exact $t$-distribution only when $\varepsilon \sim N(0, \sigma^2 I)$. Without normality, the ratio does not have an exact $t$-distribution for finite samples, so p-values from $t$-tables are only approximate.
