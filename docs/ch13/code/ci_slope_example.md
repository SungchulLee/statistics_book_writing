# Confidence Interval for Slope (Caffeine Example)

## Overview

This page demonstrates how to construct a confidence interval for the slope coefficient in simple linear regression. Using a study of 20 students examining the relationship between study hours and caffeine consumption, we compute the margin of error via the $t$-distribution and build a 95% confidence interval for $\beta_1$.

## Mathematical Background

In simple linear regression, the model is

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \qquad \varepsilon_i \overset{\text{iid}}{\sim} N(0, \sigma^2).
$$

The least-squares estimator $\hat{\beta}_1$ has a sampling distribution

$$
\hat{\beta}_1 \sim N\!\left(\beta_1,\; \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}\right).
$$

Because $\sigma^2$ is unknown, we replace it with the residual variance $s^2$ and use the $t$-distribution with $n - 2$ degrees of freedom. A $(1 - \alpha)$-level confidence interval for $\beta_1$ is

$$
\hat{\beta}_1 \pm t^*_{n-2,\,\alpha/2} \cdot \mathrm{SE}(\hat{\beta}_1),
$$

where $t^*_{n-2,\,\alpha/2}$ is the critical value from the $t$-distribution.

## Code

The following code computes a 95% confidence interval for the slope in the caffeine study.

```python
from scipy import stats

# Regression output
beta_1_hat = 0.164      # estimated slope
standard_error = 0.057  # SE of the slope

# Sample size and degrees of freedom
n = 20
df = n - 2

# 95% confidence interval
confidence_level = 0.95
alpha = 1 - confidence_level
t_star = stats.t(df).ppf(1 - alpha / 2)

margin_of_error = t_star * standard_error

ci_lower = beta_1_hat - margin_of_error
ci_upper = beta_1_hat + margin_of_error

print(f"Slope estimate: {beta_1_hat:.4f}")
print(f"Standard error: {standard_error:.4f}")
print(f"t* (df={df}): {t_star:.4f}")
print(f"Margin of error: {margin_of_error:.4f}")
print(f"\n{confidence_level:.0%} confidence interval of the slope")
print(f"{beta_1_hat:.4f} +/- {margin_of_error:.4f}")
print(f"({ci_lower:.4f}, {ci_upper:.4f})")
```

## Interpretation

- The estimated slope $\hat{\beta}_1 = 0.164$ means that, on average, each additional unit of caffeine consumption is associated with an increase of 0.164 study hours.
- The 95% confidence interval tells us that, if we were to repeat the study many times, approximately 95% of the resulting intervals would contain the true slope $\beta_1$.
- Because the interval does not contain zero (both bounds are positive), we can conclude at the 5% significance level that there is a statistically significant positive association between caffeine consumption and study hours.
- The margin of error depends on three quantities: the critical value $t^*$ (which grows as the confidence level increases or $n$ decreases), the standard error of the slope, and implicitly the spread of the predictor values.

## Exercises

**Exercise 1.** Compute a 99% confidence interval for the same slope estimate. How does the interval width compare to the 95% interval?

??? success "Solution to Exercise 1"

    Replace $\alpha = 0.05$ with $\alpha = 0.01$:

    ```python
    alpha_99 = 0.01
    t_star_99 = stats.t(df).ppf(1 - alpha_99 / 2)  # approximately 2.8784
    margin_99 = t_star_99 * standard_error
    ci_lower_99 = beta_1_hat - margin_99
    ci_upper_99 = beta_1_hat + margin_99
    print(f"99% CI: ({ci_lower_99:.4f}, {ci_upper_99:.4f})")
    ```

    The 99% interval is wider than the 95% interval because a higher confidence level requires a larger critical value $t^*$. $\square$

---

**Exercise 2.** If the sample size were $n = 50$ instead of $n = 20$ (with the same $\hat{\beta}_1$ and SE), how would the 95% confidence interval change? Explain why.

??? success "Solution to Exercise 2"

    With $n = 50$, the degrees of freedom become $df = 48$. The critical value $t^*_{48, 0.025}$ is closer to the standard normal value $z^* = 1.96$, so the interval narrows slightly. More importantly, with a larger sample the standard error itself would likely decrease (since $\mathrm{SE} \propto 1/\sqrt{\sum(x_i - \bar{x})^2}$), producing a substantially tighter interval. Larger samples yield more precise estimates. $\square$

---

**Exercise 3.** Use the confidence interval to perform a two-sided hypothesis test of $H_0\colon \beta_1 = 0$ at the $\alpha = 0.05$ level. State the decision and explain the duality between confidence intervals and hypothesis tests.

??? success "Solution to Exercise 3"

    The 95% confidence interval is approximately $(0.044, 0.284)$. Since $0$ is not contained in this interval, we reject $H_0\colon \beta_1 = 0$ at the 5% significance level. The duality states that rejecting $H_0$ at level $\alpha$ is equivalent to the $(1-\alpha)$-level confidence interval not containing the null value. $\square$

---

**Exercise 4.** Derive the formula for $\mathrm{SE}(\hat{\beta}_1)$ starting from $\hat{\beta}_1 = \sum_{i=1}^n w_i y_i$ where $w_i = (x_i - \bar{x}) / \sum_{j=1}^n (x_j - \bar{x})^2$.

??? success "Solution to Exercise 4"

    Since $\hat{\beta}_1 = \sum_i w_i y_i$ with $w_i = (x_i - \bar{x})/S_{xx}$ where $S_{xx} = \sum_j (x_j - \bar{x})^2$, and the $y_i$ are independent with variance $\sigma^2$:

    $$
    \mathrm{Var}(\hat{\beta}_1) = \sum_{i=1}^n w_i^2 \,\sigma^2 = \frac{\sigma^2}{S_{xx}^2}\sum_{i=1}^n (x_i - \bar{x})^2 = \frac{\sigma^2}{S_{xx}}.
    $$

    Therefore $\mathrm{SE}(\hat{\beta}_1) = s / \sqrt{S_{xx}}$, where $s$ is the residual standard error. $\square$

---

**Exercise 5.** Prove that as $n \to \infty$, the $t$-based confidence interval converges to the $z$-based interval. Under what conditions does the distinction matter in practice?

??? success "Solution to Exercise 5"

    The $t$-distribution with $\nu$ degrees of freedom converges in distribution to $N(0,1)$ as $\nu \to \infty$. For the confidence interval, as $n \to \infty$ we have $df = n - 2 \to \infty$, so $t^*_{n-2,\,\alpha/2} \to z_{\alpha/2}$, and the $t$-interval becomes the $z$-interval. The distinction matters in practice when $n$ is small (roughly $n < 30$), where the heavier tails of the $t$-distribution produce wider intervals that properly account for the additional uncertainty in estimating $\sigma$. $\square$
