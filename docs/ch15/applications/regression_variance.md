# Variance Testing in Regression

Linear regression assumes that the error terms have constant variance across all levels of the predictors: $\operatorname{Var}(\varepsilon_i) = \sigma^2$ for all $i$. This assumption is called **homoscedasticity**. When it fails -- when the variance of the errors depends on the predictor values or on the fitted values -- the condition is called **heteroscedasticity**. This section covers the formal tests used to detect heteroscedasticity in regression residuals and discusses the consequences of ignoring it.

## Why Heteroscedasticity Matters

In the linear regression model

$$
Y_i = \beta_0 + \beta_1 X_{i1} + \cdots + \beta_p X_{ip} + \varepsilon_i
$$

the OLS estimator $\hat{\boldsymbol{\beta}}$ remains unbiased and consistent even under heteroscedasticity. However, two important problems arise:

1. **Inefficiency.** OLS is no longer the best linear unbiased estimator (BLUE). Weighted least squares (WLS) or generalized least squares (GLS) can produce more efficient estimates.
2. **Invalid inference.** The standard errors computed by OLS assume constant variance. Under heteroscedasticity, these standard errors are biased, leading to incorrect $t$-statistics, $p$-values, and confidence intervals for the regression coefficients.

## Visual Detection

Before applying formal tests, plot the residuals against the fitted values $\hat{Y}_i$:

- **Homoscedastic pattern:** The residuals form a roughly constant band around zero.
- **Heteroscedastic pattern:** The spread of the residuals increases (or decreases) systematically with $\hat{Y}_i$. Common patterns include a "funnel" shape (spread increasing with fitted values) and a "bow-tie" shape (spread increasing then decreasing).

## The Breusch-Pagan Test

The Breusch-Pagan (1979) test is the most widely used formal test for heteroscedasticity. It tests whether the squared residuals are related to the predictor variables.

**Procedure:**

**Step 1.** Fit the regression model and obtain the OLS residuals $e_i = Y_i - \hat{Y}_i$.

**Step 2.** Regress the squared residuals $e_i^2$ on the original predictors $X_{i1}, \ldots, X_{ip}$:

$$
e_i^2 = \gamma_0 + \gamma_1 X_{i1} + \cdots + \gamma_p X_{ip} + u_i
$$

**Step 3.** Compute the test statistic as $n$ times the $R^2$ from this auxiliary regression:

$$
\text{BP} = n \cdot R^2_{\text{aux}}
$$

**Step 4.** Under $H_0\colon$ homoscedasticity, the statistic follows approximately a chi-square distribution:

$$
\text{BP} \sim \chi^2_p
$$

where $p$ is the number of predictors in the auxiliary regression.

**Hypotheses:**

$$
H_0\colon \operatorname{Var}(\varepsilon_i) = \sigma^2 \text{ (constant)}
$$

$$
H_1\colon \operatorname{Var}(\varepsilon_i) = h(X_{i1}, \ldots, X_{ip}) \text{ (depends on predictors)}
$$

Reject $H_0$ if $\text{BP} > \chi^2_{1-\alpha,\, p}$.

## White's Test

White (1980) proposed a more general test that does not require specifying the form of heteroscedasticity. Instead of regressing $e_i^2$ on the original predictors alone, White's test includes their squares and cross-products.

**Auxiliary regression for White's test:**

$$
e_i^2 = \gamma_0 + \sum_{j=1}^{p}\gamma_j X_{ij} + \sum_{j=1}^{p}\gamma_{jj} X_{ij}^2 + \sum_{j<l}\gamma_{jl} X_{ij} X_{il} + u_i
$$

The test statistic is again $n \cdot R^2_{\text{aux}}$, but now with $q$ regressors (where $q$ includes all the original predictors, their squares, and cross-products):

$$
\text{W} = n \cdot R^2_{\text{aux}} \sim \chi^2_q
$$

!!! note "Breusch-Pagan vs. White"
    The Breusch-Pagan test detects heteroscedasticity that is a **linear** function of the predictors. White's test detects heteroscedasticity of **any** functional form. White's test is more general but uses more degrees of freedom, which can reduce power when the heteroscedasticity is indeed linear.

## The Goldfeld-Quandt Test

The Goldfeld-Quandt (1965) test is a simpler approach that splits the data into two groups based on the suspected source of heteroscedasticity and applies the F-test for equal variances.

**Procedure:**

1. Order the observations by the predictor $X$ suspected of causing heteroscedasticity.
2. Drop the middle $c$ observations (typically $c \approx n/5$) to sharpen the contrast.
3. Fit separate regressions to the lower and upper groups.
4. Compute the F-statistic as the ratio of the residual sum of squares from the upper group to the lower group.

Under $H_0$, this ratio follows an $F$ distribution. The Goldfeld-Quandt test is intuitive but limited to heteroscedasticity related to a single predictor.

## Consequences and Remedies

When heteroscedasticity is detected:

1. **Heteroscedasticity-consistent standard errors.** Use robust standard errors (also called White standard errors or sandwich estimators) that remain valid under heteroscedasticity without changing the coefficient estimates:

$$
\widehat{\operatorname{Var}}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}'\mathbf{X})^{-1}\left(\sum_{i=1}^{n} e_i^2 \mathbf{x}_i\mathbf{x}_i'\right)(\mathbf{X}'\mathbf{X})^{-1}
$$

2. **Weighted least squares.** If the form of the heteroscedasticity is known or can be estimated (e.g., $\operatorname{Var}(\varepsilon_i) \propto X_i^2$), WLS produces more efficient estimates.

3. **Variance-stabilizing transformation.** Transformations such as $\ln Y$ or $\sqrt{Y}$ can sometimes stabilize the variance.

## Python Implementation

```python
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

# Generate data with heteroscedasticity
rng = np.random.default_rng(42)
n = 100
X = rng.uniform(1, 10, size=n)
epsilon = rng.normal(0, 1, size=n) * X  # variance increases with X
Y = 3 + 2 * X + epsilon

# Fit OLS regression
X_with_const = sm.add_constant(X)
model = sm.OLS(Y, X_with_const).fit()
residuals = model.resid

# Breusch-Pagan test
bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_with_const)
print(f"Breusch-Pagan statistic: {bp_stat:.4f}")
print(f"Breusch-Pagan p-value:   {bp_p:.4f}")

# White's test
white_stat, white_p, _, _ = het_white(residuals, X_with_const)
print(f"White statistic: {white_stat:.4f}")
print(f"White p-value:   {white_p:.4f}")

# Robust standard errors
robust_model = model.get_robustcov_results(cov_type='HC3')
print(f"\nOLS std errors:    {model.bse}")
print(f"Robust std errors: {robust_model.bse}")
```

## Summary

Heteroscedasticity in regression is detected through visual inspection (residual plots) and formal tests (Breusch-Pagan, White, Goldfeld-Quandt). The Breusch-Pagan test is the standard first choice for its balance of simplicity and power. When heteroscedasticity is present, robust standard errors provide valid inference without requiring a correctly specified variance function.


## Exercises

**Exercise 1.**
Describe the main concept of Variance Testing in Regression and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Variance Testing in Regression is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

---

**Exercise 2.**
State the key assumptions required by the method discussed here. How can each assumption be checked?

??? success "Solution to Exercise 2"
    The main assumptions typically include: (1) independence of observations -- verified by understanding the data collection process and checking for serial correlation; (2) distributional requirements (e.g., normality) -- checked with Q-Q plots and formal tests like Shapiro-Wilk; (3) equal variances (if applicable) -- assessed with boxplots and Levene's test. When assumptions are violated, consider robust alternatives, transformations, or nonparametric methods.

---

**Exercise 3.**
Work through a small numerical example illustrating the application of the technique from this section.

??? success "Solution to Exercise 3"
    A structured approach to applying this technique involves: (1) clearly stating the hypotheses or estimation goal; (2) verifying that the data meet the required assumptions; (3) computing the relevant test statistic, estimate, or model fit; (4) obtaining the p-value, confidence interval, or posterior distribution; (5) interpreting the result in the context of the original question. Following these steps systematically ensures a rigorous and reproducible analysis.

---

**Exercise 4.**
Compare the approach from this section with an alternative method. When would you choose each?

??? success "Solution to Exercise 4"
    The method discussed here is appropriate when its assumptions hold and the sample size is sufficient for the asymptotic approximations to be accurate. Alternative approaches include: (1) nonparametric methods -- preferred when distributional assumptions are suspect; (2) bootstrap methods -- useful when analytical reference distributions are unavailable; (3) Bayesian methods -- valuable when incorporating prior information or when direct probability statements about parameters are desired. Running multiple approaches and comparing results provides a useful robustness check.
