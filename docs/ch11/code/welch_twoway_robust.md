# Two-Way Welch Analysis of Variance (Robust HC3)

## Overview

When a two-way factorial design exhibits heteroscedastic errors, the standard ANOVA $F$-tests based on the assumption of common variance become unreliable. A practical alternative is to fit an OLS model with the full factorial specification and then use the HC3 heteroscedasticity-consistent covariance estimator combined with Wald $F$-tests to test main effects and interactions. This approach provides a robust analog of two-way ANOVA without requiring a specialized Welch-James implementation.

## The Robust OLS Approach

The model is the standard two-way factorial:

$$
y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}
$$

where $\text{Var}(\varepsilon_{ijk})$ is no longer assumed constant. The OLS coefficient estimates $\hat{\boldsymbol{\beta}}$ remain unbiased and consistent, but the classical covariance matrix $\hat{\sigma}^2 (X^\top X)^{-1}$ is invalid. The HC3 estimator replaces it with

$$
\widehat{\text{Cov}}_{\text{HC3}}(\hat{\boldsymbol{\beta}}) = (X^\top X)^{-1} \left(\sum_{i=1}^{n} \frac{\hat{e}_i^2}{(1 - h_{ii})^2} \mathbf{x}_i \mathbf{x}_i^\top \right) (X^\top X)^{-1}
$$

where $h_{ii}$ is the $i$-th diagonal element of the hat matrix $H = X(X^\top X)^{-1}X^\top$ and $\hat{e}_i$ is the $i$-th OLS residual. This estimator is consistent under heteroscedasticity and has better small-sample performance than HC0 or HC1.

## Wald F-Tests for Each Term

To test a main effect or interaction, we formulate a joint linear hypothesis $R\boldsymbol{\beta} = \mathbf{0}$ where $R$ selects the rows corresponding to that term's coefficients. The Wald $F$-statistic is

$$
F_W = \frac{1}{q} (R\hat{\boldsymbol{\beta}})^\top \bigl(R\, \widehat{\text{Cov}}_{\text{HC3}}(\hat{\boldsymbol{\beta}})\, R^\top\bigr)^{-1} (R\hat{\boldsymbol{\beta}})
$$

where $q$ is the number of restrictions (rows of $R$). Under $H_0$, $F_W$ is approximately $F_{q, \nu}$ where $\nu$ is an adjusted denominator degrees of freedom.

## Code Example

```python
import pandas as pd
from statsmodels.formula.api import ols

data = {
    "Temperature": ["High"]*3 + ["Low"]*3 + ["Medium"]*3,
    "Fertilizer":  ["A","B","C","A","B","C","A","B","C"],
    "Growth":      [12, 15, 14, 10, 13, 11, 14, 16, 15],
}
df = pd.DataFrame(data)

model = ols("Growth ~ C(Temperature) * C(Fertilizer)", data=df).fit()
rob = model.get_robustcov_results(cov_type="HC3")

# Test main effect of Temperature
pnames = model.params.index.tolist()
temp_params = [p for p in pnames if p.startswith("C(Temperature)[")]
constraint = ", ".join([f"{t} = 0" for t in temp_params])
print("Main effect: Temperature")
print(rob.f_test(constraint))

# Test interaction
inter_params = [p for p in pnames if ":" in p]
constraint_inter = ", ".join([f"{t} = 0" for t in inter_params])
print("Interaction: Temperature x Fertilizer")
print(rob.f_test(constraint_inter))
```

## Comparison with Standard ANOVA

The standard (non-robust) ANOVA table can be obtained for reference:

```python
import statsmodels.api as sm

print(sm.stats.anova_lm(model, typ=2))
```

When variances are equal, the HC3 Wald tests and the standard ANOVA give similar results. Discrepancies indicate that heteroscedasticity is affecting the standard tests.

## Interpretation

- **HC3 vs. HC0:** HC3 divides each squared residual by $(1 - h_{ii})^2$ rather than leaving it unscaled (HC0). This upward adjustment corrects for the tendency of high-leverage points to have smaller residuals, improving coverage in small samples.
- **When to use this approach:** Whenever a formal test (Levene, Bartlett) or visual inspection (residual plots) suggests unequal variances, the HC3-based Wald test is preferred over the standard $F$-test.
- **Limitations:** With very small cell sizes (as in the example above with $n = 1$ per cell), the HC3 estimator may be poorly behaved because individual leverage values $h_{ii}$ can be close to 1. Larger cell sizes improve the reliability of the robust estimator.

## Exercises

**Exercise 1.**
Explain why OLS coefficient estimates remain unbiased under heteroscedasticity, even though the standard errors are incorrect. What property of OLS is being used?

??? success "Solution to Exercise 1"
    The OLS estimator is $\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top \mathbf{y}$. Taking expectations:

    $$
    E[\hat{\boldsymbol{\beta}}] = (X^\top X)^{-1} X^\top E[\mathbf{y}] = (X^\top X)^{-1} X^\top X \boldsymbol{\beta} = \boldsymbol{\beta}
    $$

    This derivation uses only the linearity of the estimator and the assumption that $E[\mathbf{y}] = X\boldsymbol{\beta}$ (correct specification of the conditional mean). It does not use the homoscedasticity assumption. Therefore, even when $\text{Var}(\varepsilon_i) = \sigma_i^2$ varies across observations, the OLS estimates are unbiased. The Gauss-Markov theorem guarantees that OLS is BLUE (Best Linear Unbiased Estimator) only under homoscedasticity, so OLS is still unbiased without it, but no longer efficient.

---

**Exercise 2.**
Write out the restriction matrix $R$ for testing the main effect of Temperature in a model with levels High, Low, Medium (with Low as the reference). How many rows does $R$ have?

??? success "Solution to Exercise 2"
    With Low as the reference level, the model includes indicator coefficients for Temperature[T.High] and Temperature[T.Medium]. Testing the main effect of Temperature means testing

    $$
    H_0: \beta_{\text{T.High}} = 0 \text{ and } \beta_{\text{T.Medium}} = 0
    $$

    The restriction matrix selects these two coefficients from the full parameter vector $\boldsymbol{\beta} = (\beta_0, \beta_{\text{T.High}}, \beta_{\text{T.Medium}}, \beta_{\text{F.B}}, \beta_{\text{F.C}}, \ldots)^\top$. If Temperature[T.High] is the 2nd parameter and Temperature[T.Medium] is the 3rd:

    $$
    R = \begin{pmatrix} 0 & 1 & 0 & 0 & \cdots & 0 \\ 0 & 0 & 1 & 0 & \cdots & 0 \end{pmatrix}
    $$

    The matrix $R$ has $q = a - 1 = 2$ rows (one for each non-reference level), corresponding to 2 degrees of freedom for the main effect.

---

**Exercise 3.**
The HC3 estimator divides by $(1 - h_{ii})^2$ while HC2 divides by $(1 - h_{ii})$. Explain the intuition behind the $(1 - h_{ii})^2$ correction and why it improves small-sample performance.

??? success "Solution to Exercise 3"
    The OLS residual $\hat{e}_i = y_i - \hat{y}_i = (1 - h_{ii})\varepsilon_i + \text{terms involving other } \varepsilon_j$. Therefore $E[\hat{e}_i^2] \approx (1 - h_{ii})^2 \sigma_i^2$ (ignoring cross terms), which means $\hat{e}_i^2$ systematically underestimates $\sigma_i^2$ by a factor of $(1 - h_{ii})^2$.

    - **HC0** uses $\hat{e}_i^2$ directly, which is biased downward.
    - **HC2** corrects by dividing by $(1 - h_{ii})$, which gives $E[\hat{e}_i^2 / (1 - h_{ii})] \approx (1 - h_{ii})\sigma_i^2$, still biased.
    - **HC3** divides by $(1 - h_{ii})^2$, so $\hat{e}_i^2 / (1 - h_{ii})^2 \approx \sigma_i^2$, producing a nearly unbiased estimate of $\sigma_i^2$.

    High-leverage points (large $h_{ii}$) have the most severely attenuated residuals. HC3's stronger correction ensures these influential observations contribute appropriately to the variance estimate, which is especially important in small samples where a few points can have substantial leverage.

---

**Exercise 4.**
In the example code, the design has only $n = 1$ observation per cell (a $3 \times 3$ design with 9 observations and 9 parameters). Explain why the HC3 estimator is problematic in this case and suggest a minimum cell size for reliable inference.

??? success "Solution to Exercise 4"
    With $n = 1$ per cell and 9 parameters fit to 9 observations, the hat matrix is $H = I$ (identity), so $h_{ii} = 1$ for every observation. The HC3 divisor $(1 - h_{ii})^2 = 0$, making the estimator undefined (division by zero).

    Even when $h_{ii}$ is close to but not exactly 1, the HC3 estimates become extremely large and unstable. As a general rule, HC3 requires enough residual degrees of freedom for the squared residuals to provide meaningful variance estimates.

    A common recommendation is at least $n = 3$ to $5$ observations per cell for HC3 to work reliably. With $n \ge 5$ per cell in a $3 \times 3$ design ($N = 45$, $p = 9$), the maximum leverage is bounded well below 1, and the HC3 estimator is well-behaved.

---

**Exercise 5.**
Prove that the HC3 sandwich estimator is consistent for $\text{Var}(\hat{\boldsymbol{\beta}})$ under heteroscedasticity. That is, show that as $n \to \infty$, it converges to the true variance of the OLS estimator.

??? success "Solution to Exercise 5"
    The true variance of the OLS estimator under heteroscedasticity is

    $$
    \text{Var}(\hat{\boldsymbol{\beta}}) = (X^\top X)^{-1} X^\top \Omega\, X\, (X^\top X)^{-1}
    $$

    where $\Omega = \text{diag}(\sigma_1^2, \ldots, \sigma_n^2)$. The HC3 estimator replaces $\Omega$ with $\hat{\Omega}_{\text{HC3}} = \text{diag}(\hat{e}_i^2 / (1 - h_{ii})^2)$.

    As $n \to \infty$, by regularity conditions: (1) each leverage $h_{ii} \to 0$ because $h_{ii} \le p/n \to 0$, so $(1 - h_{ii})^2 \to 1$; (2) by consistency of OLS, $\hat{e}_i^2 \to \varepsilon_i^2$ for each $i$; (3) by the law of large numbers, the sample average $(1/n) X^\top \hat{\Omega}_{\text{HC3}} X \to (1/n) X^\top \Omega\, X$.

    Therefore $\widehat{\text{Cov}}_{\text{HC3}}(\hat{\boldsymbol{\beta}}) \to \text{Var}(\hat{\boldsymbol{\beta}})$ in probability. The HC3 correction is a finite-sample improvement over HC0 (which also has this asymptotic property) but converges to the same limit as $n \to \infty$. $\square$
