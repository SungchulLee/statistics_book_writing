# Independence Assumption


## Definition

The independence assumption states that the residuals (errors) of the regression model are independent of each other. In other words, the error for one observation should not be related to the error for another observation.

Formally, for any two distinct observations $i$ and $j$:

$$
\text{Cov}(\epsilon_i, \epsilon_j) = 0 \quad \text{for all } i \neq j
$$

## Importance

Violating the independence assumption can lead to:

- **Biased estimates** of regression coefficients.
- **Incorrect standard errors** — typically underestimated, making coefficients appear more significant than they truly are.
- **Invalid hypothesis tests** — t-tests and F-tests produce misleading p-values.

Independence is particularly critical in **time-series data**, where errors may be correlated over time (known as **autocorrelation**). It is also relevant in **clustered data**, where observations within groups may be more similar to each other than to observations in other groups.

## Common Violations

| Data Type | Common Violation | Example |
|-----------|-----------------|---------|
| Time-series | Autocorrelation | Stock returns, economic indicators |
| Spatial | Spatial correlation | Geographic data, environmental measurements |
| Clustered | Within-cluster correlation | Students within schools, patients within hospitals |
| Repeated measures | Serial correlation | Longitudinal studies, panel data |

## Diagnostics

- **Durbin-Watson Test:** A statistical test that detects first-order autocorrelation. A value close to 2 suggests no autocorrelation, while values significantly less than 2 indicate positive autocorrelation, and values significantly greater than 2 indicate negative autocorrelation.
- **Residual Plots:** In time-series data, plot residuals against time to check for patterns. Any discernible pattern might indicate a violation of the independence assumption.
- **Breusch-Godfrey Test:** Detects higher-order autocorrelation beyond just the first lag.

## Remedies for Non-Independence

- **Incorporate Lagged Variables:** In time-series models, including lagged variables (e.g., $Y_{t-1}$) can help account for autocorrelation.
- **Generalized Least Squares (GLS):** GLS can be used to address autocorrelation by adjusting the standard errors of the regression coefficients.
- **Mixed-Effects Models:** For clustered data, mixed-effects (hierarchical) models can account for within-cluster correlation.
- **Newey-West Standard Errors:** Heteroscedasticity and autocorrelation consistent (HAC) standard errors provide valid inference even when independence is violated.

For detailed diagnostic methods, see [Checking Independence](checking_independence.md).
## Exercises

**Exercise 1.**
State the mathematical definition of the independence assumption for regression residuals. Give one example of a dataset where this assumption is likely violated.

??? success "Solution to Exercise 1"
    The independence assumption requires:

    $$
    \text{Cov}(\varepsilon_i, \varepsilon_j) = 0 \quad \text{for all } i \neq j
    $$

    **Example:** Monthly stock returns regressed on market factors. Returns in adjacent months are likely autocorrelated because economic conditions persist over time, causing $\text{Cov}(\varepsilon_t, \varepsilon_{t+1}) \neq 0$.

---

**Exercise 2.**
Explain why violating the independence assumption is considered more serious than violating normality or homoscedasticity. What makes it harder to fix?

??? success "Solution to Exercise 2"
    Independence violations are more serious because they affect the **core structure** of the statistical inference:

    - Standard errors become biased (not just inefficient), making all $t$-tests and $F$-tests invalid.
    - The effective sample size is smaller than the nominal $n$, so confidence intervals are misleadingly narrow.
    - Unlike heteroscedasticity (fixable with robust SEs) or non-normality (mitigated by CLT), independence violations require fundamentally different models (mixed-effects, time-series models, GEE) that account for the correlation structure.
