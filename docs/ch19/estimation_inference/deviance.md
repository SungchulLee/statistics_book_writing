# Deviance and Goodness-of-Fit

## Motivation

In linear regression, the residual sum of squares measures how well a model
fits the data.  For logistic regression (and generalized linear models more
broadly) the analogous quantity is the **deviance**.  Deviance compares the
log-likelihood of the fitted model to that of a perfect model, producing a
single number that summarizes overall fit.  Differences in deviance between
nested models follow an approximate chi-squared distribution, enabling formal
hypothesis tests.

## The Saturated Model

The **saturated model** assigns one parameter per observation, so it reproduces
the data exactly: $\hat{p}_i^{\text{sat}} = y_i$.  Its log-likelihood is the
largest achievable value.  For binary responses ($y_i \in \{0,1\}$) each term
in the log-likelihood equals $\log 1 = 0$, so

$$
\ell_{\text{sat}} = \sum_{i=1}^{n}\bigl[y_i \log y_i + (1-y_i)\log(1-y_i)\bigr] = 0
$$

where we adopt the convention $0 \log 0 = 0$.

## Definition of Deviance

The **deviance** of a fitted model with predicted probabilities $\hat{p}_i$ is

$$
D = -2\bigl(\ell_{\text{fitted}} - \ell_{\text{sat}}\bigr)
  = -2\,\ell_{\text{fitted}}
$$

Because $\ell_{\text{sat}} = 0$ for binary data, the deviance simplifies to
minus twice the log-likelihood of the fitted model.  Expanding:

$$
D = -2\sum_{i=1}^{n}\bigl[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\bigr]
$$

A smaller deviance indicates a better fit.

## Null Deviance and Residual Deviance

### Null deviance

The **null model** includes only an intercept: $\hat{p}_i = \bar{y}$ for all
$i$.  Its deviance is

$$
D_0 = -2\sum_{i=1}^{n}\bigl[y_i \log \bar{y} + (1-y_i)\log(1-\bar{y})\bigr]
$$

### Residual deviance

The **residual deviance** $D$ is the deviance of the full fitted model
(intercept plus all $p-1$ predictors).  The difference $D_0 - D$ measures
how much the predictors improve the fit beyond the intercept alone.

### Proportion of deviance explained

By analogy with $R^2$ in linear regression:

$$
R^2_{\text{dev}} = 1 - \frac{D}{D_0}
$$

This quantity, sometimes called **McFadden's pseudo-$R^2$**, ranges from 0 to 1.
Values above 0.2 to 0.4 are often considered satisfactory for logistic models.

## Deviance Residuals

The overall deviance decomposes into observation-level contributions called
**deviance residuals**:

$$
d_i = \operatorname{sign}(y_i - \hat{p}_i)\,
\sqrt{-2\bigl[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\bigr]}
$$

so that $D = \sum_{i=1}^{n} d_i^2$.  Deviance residuals are preferred over raw
residuals ($y_i - \hat{p}_i$) because their distribution is closer to standard
normal when the model is correct, making diagnostic plots easier to interpret.

## Goodness-of-Fit Test

Under the null hypothesis that the fitted model is correct and the data contain
no systematic lack of fit, the residual deviance follows approximately

$$
D \;\dot\sim\; \chi^2_{n-p}
$$

where $p$ is the number of estimated parameters.  A large deviance relative
to $n - p$ degrees of freedom signals that the model does not adequately
describe the data.

!!! warning "When the Approximation Fails"
    The chi-squared approximation for the residual deviance requires grouped
    data (multiple observations at each covariate pattern).  For ungrouped
    binary data with continuous predictors, each covariate pattern is unique
    and the approximation breaks down.  In this setting, use the
    Hosmer-Lemeshow test (see [Calibration and Brier Score](../evaluation/calibration.md))
    instead.

## Likelihood Ratio Test via Deviance

To test whether a subset of predictors improves the model, compare a reduced
model (deviance $D_{\text{red}}$, with $p_0$ parameters) to the full model
(deviance $D_{\text{full}}$, with $p_1$ parameters):

$$
\Delta D = D_{\text{red}} - D_{\text{full}} \;\dot\sim\; \chi^2_{p_1 - p_0}
$$

under the null hypothesis that the additional predictors have zero coefficients.
This is the **likelihood ratio test** and is the standard approach for
comparing nested logistic regression models.

??? example "Worked Example"
    Suppose a null model with intercept only gives $D_0 = 120.5$ on 99 degrees
    of freedom.  Adding two predictors ($p_1 = 3$) yields $D = 85.3$ on 97
    degrees of freedom.

    The test statistic is $\Delta D = 120.5 - 85.3 = 35.2$ with
    $p_1 - p_0 = 2$ degrees of freedom.  Comparing to $\chi^2_2$:

    - The critical value at $\alpha = 0.05$ is $5.99$.
    - Since $35.2 \gg 5.99$, we reject the null and conclude the two predictors
      significantly improve the model.

    McFadden's pseudo-$R^2$ is $1 - 85.3/120.5 \approx 0.29$, indicating a
    moderate improvement.

## Summary Table

| Quantity | Formula | Interpretation |
|---|---|---|
| Deviance | $D = -2\,\ell_{\text{fitted}}$ | Overall lack of fit |
| Null deviance | $D_0 = -2\,\ell_{\text{null}}$ | Fit of intercept-only model |
| $\Delta D$ | $D_{\text{red}} - D_{\text{full}}$ | Improvement from added predictors |
| Deviance residual | $d_i = \operatorname{sign}(y_i - \hat{p}_i)\sqrt{-2[\cdots]}$ | Per-observation contribution |
| McFadden $R^2$ | $1 - D/D_0$ | Proportion of deviance explained |


## Exercises

**Exercise 1.**
Describe the main concept of Deviance and Goodness-of-Fit and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Deviance and Goodness-of-Fit is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
