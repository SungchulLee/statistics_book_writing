# Non-Parametric vs Parametric vs Semi-Parametric

This chapter has introduced three families of survival models: non-parametric
methods (Kaplan--Meier, Nelson--Aalen, log-rank test), fully parametric models
(exponential, Weibull, log-normal, log-logistic), and the semi-parametric Cox
model.  Each family makes different assumptions, offers different outputs, and
is suited to different analytical goals.

This section compares the three paradigms to help the analyst choose the
appropriate approach for a given problem.

## The Three Paradigms

### Non-Parametric Methods

Non-parametric methods make **no distributional assumptions** about the survival
times.  The Kaplan--Meier estimator produces a step-function estimate of the
survival curve, and the log-rank test compares curves between groups.

**Strengths:**

- No risk of distributional misspecification.
- Easy to compute and interpret.
- The Kaplan--Meier curve provides a model-free visualization of survival.

**Limitations:**

- Cannot incorporate continuous covariates (only stratification by groups).
- Estimates are step functions---no smooth survival or hazard curves.
- Efficiency is lower than correctly specified parametric models.

### Fully Parametric Models

Parametric models specify the complete distribution of event times through a
finite-dimensional parameter vector (e.g., $\lambda$ for exponential,
$(k, \lambda)$ for Weibull).

**Strengths:**

- Smooth, interpretable survival and hazard functions.
- More efficient (smaller standard errors) than non-parametric methods when
  the distributional assumption is correct.
- Extrapolation beyond the observed time range is possible (with caution).
- The likelihood framework supports AIC, BIC, and likelihood ratio tests for
  model comparison.

**Limitations:**

- Biased estimates if the distributional assumption is wrong.
- Limited to the hazard shapes that the chosen family can produce (e.g.,
  Weibull cannot capture non-monotone hazards).

### Semi-Parametric (Cox) Model

The Cox model specifies the effect of covariates on the hazard without assuming
a parametric form for the baseline hazard.

**Strengths:**

- Robust to misspecification of the baseline hazard.
- Naturally incorporates multiple covariates, including continuous variables.
- Hazard ratios have a direct, interpretable meaning.
- Partial likelihood estimation avoids the need to estimate $h_0(t)$.

**Limitations:**

- Cannot directly estimate the baseline hazard (only via the Breslow estimator
  after fitting).
- The proportional hazards assumption must hold; violations invalidate the
  standard interpretation.
- Less efficient than a correctly specified parametric model.

## Comparison Table

| Feature | Non-Parametric | Parametric | Cox (Semi-Parametric) |
|:--------|:--------------:|:----------:|:---------------------:|
| Distributional assumption | None | Full | None for baseline |
| Covariate adjustment | No (groups only) | Yes | Yes |
| Smooth hazard/survival | No | Yes | Only via Breslow |
| Efficiency (correct spec.) | Lowest | Highest | Middle |
| Robustness | Highest | Lowest | High |
| Extrapolation | No | Yes (with caution) | No |
| PH assumption | Not required | Depends on model | Required |
| Model comparison (AIC) | Not applicable | Yes | Not directly |

## Decision Guidelines

The choice among the three paradigms depends on the analytical goal and the
available information.

### Use Non-Parametric Methods When

- The goal is **exploratory**: visualize the survival experience and compare
  groups before committing to a model.
- **No covariates** need to be adjusted for, or the analyst wants a
  model-free benchmark.
- The sample is small and distributional assumptions are suspect.

### Use Parametric Models When

- A plausible distributional family is available (informed by subject-matter
  knowledge or graphical checks).
- **Smooth estimates** of the hazard or survival function are needed.
- The goal is **prediction or extrapolation** beyond the observed time range.
- Model comparison via AIC or likelihood ratio tests is desired.

### Use the Cox Model When

- **Multiple covariates** affect the hazard and the analyst wants to estimate
  their effects simultaneously.
- The baseline hazard shape is unknown or unimportant.
- The proportional hazards assumption is reasonable (verified by Schoenfeld
  residuals).
- Hazard ratios are the primary inferential target.

!!! tip "A Practical Workflow"

    1. Start with the Kaplan--Meier estimator to visualize the data and use
       the log-rank test for preliminary group comparisons.
    2. Examine the Nelson--Aalen cumulative hazard plot to assess the hazard
       shape (constant, monotone, non-monotone).
    3. If covariates are important, fit a Cox model and check the PH
       assumption.
    4. If a parametric form is suggested by the data, fit parametric models
       and compare via AIC.
    5. Report the Kaplan--Meier curve alongside the model-based results for
       transparency.

## Combining Approaches

The three paradigms are not mutually exclusive.  A thorough survival analysis
often uses all three:

- The Kaplan--Meier curve provides a non-parametric benchmark.
- Parametric models quantify the hazard shape and enable prediction.
- The Cox model estimates covariate effects without distributional assumptions.

Comparing the Kaplan--Meier curve with the parametric and Cox-based survival
curves serves as an informal goodness-of-fit check: large discrepancies
between the non-parametric and model-based curves indicate potential
misspecification.

!!! note "No Single Best Method"

    The best approach depends on the data, the research question, and the
    assumptions that can be defended.  When in doubt, the Cox model is a
    reasonable default because it avoids distributional assumptions while
    accommodating covariates.  However, it should always be supplemented with
    a Kaplan--Meier analysis and, when feasible, a parametric analysis.


## Exercises

**Exercise 1.**
Describe the main concept of Non-Parametric vs Parametric vs Semi-Parametric and explain why it matters for statistical practice.

??? success "Solution to Exercise 1"
    Non-Parametric vs Parametric vs Semi-Parametric is a core topic in statistics that provides tools for drawing reliable inferences from data. It matters because proper application ensures valid conclusions, correctly quantified uncertainty, and appropriate handling of the assumptions that underpin the method. Practitioners who understand this concept can avoid common pitfalls and choose the right analytical approach for their data.

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
