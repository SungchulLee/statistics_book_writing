# Cox Proportional Hazards

## Overview

The Cox proportional hazards model is the most widely used regression framework in
survival analysis.  It relates covariate effects to the hazard function without
specifying a parametric form for the baseline hazard, making it a semi-parametric
approach.  This page presents the model formulation, derives the partial likelihood,
discusses hazard ratio interpretation, and covers diagnostic checks for the
proportional hazards assumption.

## Model Formulation

The Cox model specifies the hazard for subject $i$ with covariate vector
$\mathbf{x}_i = (x_{i1}, \ldots, x_{ip})^\top$ as

$$
h(t \mid \mathbf{x}_i) = h_0(t) \exp(\boldsymbol{\beta}^\top \mathbf{x}_i)
$$

where:

- $h_0(t)$ is the **baseline hazard** --- an arbitrary non-negative function left
  completely unspecified.
- $\boldsymbol{\beta} = (\beta_1, \ldots, \beta_p)^\top$ are the regression
  coefficients to be estimated.
- $\exp(\boldsymbol{\beta}^\top \mathbf{x}_i)$ is the relative risk multiplier for
  subject $i$.

The **proportional hazards property** follows immediately: the hazard ratio between
any two subjects is constant over time:

$$
\frac{h(t \mid \mathbf{x}_i)}{h(t \mid \mathbf{x}_j)} = \exp\!\bigl(\boldsymbol{\beta}^\top(\mathbf{x}_i - \mathbf{x}_j)\bigr)
$$

The baseline hazard $h_0(t)$ cancels in the ratio.

## Partial Likelihood

Let $t_{(1)} < \cdots < t_{(K)}$ be the $K$ distinct ordered event times, and let
$i_j$ denote the subject experiencing the event at $t_{(j)}$.  The **risk set** at
$t_{(j)}$ is

$$
\mathcal{R}_j = \{i : t_i \geq t_{(j)}\}
$$

The partial likelihood is

$$
PL(\boldsymbol{\beta}) = \prod_{j=1}^{K} \frac{\exp(\boldsymbol{\beta}^\top \mathbf{x}_{i_j})}{\sum_{l \in \mathcal{R}_j} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)}
$$

The baseline hazard $h_0(t_{(j)})$ appears in both numerator and denominator and
cancels.  This is the key insight of Cox (1972): covariate effects can be estimated
without knowing $h_0(t)$.

The partial log-likelihood is

$$
\ell_P(\boldsymbol{\beta}) = \sum_{j=1}^{K} \left[\boldsymbol{\beta}^\top \mathbf{x}_{i_j} - \ln\!\left(\sum_{l \in \mathcal{R}_j} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)\right)\right]
$$

### Estimation

The MLE $\hat{\boldsymbol{\beta}}$ is obtained by maximizing $\ell_P$ using
Newton-Raphson iteration:

$$
\boldsymbol{\beta}^{(m+1)} = \boldsymbol{\beta}^{(m)} + \mathcal{I}(\boldsymbol{\beta}^{(m)})^{-1} U(\boldsymbol{\beta}^{(m)})
$$

where $U(\boldsymbol{\beta})$ is the score vector and $\mathcal{I}(\boldsymbol{\beta})$
is the observed information matrix.

## Hazard Ratios

The exponentiated coefficient $\exp(\hat{\beta}_j)$ is the **hazard ratio** for a
one-unit increase in covariate $x_j$, holding all other covariates fixed:

$$
\text{HR}_j = \exp(\hat{\beta}_j)
$$

| $\text{HR}$ | Interpretation |
|:------------:|:---------------|
| $> 1$ | Higher hazard (shorter survival) |
| $= 1$ | No effect |
| $< 1$ | Lower hazard (longer survival) |

For a $c$-unit increase in a continuous covariate, the hazard ratio is
$\exp(c \cdot \hat{\beta}_j)$.

### Confidence Interval

A $100(1 - \alpha)\%$ confidence interval for the hazard ratio is

$$
\text{CI}_{\text{HR}} = \bigl(\exp(\hat{\beta}_j - z_{\alpha/2} \cdot \text{se}(\hat{\beta}_j)),\; \exp(\hat{\beta}_j + z_{\alpha/2} \cdot \text{se}(\hat{\beta}_j))\bigr)
$$

If the interval excludes 1, the covariate effect is statistically significant.

## Breslow Estimator of the Baseline Hazard

After estimating $\hat{\boldsymbol{\beta}}$, the baseline cumulative hazard is
estimated by

$$
\hat{H}_0(t) = \sum_{j:\, t_{(j)} \leq t} \frac{d_j}{\sum_{l \in \mathcal{R}_j} \exp(\hat{\boldsymbol{\beta}}^\top \mathbf{x}_l)}
$$

The subject-specific survival function is then

$$
\hat{S}(t \mid \mathbf{x}) = \exp\!\bigl(-\hat{H}_0(t)\bigr)^{\exp(\hat{\boldsymbol{\beta}}^\top \mathbf{x})}
$$

## Checking the Proportional Hazards Assumption

The proportional hazards (PH) assumption is critical for the validity of the Cox model.
If violated, the hazard ratio is not constant over time and the standard interpretation
breaks down.

### Graphical Methods

- **Log-log plot**: Plot $\ln(-\ln \hat{S}(t))$ versus $\ln t$ for each group.  Under
  PH, the curves should be approximately parallel.
- **Schoenfeld residuals**: Plot scaled Schoenfeld residuals against time for each
  covariate.  A non-zero slope suggests time-varying effects.

### Formal Test

The Grambsch-Therneau test regresses scaled Schoenfeld residuals on a function of
time.  A significant slope for covariate $j$ indicates that $\beta_j$ changes over
time.

!!! warning "Consequences of PH Violation"

    When proportional hazards do not hold, the estimated hazard ratio is a
    time-averaged summary that may not accurately represent the effect at any
    particular time.  Remedies include stratification, time-varying coefficients,
    or switching to a parametric accelerated failure time model.

## Implementation Sketch

```python
import numpy as np

def partial_log_likelihood(beta, X, times, events):
    """
    Compute the partial log-likelihood for the Cox model.

    Parameters
    ----------
    beta   : 1-d array   Coefficient vector (length p).
    X      : 2-d array   Covariate matrix (n x p).
    times  : 1-d array   Observed times.
    events : 1-d array   Event indicators (1 = event, 0 = censored).

    Returns
    -------
    ll : float   Partial log-likelihood value.
    """
    risk_scores = X @ beta                # linear predictor for each subject
    exp_scores = np.exp(risk_scores)

    order = np.argsort(-times)            # sort by decreasing time
    sorted_events = events[order]
    sorted_exp = exp_scores[order]
    sorted_scores = risk_scores[order]

    cumsum_exp = np.cumsum(sorted_exp)    # cumulative risk set sum

    ll = np.sum(sorted_events * (sorted_scores - np.log(cumsum_exp)))
    return ll
```

This implementation sorts subjects by decreasing time so that a cumulative sum
efficiently computes the denominator of the partial likelihood at each event time.

!!! note "Production Use"

    For real analyses, use established libraries such as `lifelines` or `scikit-survival`
    that handle ties (Breslow/Efron methods), compute standard errors, and provide
    diagnostic tools.

## Interpretation

- The Cox model estimates **relative effects** of covariates on the hazard, not absolute
  risk levels.
- Hazard ratios quantify multiplicative changes in the instantaneous event rate, not
  in cumulative probabilities.
- The baseline hazard is a nuisance parameter; the Breslow estimator recovers it when
  absolute survival predictions are needed.
- Always verify the PH assumption before interpreting hazard ratios as time-constant
  effects.

## Exercises

**Exercise 1.**
Partial Likelihood Construction

Three subjects have the following data:

| Subject | Time | Event | $x$ |
|:-------:|:----:|:-----:|:---:|
| A | 2 | 1 | 0.5 |
| B | 3 | 0 | 1.2 |
| C | 5 | 1 | 0.8 |

Write the partial likelihood $PL(\beta)$ for this dataset.

??? success "Solution to Exercise 1"

    Event times: $t_{(1)} = 2$ (subject A) and $t_{(2)} = 5$ (subject C).

    At $t_{(1)} = 2$: risk set $\mathcal{R}_1 = \{A, B, C\}$.

    $$
    \frac{\exp(0.5\beta)}{\exp(0.5\beta) + \exp(1.2\beta) + \exp(0.8\beta)}
    $$

    At $t_{(2)} = 5$: subject A had the event at $t = 2$ and subject B was censored at
    $t = 3$, so $\mathcal{R}_2 = \{C\}$.

    $$
    \frac{\exp(0.8\beta)}{\exp(0.8\beta)} = 1
    $$

    The partial likelihood is:

    $$
    PL(\beta) = \frac{\exp(0.5\beta)}{\exp(0.5\beta) + \exp(1.2\beta) + \exp(0.8\beta)}
    $$

---

**Exercise 2.**
Hazard Ratio Interpretation

A Cox model for time-to-default includes three covariates:

| Covariate | $\hat{\beta}$ | $\text{se}(\hat{\beta})$ |
|:----------|:-------------:|:------------------------:|
| Debt-to-income ratio | 0.42 | 0.10 |
| Credit score (per 100 pts) | $-0.55$ | 0.12 |
| Secured loan (1 = yes) | $-0.30$ | 0.18 |

**(a)** Compute and interpret the hazard ratio for each covariate.

**(b)** Which covariates are significant at the 5% level?

??? success "Solution to Exercise 2"

    **(a)** Hazard ratios:

    - Debt-to-income: $\text{HR} = e^{0.42} = 1.522$.  A one-unit increase in the
      debt-to-income ratio is associated with a 52.2% increase in the default hazard.
    - Credit score: $\text{HR} = e^{-0.55} = 0.577$.  A 100-point increase in credit
      score is associated with a 42.3% reduction in the default hazard.
    - Secured loan: $\text{HR} = e^{-0.30} = 0.741$.  Secured loans have a 25.9%
      lower default hazard compared to unsecured loans.

    **(b)** Wald test: $|z| = |\hat{\beta}| / \text{se}$:

    - Debt-to-income: $|z| = 0.42/0.10 = 4.20 > 1.96$ --- significant.
    - Credit score: $|z| = 0.55/0.12 = 4.58 > 1.96$ --- significant.
    - Secured loan: $|z| = 0.30/0.18 = 1.67 < 1.96$ --- not significant.

    Debt-to-income ratio and credit score are significant at the 5% level.  The
    secured loan indicator is not.

---

**Exercise 3.**
Proportional Hazards Check

An analyst fits a Cox model with a treatment indicator ($x = 1$ for treatment, $x = 0$
for control).  The log-log survival plot shows the two curves crossing at $t = 12$
months.

**(a)** What does this crossing imply about the proportional hazards assumption?

**(b)** Suggest two remedies.

??? success "Solution to Exercise 3"

    **(a)** Crossing of the $\ln(-\ln \hat{S}(t))$ curves indicates that the hazard
    ratio between treatment and control is **not constant** over time.  The
    proportional hazards assumption is violated.  Before $t = 12$, one group has a
    higher hazard; after $t = 12$, the other group does.

    **(b)** Two remedies:

    1. **Stratification**: Fit a stratified Cox model that allows a separate
       baseline hazard for each stratum (e.g., early vs late period) while
       constraining covariate effects to be the same across strata.
    2. **Time-varying coefficient**: Extend the Cox model to allow $\beta(t)$ by
       including an interaction between the treatment indicator and a function of
       time (e.g., $x \cdot \ln t$).

---

**Exercise 4.**
Breslow Estimator

Given the partial likelihood estimate $\hat{\beta} = 0.4$ and the following data:

| $t_{(j)}$ | $d_j$ | Subjects in $\mathcal{R}_j$ | Their $x$ values |
|:----------:|:-----:|:---------------------------:|:-----------------:|
| 3 | 1 | A, B, C | 1, 0, 0.5 |
| 7 | 1 | B, C | 0, 0.5 |

Compute the Breslow estimate $\hat{H}_0(7)$.

??? success "Solution to Exercise 4"

    At $t_{(1)} = 3$: the denominator is

    $$
    \sum_{l \in \mathcal{R}_1} e^{0.4 x_l} = e^{0.4} + e^{0} + e^{0.2} = 1.492 + 1.000 + 1.221 = 3.713
    $$

    Increment: $d_1 / 3.713 = 1/3.713 = 0.269$.

    At $t_{(2)} = 7$: the denominator is

    $$
    \sum_{l \in \mathcal{R}_2} e^{0.4 x_l} = e^{0} + e^{0.2} = 1.000 + 1.221 = 2.221
    $$

    Increment: $d_2 / 2.221 = 1/2.221 = 0.450$.

    $$
    \hat{H}_0(7) = 0.269 + 0.450 = 0.719
    $$

---

**Exercise 5.**
Why the Baseline Hazard Cancels

Prove that the baseline hazard $h_0(t)$ cancels in the conditional probability used to
construct the partial likelihood.  Specifically, show that

$$
\frac{h(t \mid \mathbf{x}_{i_j})}{\sum_{l \in \mathcal{R}_j} h(t \mid \mathbf{x}_l)} = \frac{\exp(\boldsymbol{\beta}^\top \mathbf{x}_{i_j})}{\sum_{l \in \mathcal{R}_j} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)}
$$

??? success "Solution to Exercise 5"

    By the Cox model specification, the hazard for subject $i$ at time $t$ is

    $$
    h(t \mid \mathbf{x}_i) = h_0(t) \exp(\boldsymbol{\beta}^\top \mathbf{x}_i)
    $$

    Substituting into the ratio:

    $$
    \frac{h(t \mid \mathbf{x}_{i_j})}{\sum_{l \in \mathcal{R}_j} h(t \mid \mathbf{x}_l)} = \frac{h_0(t) \exp(\boldsymbol{\beta}^\top \mathbf{x}_{i_j})}{\sum_{l \in \mathcal{R}_j} h_0(t) \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)}
    $$

    Since all terms in the numerator and denominator are evaluated at the same time
    $t$, the factor $h_0(t) > 0$ can be factored out of the sum in the denominator:

    $$
    = \frac{h_0(t) \exp(\boldsymbol{\beta}^\top \mathbf{x}_{i_j})}{h_0(t) \sum_{l \in \mathcal{R}_j} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)} = \frac{\exp(\boldsymbol{\beta}^\top \mathbf{x}_{i_j})}{\sum_{l \in \mathcal{R}_j} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)}
    $$

    The cancellation holds because $h_0(t)$ is a common multiplicative factor at each
    event time.  This is what makes the Cox model semi-parametric: the regression
    coefficients $\boldsymbol{\beta}$ can be estimated without specifying $h_0(t)$.
    $\square$
