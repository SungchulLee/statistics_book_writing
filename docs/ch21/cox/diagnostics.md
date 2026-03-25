# Model Diagnostics (Schoenfeld Residuals)

Fitting a Cox model is only the first step.  Before interpreting hazard ratios,
the analyst must verify that the model captures the data structure adequately.
Model diagnostics for the Cox model center on three types of residuals, each
designed to detect a different kind of misspecification.

This section defines the main residual types---Schoenfeld, martingale, and
deviance---and explains how to use them for checking the proportional hazards
assumption, detecting non-linearity, and identifying influential observations.

## Schoenfeld Residuals

Schoenfeld residuals were introduced in the previous section as the basis for
the proportional hazards test.  Here we describe their construction and
interpretation in more detail.

### Definition

At each event time $t_{(j)}$, the Schoenfeld residual for covariate $k$ is

$$
r_{jk} = x_{i_j, k} - \bar{x}_{jk}(\hat{\boldsymbol{\beta}})
$$

where $x_{i_j, k}$ is the covariate value for the subject who experienced the
event, and $\bar{x}_{jk}$ is the weighted average of covariate $k$ in the risk
set:

$$
\bar{x}_{jk}(\boldsymbol{\beta}) = \frac{\sum_{l \in \mathcal{R}_j} x_{lk} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)}{\sum_{l \in \mathcal{R}_j} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)}
$$

There is one Schoenfeld residual vector per event (not per subject).  Censored
observations do not generate Schoenfeld residuals.

### Scaled Schoenfeld Residuals

The **scaled Schoenfeld residuals** are defined as

$$
r_{jk}^* = d \cdot [\mathcal{I}^{-1}]_{kk} \cdot r_{jk} + \hat{\beta}_k
$$

where $d$ is the total number of events and $\mathcal{I}$ is the observed
information matrix.  Under the PH assumption, $E[r_{jk}^*] \approx \beta_k$
at every event time.

### Diagnostic Use

Plot the scaled Schoenfeld residuals $r_{jk}^*$ against time (or a function
of time such as $\ln t$).  Under the PH assumption, the plot should show a
**random scatter around a horizontal line** at $\hat{\beta}_k$.

- A positive trend indicates the covariate effect strengthens over time.
- A negative trend indicates the effect weakens over time.
- A U-shaped or inverted-U pattern suggests a non-monotone time-varying effect.

Fit a LOESS smoother to the residual-vs-time plot to visualize the trend.

## Martingale Residuals

Martingale residuals assess the **overall fit** of the model and are
particularly useful for detecting non-linearity in the relationship between a
continuous covariate and the log hazard.

### Definition

The martingale residual for subject $i$ is

$$
\hat{M}_i = \delta_i - \hat{H}_0(t_i) \exp(\hat{\boldsymbol{\beta}}^\top \mathbf{x}_i)
$$

where $\hat{H}_0(t_i)$ is the Breslow estimate of the baseline cumulative
hazard at $t_i$.

**Interpretation:** $\hat{M}_i$ equals the observed number of events for
subject $i$ (either 0 or 1) minus the expected number under the fitted model.

- $\hat{M}_i > 0$: the event occurred sooner than the model predicted.
- $\hat{M}_i < 0$: the subject survived longer than expected (or was censored
  before the predicted event time).

### Properties

- $\sum_{i=1}^{n} \hat{M}_i = 0$ (the residuals sum to zero).
- $\hat{M}_i \in (-\infty, 1]$: bounded above by 1, unbounded below.
- The distribution of martingale residuals is **skewed**, which limits their
  usefulness for standard residual plots.

### Detecting Non-Linearity

To check whether a continuous covariate $x_k$ enters the model with the
correct functional form:

1. Fit the Cox model **without** $x_k$.
2. Compute the martingale residuals from this reduced model.
3. Plot the residuals against $x_k$.
4. Fit a LOESS smoother.

If the smoother deviates systematically from a straight line, the linear
specification of $x_k$ in the Cox model is inadequate.  The shape of the
smoother suggests the correct transformation (e.g., $\ln x_k$, $x_k^2$).

## Deviance Residuals

Deviance residuals are a **symmetrized transformation** of martingale
residuals that produce a more symmetric distribution, making them easier to
interpret in residual plots.

### Definition

$$
\hat{d}_i = \text{sign}(\hat{M}_i) \sqrt{-2\bigl[\hat{M}_i + \delta_i \ln(\delta_i - \hat{M}_i)\bigr]}
$$

where $\text{sign}(\hat{M}_i)$ is $+1$ if $\hat{M}_i > 0$ and $-1$ if
$\hat{M}_i < 0$.

### Diagnostic Use

Plot deviance residuals against the linear predictor
$\hat{\boldsymbol{\beta}}^\top \mathbf{x}_i$ or against individual covariates.
Under a correctly specified model, the residuals should be randomly scattered
around zero with no obvious patterns.

Subjects with large absolute deviance residuals ($|\hat{d}_i| > 2$ or $3$) are
**poorly fit** by the model and warrant investigation.

## Influential Observations

An observation is **influential** if removing it substantially changes the
estimated coefficients.  Influence in the Cox model is measured by:

### Score Residuals

The score residual for subject $i$ is a $p$-dimensional vector

$$
\mathbf{L}_i = \frac{\partial \ell_i}{\partial \boldsymbol{\beta}}
$$

representing the contribution of subject $i$ to the score function.

### dfbeta Residuals

The **dfbeta** for subject $i$ approximates the change in $\hat{\boldsymbol{\beta}}$ when subject $i$ is removed:

$$
\text{dfbeta}_i \approx \mathcal{I}^{-1} \mathbf{L}_i
$$

Plot dfbeta values for each covariate against subject index or time.  Subjects
with disproportionately large dfbeta values are influential and should be
examined for data errors or genuine outlier behavior.

!!! warning "Do Not Automatically Remove Influential Observations"

    An influential observation is not necessarily an error.  It may represent
    a genuine extreme case that carries valuable information.  Investigate the
    subject's characteristics before deciding whether to exclude it.  Conduct
    a sensitivity analysis by comparing results with and without the
    influential observation.

## Diagnostic Summary

| Residual Type | Checks For | One Per |
|:-------------|:-----------|:--------|
| Schoenfeld | PH assumption (time-varying effects) | Event |
| Martingale | Functional form (non-linearity) | Subject |
| Deviance | Overall goodness of fit | Subject |
| Score / dfbeta | Influential observations | Subject |

A thorough Cox model diagnostic analysis examines all four residual types.
Begin with the Schoenfeld test for PH, then check functional form with
martingale residuals, examine overall fit with deviance residuals, and conclude
with an influence analysis.
