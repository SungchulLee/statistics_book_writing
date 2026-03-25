# Proportional Hazards Assumption

The Cox model assumes that the hazard ratio between any two covariate profiles
is **constant over time**.  This is the **proportional hazards (PH) assumption**.
If the assumption fails---for example, a treatment that is protective early but
harmful late---the estimated hazard ratios are time-averaged summaries that may
not represent the covariate effect at any particular point in time.

This section explains the PH assumption formally, presents graphical and
statistical methods for checking it, and discusses remedies when it is violated.

## Formal Statement

The Cox model specifies

$$
h(t \mid \mathbf{x}) = h_0(t) \exp(\boldsymbol{\beta}^\top \mathbf{x})
$$

The PH assumption requires that for any two subjects with covariate vectors
$\mathbf{x}_1$ and $\mathbf{x}_2$,

$$
\frac{h(t \mid \mathbf{x}_1)}{h(t \mid \mathbf{x}_2)} = \exp\!\bigl(\boldsymbol{\beta}^\top (\mathbf{x}_1 - \mathbf{x}_2)\bigr)
$$

for **all** $t \geq 0$.  The right-hand side does not depend on $t$; if it
did, the proportional hazards assumption would be violated.

## Graphical Methods

### Log-Log Survival Plot

Under the Cox model, the cumulative hazard for a subject with covariates
$\mathbf{x}$ is

$$
H(t \mid \mathbf{x}) = H_0(t) \exp(\boldsymbol{\beta}^\top \mathbf{x})
$$

Taking the log of the negative log survival:

$$
\ln\!\bigl(-\ln S(t \mid \mathbf{x})\bigr) = \ln H_0(t) + \boldsymbol{\beta}^\top \mathbf{x}
$$

For two groups (e.g., treatment vs control), the curves
$\ln(-\ln \hat{S}_1(t))$ and $\ln(-\ln \hat{S}_2(t))$ should be **parallel**
over time if PH holds.  The vertical distance between the curves equals
$\boldsymbol{\beta}^\top (\mathbf{x}_1 - \mathbf{x}_2)$, which is constant.

!!! tip "How to Read the Log-Log Plot"

    Parallel curves support the PH assumption.  Curves that converge, diverge,
    or cross indicate non-proportional hazards.  Crossing curves are the
    strongest evidence of a PH violation.

### Observed vs Expected Survival Plots

Compare the Kaplan--Meier curves for each group with the Cox model's predicted
survival curves.  Systematic deviations suggest the model is misspecified.

## Schoenfeld Test

The formal statistical test for the PH assumption is based on **Schoenfeld
residuals** (Grambsch and Therneau, 1994).

For each event at time $t_{(j)}$ and each covariate $x_k$, the Schoenfeld
residual is

$$
r_{jk} = x_{i_j, k} - \bar{x}_{jk}(\hat{\boldsymbol{\beta}})
$$

where $x_{i_j, k}$ is the value of covariate $k$ for the subject who
experienced the event, and $\bar{x}_{jk}$ is the risk-set-weighted mean of
covariate $k$ at time $t_{(j)}$.

Under the PH assumption, the Schoenfeld residuals have **no systematic trend
over time**.  The test procedure is:

1. Compute the scaled Schoenfeld residuals for each covariate.
2. Regress the scaled residuals on a function of time (typically $t$, $\ln t$,
   or the Kaplan--Meier transform of $t$).
3. Test whether the slope is significantly different from zero.

The test statistic for covariate $k$ is

$$
\chi^2_k = \frac{\left(\sum_{j} g(t_{(j)}) \, r_{jk}^*\right)^2}{\text{Var}\left(\sum_{j} g(t_{(j)}) \, r_{jk}^*\right)}
$$

where $r_{jk}^*$ are the scaled Schoenfeld residuals and $g(t)$ is the chosen
time function.  Under $H_0$ (PH holds), $\chi^2_k \sim \chi^2_1$.

A **global test** that simultaneously tests PH for all covariates is also
available, with $p$ degrees of freedom (one per covariate).

!!! example "Schoenfeld Test Output"

    A Cox model with two covariates (age and treatment) yields:

    | Covariate | $\chi^2$ | p-value |
    |:----------|:--------:|:-------:|
    | Age | 0.82 | 0.365 |
    | Treatment | 7.45 | 0.006 |
    | GLOBAL | 8.31 | 0.016 |

    The PH assumption holds for age but is violated for treatment.  The
    treatment effect changes over time.

## Remedies for PH Violations

When the PH assumption fails for a covariate, several strategies are available:

### 1. Stratification

Split the baseline hazard by the offending covariate.  For a binary variable
$z$ with two levels, the stratified Cox model is

$$
h(t \mid \mathbf{x}, z=g) = h_{0g}(t) \exp(\boldsymbol{\beta}^\top \mathbf{x})
$$

Each stratum $g$ has its own baseline hazard, but the covariate effects
$\boldsymbol{\beta}$ are shared.  Stratification allows the hazard shape to
differ across groups without requiring a separate model.

### 2. Time-Varying Coefficients

Replace the constant $\beta_k$ with a time-dependent coefficient $\beta_k(t)$:

$$
h(t \mid \mathbf{x}) = h_0(t) \exp\!\left(\sum_{k} \beta_k(t) \, x_k\right)
$$

A simple approach is to include an interaction between the covariate and a
function of time (e.g., $x_k \cdot \ln t$):

$$
h(t \mid \mathbf{x}) = h_0(t) \exp(\beta_k x_k + \gamma_k x_k \ln t)
$$

If $\gamma_k = 0$, the standard Cox model is recovered.

### 3. Piecewise Cox Model

Split the time axis into intervals and fit separate Cox models (or estimate
separate coefficients) within each interval.  This allows the hazard ratio to
change across predefined time windows.

!!! warning "Do Not Ignore PH Violations"

    When the PH assumption is violated, the reported hazard ratios are
    misleading because they represent a time-averaged effect that may not
    hold at any specific time.  Always check the assumption before
    interpreting Cox model results.

## Summary

| Method | Type | What It Assesses |
|:-------|:-----|:-----------------|
| Log-log plot | Graphical | Parallel curves imply PH |
| Schoenfeld test | Statistical | Correlation of residuals with time |
| Stratification | Remedy | Allows different baseline hazards |
| Time-varying coefficients | Remedy | Allows $\beta(t)$ to change with time |
| Piecewise model | Remedy | Different $\beta$ in each time interval |
