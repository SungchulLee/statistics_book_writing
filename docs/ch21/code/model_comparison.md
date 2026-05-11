# Survival Model Comparison

## Overview

Survival analysis offers three modeling paradigms --- non-parametric (Kaplan-Meier),
fully parametric (exponential, Weibull, log-normal), and semi-parametric (Cox) --- each
with different assumptions, strengths, and outputs.  Choosing the right model for a
given problem requires understanding the trade-offs.  This page provides a systematic
comparison framework, covers formal model selection tools, and illustrates a practical
workflow for survival model selection.

## The Three Paradigms

### Non-Parametric Methods

Non-parametric methods make **no distributional assumptions**.  The Kaplan-Meier
estimator produces a step-function estimate of $S(t)$, and the log-rank test compares
groups.

- **Strengths**: No risk of distributional misspecification; model-free visualization.
- **Limitations**: Cannot adjust for continuous covariates; step-function estimates
  only; lower statistical efficiency.

### Fully Parametric Models

Parametric models specify the complete distribution through a parameter vector (e.g.,
$\lambda$ for exponential, $(k, \lambda)$ for Weibull):

$$
f(t \mid \boldsymbol{\theta}), \quad S(t \mid \boldsymbol{\theta}), \quad h(t \mid \boldsymbol{\theta})
$$

- **Strengths**: Smooth hazard and survival estimates; higher efficiency when correctly
  specified; extrapolation capability; AIC/BIC-based model comparison.
- **Limitations**: Biased if the distributional assumption is wrong; limited hazard
  shapes.

### Semi-Parametric (Cox) Model

The Cox model specifies covariate effects on the hazard without assuming a form for the
baseline:

$$
h(t \mid \mathbf{x}) = h_0(t)\exp(\boldsymbol{\beta}^\top \mathbf{x})
$$

- **Strengths**: Robust to baseline hazard misspecification; naturally handles multiple
  covariates; interpretable hazard ratios.
- **Limitations**: Cannot directly estimate $h_0(t)$; requires the proportional
  hazards assumption; less efficient than a correctly specified parametric model.

## Comparison Table

| Feature | Non-Parametric | Parametric | Cox |
|:--------|:--------------:|:----------:|:---:|
| Distributional assumption | None | Full | None for baseline |
| Covariate adjustment | Groups only | Yes | Yes |
| Smooth hazard/survival | No | Yes | Via Breslow only |
| Statistical efficiency | Lowest | Highest (if correct) | Middle |
| Robustness to misspecification | Highest | Lowest | High |
| Extrapolation | No | Yes (with caution) | No |
| PH assumption required | No | Depends | Yes |
| AIC/BIC comparison | N/A | Yes | Not directly |

## Model Selection Tools

### Information Criteria

For comparing parametric models fitted to the same dataset:

$$
\text{AIC} = -2\hat{\ell} + 2p
$$

$$
\text{BIC} = -2\hat{\ell} + p \ln n
$$

Lower values indicate a better fit-complexity trade-off.

!!! note "AIC vs BIC"

    AIC tends to favor more complex models and is suited for prediction.  BIC penalizes
    complexity more heavily and is consistent for model selection (selects the true
    model as $n \to \infty$ if it is among the candidates).

### Likelihood Ratio Test

For **nested** models (e.g., exponential vs Weibull, where exponential sets $k = 1$):

$$
\Lambda = 2[\hat{\ell}_{\text{full}} - \hat{\ell}_{\text{reduced}}] \;\xrightarrow{d}\; \chi^2_q
$$

where $q$ is the difference in the number of parameters.

### Graphical Diagnostics

Formal criteria should always be supplemented with visual checks:

1. **KM overlay**: Plot the Kaplan-Meier curve alongside each fitted parametric
   survival curve.  Systematic deviations indicate misspecification.
2. **Log-cumulative hazard plot**: $\ln \hat{H}(t)$ vs $\ln t$ should be linear for
   the Weibull model.
3. **Cox-Snell residuals**: If the model is correct, the Cox-Snell residuals should
   follow an exponential(1) distribution.  A plot of the Nelson-Aalen estimate of the
   residual hazard against the residuals should approximate a 45-degree line.

```python
import numpy as np
import matplotlib.pyplot as plt

def cox_snell_diagnostic(residuals):
    """
    Plot Cox-Snell residual diagnostic.

    If the model is correct, the cumulative hazard of the residuals
    should follow a unit exponential, i.e., H(r) = r.
    """
    sorted_r = np.sort(residuals)
    n = len(sorted_r)
    # Nelson-Aalen estimate for the residuals
    H_na = -np.log(1 - np.arange(1, n + 1) / (n + 1))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(sorted_r, H_na, "o", markersize=3, label="Cox-Snell residuals")
    ax.plot([0, sorted_r.max()], [0, sorted_r.max()], "r--", label="45-degree line")
    ax.set_xlabel("Cox-Snell Residuals")
    ax.set_ylabel("Cumulative Hazard")
    ax.legend()
    plt.tight_layout()
    plt.show()
```

## Practical Workflow

A systematic approach to survival model selection:

1. **Explore**: Compute the Kaplan-Meier estimator and Nelson-Aalen cumulative hazard
   for each group.  Visualize the data.
2. **Assess hazard shape**: Inspect the Nelson-Aalen plot.  Is the hazard constant
   (linear $\hat{H}(t)$)?  Monotone?  Non-monotone?
3. **Fit candidates**: Based on the hazard shape, fit appropriate parametric models
   (exponential, Weibull, log-normal, log-logistic) and the Cox model.
4. **Compare**: Use AIC/BIC for non-nested parametric models, LRT for nested models,
   and overlay plots against the Kaplan-Meier curve.
5. **Diagnose**: Check the PH assumption for the Cox model (Schoenfeld residuals) and
   goodness-of-fit for parametric models (Cox-Snell residuals).
6. **Report**: Present the Kaplan-Meier curve as a non-parametric benchmark alongside
   the final model results.

!!! tip "Default Recommendation"

    When in doubt, the Cox model is a reasonable default.  It avoids distributional
    assumptions while accommodating covariates.  However, it should always be
    supplemented with a Kaplan-Meier analysis and, when feasible, a parametric analysis.

## Interpretation

- **No single best method** exists.  The choice depends on the research question, the
  hazard shape, and the assumptions that can be defended.
- **Non-parametric methods** are always appropriate for exploration and provide a
  model-free benchmark.
- **Parametric models** are preferred when smooth estimates or extrapolation are needed
  and the distributional assumption is supported by the data.
- **The Cox model** is preferred when covariate effects are the primary target and the
  baseline hazard shape is unimportant.
- The three paradigms are **complementary**: a thorough survival analysis typically
  employs all three.

## Exercises

**Exercise 1.**
Paradigm Selection

For each scenario, state which modeling paradigm (non-parametric, parametric, or
semi-parametric) is most appropriate and explain why.

**(a)** An exploratory study with 30 patients and no covariates.

**(b)** A clinical trial where the primary goal is to estimate the effect of treatment
on survival, adjusting for age and disease stage.

**(c)** A reliability study where the goal is to predict failure times beyond the
observation period.

??? success "Solution to Exercise 1"

    **(a)** **Non-parametric** (Kaplan-Meier).  With a small sample and no covariates,
    distributional assumptions are risky.  The Kaplan-Meier estimator provides a
    model-free survival curve and the log-rank test can compare groups if needed.

    **(b)** **Semi-parametric** (Cox model).  The goal is to estimate covariate effects
    (treatment, age, disease stage) on the hazard.  The Cox model handles multiple
    covariates without specifying the baseline hazard distribution.

    **(c)** **Parametric** model (e.g., Weibull).  Extrapolation beyond the observed
    time range requires a fully specified distribution.  Non-parametric and Cox models
    cannot extrapolate.

---

**Exercise 2.**
AIC Comparison

Three parametric models are fitted to 150 observations:

| Model | Parameters ($p$) | Max log-likelihood ($\hat{\ell}$) |
|:------|:----------------:|:---------------------------------:|
| Exponential | 1 | $-420.3$ |
| Weibull | 2 | $-405.8$ |
| Log-logistic | 2 | $-407.2$ |

**(a)** Compute the AIC and BIC for each model.

**(b)** Which model is preferred by AIC?  By BIC?

**(c)** Perform a likelihood ratio test of exponential vs Weibull.

??? success "Solution to Exercise 2"

    **(a)** AIC $= -2\hat{\ell} + 2p$; BIC $= -2\hat{\ell} + p\ln n$ ($\ln 150 = 5.011$):

    | Model | AIC | BIC |
    |:------|:---:|:---:|
    | Exponential | $2(420.3) + 2 = 842.6$ | $2(420.3) + 5.011 = 845.6$ |
    | Weibull | $2(405.8) + 4 = 815.6$ | $2(405.8) + 10.022 = 821.6$ |
    | Log-logistic | $2(407.2) + 4 = 818.4$ | $2(407.2) + 10.022 = 824.4$ |

    **(b)** The Weibull model is preferred by both AIC (815.6) and BIC (821.6).

    **(c)** Likelihood ratio test (exponential vs Weibull):

    $$
    \Lambda = 2[-405.8 - (-420.3)] = 2 \times 14.5 = 29.0
    $$

    Under $H_0: k = 1$, $\Lambda \sim \chi^2_1$.  Critical value at $\alpha = 0.05$:
    3.84.  Since $29.0 \gg 3.84$, we reject the exponential model.  The constant
    hazard assumption is not supported.

---

**Exercise 3.**
Cox-Snell Residuals

Explain how Cox-Snell residuals are defined and how they are used to assess overall
model fit.

??? success "Solution to Exercise 3"

    For a fitted parametric model with estimated survival function $\hat{S}(t_i)$, the
    Cox-Snell residual for subject $i$ is

    $$
    r_i = -\ln \hat{S}(t_i) = \hat{H}(t_i)
    $$

    the estimated cumulative hazard at the observed time.  For censored observations,
    $r_i$ is also censored.

    **Key property**: If the model is correctly specified, the Cox-Snell residuals
    follow an $\text{Exp}(1)$ distribution, regardless of the underlying survival
    distribution.  This is because if $T \sim F$, then $-\ln S(T) \sim \text{Exp}(1)$.

    **Diagnostic**: Compute the Nelson-Aalen estimate of the cumulative hazard of the
    residuals.  Plot it against the residuals themselves.  If the model fits well, the
    points should follow the 45-degree line $H(r) = r$.  Deviations indicate
    misspecification.

---

**Exercise 4.**
Efficiency Comparison

**(a)** Explain what is meant by "statistical efficiency" in the context of survival
model comparison.

**(b)** Why is a correctly specified parametric model more efficient than the
Kaplan-Meier estimator?

**(c)** Under what condition does this efficiency advantage become a disadvantage?

??? success "Solution to Exercise 4"

    **(a)** Statistical efficiency refers to the precision of estimates --- a more
    efficient estimator has smaller variance (narrower confidence intervals) for the
    same sample size.

    **(b)** A correctly specified parametric model uses the known distributional form
    to "borrow strength" across all observations.  The Kaplan-Meier estimator treats
    each event time independently without leveraging distributional structure.  The
    parametric model's smaller variance comes from the correct structural assumption,
    which reduces the effective number of quantities being estimated.

    **(c)** The efficiency advantage becomes a disadvantage when the parametric
    assumption is **wrong**.  A misspecified model is not only inefficient but also
    **biased**: it converges to the wrong survival function.  The Kaplan-Meier
    estimator is always consistent, regardless of the true distribution.  Thus,
    efficiency gains from parametric models come at the cost of robustness.

---

**Exercise 5.**
Combining Paradigms

A clinical trial investigates the effect of a new drug on overall survival.  Describe
a complete analysis strategy that uses all three paradigms and explain what each
contributes.

??? success "Solution to Exercise 5"

    **Step 1 --- Non-parametric exploration.**  Compute the Kaplan-Meier survival curves
    for the treatment and control groups.  Visualize the curves with 95% confidence
    bands.  Apply the log-rank test for a preliminary comparison.  Inspect the
    Nelson-Aalen cumulative hazard plot to assess the hazard shape (constant, monotone,
    or non-monotone).

    **Step 2 --- Cox model for covariate adjustment.**  Fit a Cox proportional hazards
    model with treatment indicator, age, sex, and disease stage as covariates.  Estimate
    hazard ratios with confidence intervals.  Check the PH assumption using Schoenfeld
    residuals.

    **Step 3 --- Parametric model for prediction.**  Based on the hazard shape from
    Step 1, fit candidate parametric models (e.g., Weibull if the hazard is monotone).
    Compare via AIC.  Use the best parametric model to extrapolate median survival time
    and produce smooth survival curve estimates.

    **Step 4 --- Integration and reporting.**  Overlay the parametric survival curve and
    the Cox-based Breslow survival estimate on the Kaplan-Meier curve.  Agreement among
    all three methods strengthens confidence in the conclusions.  Discrepancies flag
    potential misspecification.  Report the treatment hazard ratio from the Cox model as
    the primary result, the Kaplan-Meier curve as the model-free benchmark, and the
    parametric model for smooth estimates and predictions.
