# Interpreting Hazard Ratios

The Cox model estimates regression coefficients $\boldsymbol{\beta}$, but these
coefficients are not directly interpretable on a natural scale.  The standard
practice is to exponentiate each coefficient to obtain a **hazard ratio** (HR),
which measures the multiplicative effect of a covariate on the instantaneous
event rate.

This section explains how to interpret hazard ratios for different covariate
types, construct confidence intervals, and avoid common misinterpretations.

## Definition of the Hazard Ratio

Recall the Cox model:

$$
h(t \mid \mathbf{x}) = h_0(t) \exp(\boldsymbol{\beta}^\top \mathbf{x})
$$

For a single covariate $x_j$, holding all other covariates fixed, the hazard
ratio for a one-unit increase in $x_j$ is

$$
\text{HR}_j = \frac{h(t \mid x_j + 1, \mathbf{x}_{-j})}{h(t \mid x_j, \mathbf{x}_{-j})} = \exp(\beta_j)
$$

The baseline hazard $h_0(t)$ cancels, and the ratio does not depend on $t$
(the proportional hazards property).

## Interpretation by Covariate Type

### Binary Covariates

For a binary covariate (e.g., $x = 1$ for treatment, $x = 0$ for control):

$$
\text{HR} = \exp(\hat{\beta})
$$

- $\text{HR} > 1$: the treatment group has a **higher** hazard (shorter
  survival, more events).
- $\text{HR} < 1$: the treatment group has a **lower** hazard (longer
  survival, fewer events).
- $\text{HR} = 1$: no difference between groups.

!!! example "Treatment Effect on Survival"

    A Cox model for post-surgery survival yields $\hat{\beta} = -0.35$ for
    a new drug (coded 1) vs standard care (coded 0).  The hazard ratio is
    $\text{HR} = e^{-0.35} = 0.705$.  Patients receiving the new drug have
    a 29.5% lower hazard of death at any time, holding other covariates
    constant.

### Continuous Covariates

For a continuous covariate (e.g., age in years), $\exp(\hat{\beta})$ is the
hazard ratio for a **one-unit** increase.  This may be too fine a scale for
practical interpretation.

For a $c$-unit increase, the hazard ratio is

$$
\text{HR}_c = \exp(c \cdot \hat{\beta})
$$

!!! example "Age Effect on Default"

    A Cox model for loan default yields $\hat{\beta}_{\text{age}} = 0.03$
    per year.  The HR per year is $e^{0.03} = 1.030$ (a 3.0% increase in
    hazard per year).  The HR per decade is $e^{10 \times 0.03} = e^{0.3} = 1.350$
    (a 35.0% increase in hazard per 10-year increase in borrower age).

### Categorical Covariates with More Than Two Levels

A categorical variable with $G$ levels is encoded as $G - 1$ dummy variables
relative to a reference category.  Each hazard ratio compares one level to the
reference.

!!! example "Industry Effect"

    A Cox model for corporate default includes industry with three levels:
    manufacturing (reference), retail ($\hat{\beta}_1 = 0.45$), and
    technology ($\hat{\beta}_2 = -0.20$).

    - Retail vs manufacturing: $\text{HR} = e^{0.45} = 1.57$.  Retail firms
      default at 1.57 times the rate of manufacturing firms.
    - Technology vs manufacturing: $\text{HR} = e^{-0.20} = 0.82$.
      Technology firms default at 0.82 times the rate of manufacturing firms.

## Confidence Intervals for Hazard Ratios

A $100(1 - \alpha)\%$ confidence interval for $\beta_j$ is

$$
\hat{\beta}_j \pm z_{\alpha/2} \cdot \text{se}(\hat{\beta}_j)
$$

where $\text{se}(\hat{\beta}_j) = \sqrt{[\mathcal{I}^{-1}]_{jj}}$ is obtained
from the inverse of the observed information matrix.

Exponentiating the endpoints gives the confidence interval for the hazard ratio:

$$
\text{CI}_{\text{HR}} = \bigl(\exp(\hat{\beta}_j - z_{\alpha/2} \cdot \text{se}),\; \exp(\hat{\beta}_j + z_{\alpha/2} \cdot \text{se})\bigr)
$$

If the interval excludes 1, the covariate has a statistically significant
effect at the $\alpha$ level.

## Hypothesis Test for a Single Coefficient

The **Wald test** for $H_0: \beta_j = 0$ (equivalently, $\text{HR}_j = 1$) is

$$
z = \frac{\hat{\beta}_j}{\text{se}(\hat{\beta}_j)} \;\xrightarrow{d}\; N(0, 1)
$$

The p-value is $2[1 - \Phi(|z|)]$.  Alternatively, the likelihood ratio test
compares the partial log-likelihood with and without the covariate.

## Common Misinterpretations

!!! warning "Hazard Ratio Is Not a Risk Ratio"

    $\text{HR} = 2$ does **not** mean "twice as likely to experience the
    event."  It means the **instantaneous rate** is twice as high at every
    time point.  The cumulative probability of the event depends on the entire
    hazard trajectory, not just its ratio.

!!! warning "HR Does Not Imply Shorter Median Survival"

    $\text{HR} = 2$ does not mean the median survival is halved.  The
    relationship between the hazard ratio and median survival depends on the
    shape of the baseline hazard.

Additional pitfalls:

- **Conditioning.** The HR is conditional on all other covariates being held
  fixed.  A marginal hazard ratio (not conditioning on other covariates) can
  differ substantially due to confounding.
- **Non-proportional hazards.** If the proportional hazards assumption is
  violated, the estimated HR is a time-averaged summary that may not
  represent the covariate effect at any particular time.

## Summary Table

| Covariate Type | HR Formula | Interpretation |
|:---------------|:-----------|:---------------|
| Binary (0/1) | $e^{\hat{\beta}}$ | Hazard in group 1 relative to group 0 |
| Continuous (1-unit) | $e^{\hat{\beta}}$ | Hazard change per unit increase |
| Continuous ($c$-unit) | $e^{c\hat{\beta}}$ | Hazard change per $c$-unit increase |
| Categorical ($G$ levels) | $e^{\hat{\beta}_g}$ | Level $g$ vs reference level |
