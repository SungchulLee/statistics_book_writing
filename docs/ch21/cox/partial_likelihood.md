# Partial Likelihood and the Cox Model

Parametric survival models require the analyst to specify the distribution of
event times.  If the chosen distribution is wrong, inferences about covariate
effects may be biased.  The **Cox proportional hazards model**, introduced by
Cox (1972), avoids this specification entirely.  It models the effect of
covariates on the hazard rate while leaving the **baseline hazard function**
completely unspecified.

This semi-parametric approach is the most widely used regression model in
survival analysis.  This section defines the Cox model, derives the partial
likelihood for estimating covariate effects, and introduces the Breslow
estimator for the baseline cumulative hazard.

## The Cox Proportional Hazards Model

The Cox model specifies the hazard function for subject $i$ with covariate
vector $\mathbf{x}_i = (x_{i1}, \ldots, x_{ip})^\top$ as

$$
h(t \mid \mathbf{x}_i) = h_0(t) \exp(\boldsymbol{\beta}^\top \mathbf{x}_i)
$$

where:

- $h_0(t)$ is the **baseline hazard function** --- the hazard when all
  covariates are zero.  It is an arbitrary, unspecified non-negative function.
- $\boldsymbol{\beta} = (\beta_1, \ldots, \beta_p)^\top$ is the vector of
  regression coefficients.
- $\exp(\boldsymbol{\beta}^\top \mathbf{x}_i)$ is the **relative risk**
  associated with covariate profile $\mathbf{x}_i$.

The model is called "proportional hazards" because the ratio of hazards for
any two subjects $i$ and $j$ is constant over time:

$$
\frac{h(t \mid \mathbf{x}_i)}{h(t \mid \mathbf{x}_j)} = \frac{\exp(\boldsymbol{\beta}^\top \mathbf{x}_i)}{\exp(\boldsymbol{\beta}^\top \mathbf{x}_j)} = \exp\!\bigl(\boldsymbol{\beta}^\top (\mathbf{x}_i - \mathbf{x}_j)\bigr)
$$

This ratio does not depend on $t$.

!!! note "Semi-Parametric Nature"

    The Cox model is semi-parametric: the covariate effects
    $\boldsymbol{\beta}$ are finite-dimensional parameters, but the baseline
    hazard $h_0(t)$ is an infinite-dimensional nuisance parameter.  The
    partial likelihood eliminates $h_0(t)$ from the estimation of
    $\boldsymbol{\beta}$.

## The Partial Likelihood

Let the $K$ distinct ordered event times be $t_{(1)} < \cdots < t_{(K)}$, and
let $i_j$ denote the subject who experiences the event at $t_{(j)}$.  The
**risk set** at time $t_{(j)}$ is

$$
\mathcal{R}_j = \{i : t_i \geq t_{(j)}\}
$$

the set of all subjects still under observation just before $t_{(j)}$.

At event time $t_{(j)}$, the conditional probability that subject $i_j$ is the
one who experiences the event, given that exactly one event occurs among the
risk set, is

$$
\frac{h(t_{(j)} \mid \mathbf{x}_{i_j})}{\sum_{l \in \mathcal{R}_j} h(t_{(j)} \mid \mathbf{x}_l)} = \frac{h_0(t_{(j)}) \exp(\boldsymbol{\beta}^\top \mathbf{x}_{i_j})}{\sum_{l \in \mathcal{R}_j} h_0(t_{(j)}) \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)} = \frac{\exp(\boldsymbol{\beta}^\top \mathbf{x}_{i_j})}{\sum_{l \in \mathcal{R}_j} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)}
$$

The baseline hazard $h_0(t_{(j)})$ cancels in the ratio.  The **partial
likelihood** is the product of these conditional probabilities over all event
times:

$$
PL(\boldsymbol{\beta}) = \prod_{j=1}^{K} \frac{\exp(\boldsymbol{\beta}^\top \mathbf{x}_{i_j})}{\sum_{l \in \mathcal{R}_j} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)}
$$

The **partial log-likelihood** is

$$
\ell_P(\boldsymbol{\beta}) = \sum_{j=1}^{K} \left[\boldsymbol{\beta}^\top \mathbf{x}_{i_j} - \ln\!\left(\sum_{l \in \mathcal{R}_j} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)\right)\right]
$$

## Estimation

The MLE $\hat{\boldsymbol{\beta}}$ maximizes $\ell_P(\boldsymbol{\beta})$.
The score and information are:

**Score vector:**

$$
U(\boldsymbol{\beta}) = \sum_{j=1}^{K} \left[\mathbf{x}_{i_j} - \bar{\mathbf{x}}_j(\boldsymbol{\beta})\right]
$$

where $\bar{\mathbf{x}}_j(\boldsymbol{\beta})$ is the weighted mean of
covariates in the risk set:

$$
\bar{\mathbf{x}}_j(\boldsymbol{\beta}) = \frac{\sum_{l \in \mathcal{R}_j} \mathbf{x}_l \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)}{\sum_{l \in \mathcal{R}_j} \exp(\boldsymbol{\beta}^\top \mathbf{x}_l)}
$$

**Observed information:**

$$
\mathcal{I}(\boldsymbol{\beta}) = \sum_{j=1}^{K} \mathbf{V}_j(\boldsymbol{\beta})
$$

where $\mathbf{V}_j$ is the weighted covariance matrix of covariates in the
risk set at $t_{(j)}$.

The Newton--Raphson iteration is

$$
\boldsymbol{\beta}^{(m+1)} = \boldsymbol{\beta}^{(m)} + \mathcal{I}(\boldsymbol{\beta}^{(m)})^{-1} U(\boldsymbol{\beta}^{(m)})
$$

## Handling Tied Event Times

When multiple events occur at the same time, the partial likelihood must be
modified.  Common approaches:

- **Breslow approximation** (default in most software): treats ties as if
  events occur sequentially within the same risk set.  Fast but approximate.
- **Efron approximation**: more accurate than Breslow by averaging over
  possible orderings of tied events.
- **Exact partial likelihood**: enumerates all possible orderings.
  Computationally expensive for many ties.

!!! tip "Which Tie-Handling Method?"

    The Breslow and Efron methods give nearly identical results when ties are
    rare.  When ties are common (e.g., discrete event times), the Efron method
    is recommended.  The exact method is reserved for small datasets with many
    ties.

## The Breslow Estimator of the Baseline Hazard

After estimating $\hat{\boldsymbol{\beta}}$, the baseline cumulative hazard can
be estimated using the **Breslow estimator**:

$$
\hat{H}_0(t) = \sum_{j:\, t_{(j)} \leq t} \frac{d_j}{\sum_{l \in \mathcal{R}_j} \exp(\hat{\boldsymbol{\beta}}^\top \mathbf{x}_l)}
$$

where $d_j$ is the number of events at $t_{(j)}$.  The baseline survival
function is then

$$
\hat{S}_0(t) = \exp\!\bigl(-\hat{H}_0(t)\bigr)
$$

The survival function for a subject with covariates $\mathbf{x}$ is

$$
\hat{S}(t \mid \mathbf{x}) = \hat{S}_0(t)^{\exp(\hat{\boldsymbol{\beta}}^\top \mathbf{x})}
$$

!!! warning "The Breslow Estimator Is a Step Function"

    Like the Nelson--Aalen estimator, $\hat{H}_0(t)$ jumps only at observed
    event times.  Between event times the estimate is constant.  Smoothing
    techniques can produce a continuous estimate if needed, but the step
    function is the standard output.


## Exercises

**Exercise 1.**
Write the partial likelihood for the Cox proportional hazards model and explain why it is called "partial."

??? success "Solution to Exercise 1"
    For ordered event times $t_{(1)} < t_{(2)} < \dots < t_{(D)}$ with corresponding subjects $j_1, j_2, \dots, j_D$, the partial likelihood is:

    $$
    L(\boldsymbol{\beta}) = \prod_{i=1}^D \frac{\exp(\mathbf{x}_{j_i}^T\boldsymbol{\beta})}{\sum_{k \in R(t_{(i)})} \exp(\mathbf{x}_k^T\boldsymbol{\beta})}
    $$

    where $R(t_{(i)})$ is the risk set at time $t_{(i)}$ (all subjects still under observation).

    It is called "partial" because it uses only the order of events, not the actual event times or the baseline hazard $h_0(t)$. By eliminating $h_0(t)$, the Cox model avoids specifying the baseline hazard function, making it semi-parametric.

---

**Exercise 2.**
In a Cox model, $\hat{\beta} = 0.5$ for a treatment indicator. Interpret this as a hazard ratio.

??? success "Solution to Exercise 2"
    The hazard ratio is $\text{HR} = e^{\hat{\beta}} = e^{0.5} = 1.649$.

    Interpretation: the treatment group has a hazard (instantaneous risk of the event) that is 1.649 times the control group's hazard at every time point (proportional hazards assumption). Equivalently, the treatment increases the hazard by about 65%.

    If the event is death, $\text{HR} = 1.649$ means the treatment group has a 65% higher instantaneous risk of death at any given time compared to control. If $\text{HR} < 1$, the treatment is protective.

---

**Exercise 3.**
Explain the proportional hazards assumption. How is it expressed mathematically?

??? success "Solution to Exercise 3"
    The proportional hazards assumption states that the hazard ratio between any two subjects is constant over time:

    $$
    \frac{h(t \mid \mathbf{x}_i)}{h(t \mid \mathbf{x}_j)} = \frac{h_0(t)\exp(\mathbf{x}_i^T\boldsymbol{\beta})}{h_0(t)\exp(\mathbf{x}_j^T\boldsymbol{\beta})} = \exp\!\left((\mathbf{x}_i - \mathbf{x}_j)^T\boldsymbol{\beta}\right)
    $$

    The baseline hazard $h_0(t)$ cancels, making the ratio independent of time. If the ratio changes over time (e.g., a treatment works well initially but its effect fades), the PH assumption is violated and the Cox model is misspecified. Schoenfeld residuals and log-log survival plots are used to check this assumption.

---

**Exercise 4.**
Why can the Cox model handle censored observations? Explain how censored subjects enter the partial likelihood.

??? success "Solution to Exercise 4"
    Censored subjects contribute to the partial likelihood through the **risk set** but not as events. At each event time $t_{(i)}$, the risk set $R(t_{(i)})$ includes all subjects who are still alive and under observation -- both those who will eventually experience the event and those who will be censored later.

    A censored subject contributes to the denominators of the partial likelihood (they were "at risk" before being censored) but never appears in a numerator (they did not have an observed event). This is valid under the assumption that censoring is non-informative (the reason for censoring is unrelated to the event risk).

    This elegant handling of censoring -- without discarding censored observations or imputing event times -- is a key advantage of the Cox model.
