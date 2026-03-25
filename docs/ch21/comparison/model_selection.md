# AIC and Concordance Index

When multiple survival models are fit to the same data, the analyst needs
objective criteria to select the best one.  Two complementary tools serve
this purpose: the **Akaike Information Criterion (AIC)**, which balances
goodness of fit against model complexity for parametric models, and the
**concordance index (C-index)**, which measures how well a model discriminates
between subjects who experience the event sooner versus later.

This section defines both criteria, discusses their scope and limitations, and
introduces the integrated Brier score as an additional evaluation metric.

## Akaike Information Criterion

The AIC for a parametric survival model with log-likelihood $\ell(\hat{\boldsymbol{\theta}})$
and $p$ estimated parameters is

$$
\text{AIC} = -2\ell(\hat{\boldsymbol{\theta}}) + 2p
$$

Lower AIC indicates a better trade-off between fit and complexity.  AIC
penalizes each additional parameter by 2 units, discouraging overfitting.

### Using AIC for Survival Models

AIC is directly applicable to **parametric** survival models (exponential,
Weibull, log-normal, log-logistic) because they share the same likelihood
framework.

!!! example "Comparing Parametric Models"

    | Model | Parameters ($p$) | Log-Likelihood | AIC |
    |:------|:----------------:|:--------------:|:---:|
    | Exponential | 1 | $-312.5$ | 627.0 |
    | Weibull | 2 | $-305.8$ | 615.6 |
    | Log-normal | 2 | $-307.1$ | 618.2 |
    | Log-logistic | 2 | $-306.4$ | 616.8 |

    The Weibull model has the lowest AIC (615.6) and is preferred among
    these candidates.

### BIC as an Alternative

The Bayesian Information Criterion replaces the fixed penalty $2p$ with a
sample-size-dependent penalty:

$$
\text{BIC} = -2\ell(\hat{\boldsymbol{\theta}}) + p \ln n
$$

BIC penalizes complexity more heavily than AIC when $n > e^2 \approx 7.4$
(which is almost always the case in practice).  BIC tends to select simpler
models than AIC.

### Limitations of AIC for the Cox Model

The Cox model maximizes the **partial** likelihood, not a full likelihood.
AIC computed from the partial log-likelihood is sometimes used in practice, but
its theoretical justification is weaker than for parametric models.  Some
software reports a partial-likelihood-based AIC for the Cox model, but
comparisons should be limited to Cox models with the same baseline (e.g.,
different covariate sets within the same Cox framework).

!!! warning "Do Not Compare AIC Across Model Families"

    AIC values from a Weibull model and a Cox model are not directly
    comparable because they are based on different likelihoods (full vs
    partial).  Use the concordance index for cross-family comparisons.

## Concordance Index

The **concordance index** (C-index or C-statistic), proposed by Harrell (1982),
measures a model's ability to correctly rank subjects by their predicted risk.
It is the survival-analysis analogue of the AUC in binary classification.

### Definition

Consider all **usable pairs** of subjects $(i, j)$ such that the subject with
the shorter observed time experienced the event (so the ordering is known).
For each usable pair, the model is **concordant** if the subject predicted to
have higher risk (lower predicted survival time or higher linear predictor)
indeed experienced the event first.

$$
C = \frac{\text{Number of concordant pairs}}{\text{Number of usable pairs}}
$$

- $C = 1.0$: perfect discrimination---the model correctly ranks every pair.
- $C = 0.5$: random guessing---the model has no discriminative ability.
- $C < 0.5$: the model systematically reverses the ordering (unusual in
  practice).

### Computing the C-Index

For a Cox model with linear predictor $\hat{\eta}_i = \hat{\boldsymbol{\beta}}^\top \mathbf{x}_i$:

1. Enumerate all pairs $(i, j)$ where $t_i < t_j$ and $\delta_i = 1$ (subject
   $i$ experienced the event first and is not censored).
2. The pair is concordant if $\hat{\eta}_i > \hat{\eta}_j$ (higher predicted
   risk for the subject who failed earlier).
3. If $\hat{\eta}_i = \hat{\eta}_j$, the pair is counted as 0.5 concordant.

!!! note "Handling Censored Observations"

    Pairs where the shorter-time subject is censored are excluded because the
    ordering is ambiguous.  Pairs where the longer-time subject is censored
    are included only if the censoring time exceeds the event time of the
    other subject.

### Interpreting the C-Index

| C-Index Range | Interpretation |
|:-------------|:---------------|
| 0.50 -- 0.60 | Poor discrimination |
| 0.60 -- 0.70 | Moderate |
| 0.70 -- 0.80 | Good |
| 0.80 -- 0.90 | Excellent |
| 0.90 -- 1.00 | Outstanding (rare in practice) |

### Advantages and Limitations

**Advantages:**

- Applicable to **any** survival model (non-parametric, parametric, Cox).
- Does not require a distributional assumption.
- Directly interpretable as the probability of correctly ranking a random pair.

**Limitations:**

- Measures **discrimination** only, not calibration (how well predicted
  probabilities match observed frequencies).
- Sensitive to the censoring distribution: heavy censoring reduces the number
  of usable pairs.
- Does not account for the time horizon; a model may discriminate well at
  short horizons but poorly at long horizons.

## Integrated Brier Score

The **Brier score** evaluates both discrimination and calibration at a specific
time $t$:

$$
\text{BS}(t) = \frac{1}{n} \sum_{i=1}^{n} \left[\hat{S}(t \mid \mathbf{x}_i) - \mathbf{1}(T_i > t)\right]^2 \cdot w_i(t)
$$

where $w_i(t)$ are inverse-probability-of-censoring weights that correct for
the fact that the true survival status $\mathbf{1}(T_i > t)$ is unknown for
censored subjects with $t_i < t$.

The **integrated Brier score (IBS)** averages the Brier score over a range of
time points:

$$
\text{IBS} = \frac{1}{t_{\max} - t_{\min}} \int_{t_{\min}}^{t_{\max}} \text{BS}(t)\, dt
$$

Lower IBS indicates better overall predictive performance.

!!! tip "When to Use Each Metric"

    - **AIC**: Comparing parametric models with the same likelihood type.
    - **C-index**: Comparing any models on discrimination.
    - **IBS**: Comparing any models on overall predictive accuracy
      (discrimination + calibration).

## Summary

| Criterion | Applicable Models | Measures | Strengths |
|:----------|:-----------------|:---------|:----------|
| AIC | Parametric (full likelihood) | Fit vs complexity | Theoretically grounded |
| BIC | Parametric (full likelihood) | Fit vs complexity (stronger penalty) | Consistent model selection |
| C-index | All | Discrimination | Model-agnostic |
| IBS | All | Discrimination + calibration | Comprehensive evaluation |
