# Instrumental Variables (Introduction)

When randomized experiments are infeasible and unmeasured confounders threaten the validity of observational estimates, **instrumental variables** (IVs) offer a way to estimate causal effects despite the confounding. The idea is to find a variable -- the instrument -- that affects the treatment but has no direct effect on the outcome, so it can serve as a source of "as-if random" variation in the treatment. This section introduces the core logic and assumptions of IV estimation at an introductory level.

---

## The Problem: Unmeasured Confounding

Recall that a confounder $U$ creates bias in the estimated effect of $X$ on $Y$ when $U$ is associated with both:

$$
X \leftarrow U \rightarrow Y
$$

If $U$ is unmeasured, standard methods (regression, matching, stratification) cannot remove the bias. We need a different strategy.

---

## The Instrument

An **instrumental variable** $Z$ is a variable that:

1. **Relevance**: $Z$ is associated with the treatment $X$.
2. **Exclusion restriction**: $Z$ affects the outcome $Y$ **only through** $X$ (no direct effect on $Y$).
3. **Independence**: $Z$ is not associated with the unmeasured confounders $U$.

The causal structure can be depicted as:

$$
Z \rightarrow X \rightarrow Y, \quad U \rightarrow X, \quad U \rightarrow Y
$$

with no arrow from $Z$ to $Y$ or from $U$ to $Z$.

The instrument $Z$ provides variation in $X$ that is free from the influence of $U$. By isolating this "clean" variation, IV methods can estimate the causal effect of $X$ on $Y$ even when $U$ is unmeasured.

---

## Intuition: The Two-Stage Process

The logic of IV estimation can be understood in two stages:

**Stage 1**: Use $Z$ to predict $X$. Since $Z$ is independent of $U$, the predicted values $\hat{X}$ contain only the "clean" variation in $X$ (the part driven by $Z$, not by $U$).

**Stage 2**: Regress $Y$ on $\hat{X}$. Since $\hat{X}$ is free of confounding by $U$, the resulting coefficient estimates the causal effect of $X$ on $Y$.

This procedure is called **two-stage least squares** (2SLS) and is the most common IV estimator.

---

## Simple IV Estimator

In the simplest case with a single instrument $Z$ and no additional covariates, the IV estimator of the causal effect $\beta$ of $X$ on $Y$ is the **Wald estimator**:

$$
\hat{\beta}_{IV} = \frac{\text{Cov}(Z, Y)}{\text{Cov}(Z, X)}
$$

This ratio captures the total effect of $Z$ on $Y$ divided by the effect of $Z$ on $X$, yielding the per-unit effect of $X$ on $Y$.

Equivalently, using sample correlations:

$$
\hat{\beta}_{IV} = \frac{r_{ZY}}{r_{ZX}} \cdot \frac{S_Y}{S_X}
$$

---

## Classic Examples of Instruments

### Distance to College as an Instrument for Education

To estimate the causal effect of education ($X$) on wages ($Y$), Card (1993) used distance to the nearest college ($Z$) as an instrument:

- **Relevance**: people who live closer to a college are more likely to attend.
- **Exclusion**: distance to college does not directly affect wages (after controlling for geographic factors).
- **Independence**: distance to college is plausibly unrelated to unmeasured ability ($U$).

### Quarter of Birth as an Instrument for Schooling

Angrist and Krueger (1991) used quarter of birth ($Z$) as an instrument for years of schooling ($X$):

- **Relevance**: compulsory schooling laws interact with birth quarter to create variation in schooling.
- **Exclusion**: birth quarter has no direct effect on wages.
- **Independence**: birth quarter is essentially random.

### Rainfall as an Instrument for Economic Activity

Miguel, Satyanath, and Sergenti (2004) used rainfall ($Z$) as an instrument for economic growth ($X$) when studying the effect of economic conditions on civil conflict ($Y$) in Africa.

---

## Assumptions and Their Violations

The validity of IV estimation depends critically on the three assumptions:

| Assumption | Violation | Consequence |
|:---|:---|:---|
| Relevance ($Z$ affects $X$) | Weak instrument ($Z$ barely affects $X$) | Large bias, unreliable inference |
| Exclusion ($Z$ affects $Y$ only through $X$) | Direct effect of $Z$ on $Y$ | Biased IV estimate |
| Independence ($Z$ independent of $U$) | $Z$ correlated with confounders | Biased IV estimate |

!!! warning "Weak instruments"
    When the instrument is only weakly correlated with the treatment, the IV estimator has large variance and can be severely biased, sometimes worse than the ordinary (confounded) OLS estimator. A common rule of thumb is that the first-stage F-statistic should exceed 10.

The exclusion restriction is typically the most controversial assumption because it cannot be tested directly from the data. It must be justified on substantive grounds.

---

## IV vs Other Methods

| Method | Handles unmeasured confounders? | Key requirement |
|:---|:---:|:---|
| OLS regression | No | All confounders measured |
| Matching / propensity scores | No | All confounders measured |
| Instrumental variables | Yes | Valid instrument available |
| Randomized experiment | Yes | Ethical and feasible |

IV methods fill an important gap between observational studies (which require all confounders to be measured) and experiments (which may be infeasible). The cost is that a valid instrument must be found, and the exclusion restriction must be credible.

---

## Summary

Instrumental variables provide a method for estimating causal effects in the presence of unmeasured confounding. An instrument is a variable that is relevant (affects the treatment), satisfies the exclusion restriction (affects the outcome only through the treatment), and is independent of unmeasured confounders. The two-stage least squares estimator uses the instrument to isolate variation in the treatment that is free from confounding. While powerful, IV methods require strong and often untestable assumptions, particularly the exclusion restriction. They are most convincing when the instrument has a clear, well-understood mechanism of action.
