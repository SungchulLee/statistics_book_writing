# Nelson-Aalen Cumulative Hazard

The Kaplan--Meier estimator targets the survival function $S(t)$ directly.  An
alternative non-parametric approach estimates the **cumulative hazard function**
$H(t)$ instead and then recovers the survival curve via the relationship
$S(t) = \exp(-H(t))$.  This is the **Nelson--Aalen estimator**, developed
independently by Nelson (1972) and Aalen (1978).

The Nelson--Aalen estimator is particularly useful when the cumulative hazard
itself is the quantity of interest, for instance when assessing whether the
hazard is constant (straight line on a cumulative hazard plot) or when
constructing confidence intervals with better small-sample properties.

## Definition

Using the same notation as the Kaplan--Meier section, let
$t_{(1)} < t_{(2)} < \cdots < t_{(K)}$ be the distinct ordered event times.
At each event time $t_{(j)}$:

- $d_j$ = number of events at $t_{(j)}$.
- $n_j$ = number of subjects at risk just before $t_{(j)}$.

The **Nelson--Aalen estimator** of the cumulative hazard is

$$
\hat{H}(t) = \sum_{j:\, t_{(j)} \leq t} \frac{d_j}{n_j}
$$

Each term $d_j / n_j$ estimates the conditional probability of the event at
time $t_{(j)}$ given survival to that point, which approximates
$h(t_{(j)}) \cdot \Delta t$ for small intervals.

## Relationship to the Kaplan--Meier Estimator

The Kaplan--Meier and Nelson--Aalen estimators target the same underlying
survival function through different routes.  Using the identity
$-\ln(1 - x) \approx x$ for small $x$:

$$
\hat{H}_{\text{NA}}(t) = \sum_{j:\, t_{(j)} \leq t} \frac{d_j}{n_j} \approx -\sum_{j:\, t_{(j)} \leq t} \ln\!\left(1 - \frac{d_j}{n_j}\right) = -\ln \hat{S}_{\text{KM}}(t)
$$

The approximation is accurate when $d_j / n_j$ is small at each event time,
which is typical in large samples.  Therefore,

$$
\hat{S}_{\text{NA}}(t) = \exp\!\bigl(-\hat{H}_{\text{NA}}(t)\bigr) \approx \hat{S}_{\text{KM}}(t)
$$

For finite samples the two can differ slightly.  The Nelson--Aalen-based
survival estimate $\hat{S}_{\text{NA}}(t)$ is always at least as large as the
Kaplan--Meier estimate $\hat{S}_{\text{KM}}(t)$, because
$\exp(-x) \geq 1 - x$ for all $x \geq 0$.

## Variance Estimation

The variance of the Nelson--Aalen estimator is estimated by

$$
\widehat{\text{Var}}\bigl(\hat{H}(t)\bigr) = \sum_{j:\, t_{(j)} \leq t} \frac{d_j}{n_j^2}
$$

This follows from treating each increment $d_j / n_j$ as approximately
independent with variance $d_j / n_j^2$ (derived from the binomial variance
of $d_j$ given $n_j$).

An approximate $100(1 - \alpha)\%$ confidence interval for $H(t)$ is

$$
\hat{H}(t) \pm z_{\alpha/2} \sqrt{\sum_{j:\, t_{(j)} \leq t} \frac{d_j}{n_j^2}}
$$

where $z_{\alpha/2}$ is the standard normal quantile.

!!! tip "Log-Transformed Confidence Intervals"

    Because $H(t) \geq 0$, the linear confidence interval above may produce
    negative lower bounds for small $H(t)$.  A log transformation avoids this:
    construct an interval for $\ln \hat{H}(t)$ and exponentiate the endpoints.

## Worked Example

Using the same data from the Kaplan--Meier section:

| $t_{(j)}$ | $n_j$ | $d_j$ | $d_j / n_j$ | $\hat{H}(t_{(j)})$ |
|:----------:|:-----:|:-----:|:------------:|:-------------------:|
| 1 | 8 | 1 | 0.125 | 0.125 |
| 3 | 6 | 1 | 0.167 | 0.292 |
| 5 | 4 | 2 | 0.500 | 0.792 |
| 9 | 1 | 1 | 1.000 | 1.792 |

The Nelson--Aalen survival estimate at $t = 5$ is

$$
\hat{S}_{\text{NA}}(5) = \exp(-0.792) = 0.453
$$

Compare with the Kaplan--Meier estimate $\hat{S}_{\text{KM}}(5) = 0.365$.
The difference is noticeable here because $d_3 / n_3 = 0.500$ is not small,
so the approximation $\exp(-x) \approx 1 - x$ is poor at that step.

The estimated variance of $\hat{H}(5)$ is

$$
\widehat{\text{Var}}\bigl(\hat{H}(5)\bigr) = \frac{1}{64} + \frac{1}{36} + \frac{2}{16} = 0.0156 + 0.0278 + 0.125 = 0.169
$$

A 95% confidence interval for $H(5)$ is $0.792 \pm 1.96 \sqrt{0.169} = (0.0, 1.60)$.

## When to Use the Nelson--Aalen Estimator

The Nelson--Aalen estimator is preferred over the Kaplan--Meier estimator in
several settings:

- **Cumulative hazard plots.** Plotting $\hat{H}(t)$ against $t$ reveals the
  hazard structure.  A straight line suggests a constant hazard (exponential
  model); upward curvature suggests increasing hazard (Weibull with $k > 1$).
- **Small samples.** The Nelson--Aalen estimator has slightly less bias than
  the Kaplan--Meier estimator in small samples.
- **Building blocks.** The Nelson--Aalen estimator appears in the Breslow
  estimator for the baseline cumulative hazard in the Cox model (Section 21.4).

!!! note "Graphical Model Checking"

    A plot of $\hat{H}(t)$ vs $t$ is a powerful informal diagnostic.  If the
    plot is approximately linear through the origin, the exponential model is
    reasonable.  If the plot of $\ln \hat{H}(t)$ vs $\ln t$ is approximately
    linear, the Weibull model is appropriate.

## Summary

| Property | Nelson--Aalen | Kaplan--Meier |
|:---------|:-------------|:--------------|
| Estimand | $H(t)$ | $S(t)$ |
| Formula | $\sum d_j / n_j$ | $\prod (1 - d_j / n_j)$ |
| Range | $[0, \infty)$ | $[0, 1]$ |
| Bias (small samples) | Slightly less | Slightly more |
| Large-sample equivalence | $\hat{H} \approx -\ln \hat{S}_{\text{KM}}$ | $\hat{S}_{\text{KM}} \approx \exp(-\hat{H})$ |
