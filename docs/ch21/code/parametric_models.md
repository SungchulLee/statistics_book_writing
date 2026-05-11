# Parametric Survival Models

## Overview

Parametric survival models assume that event times follow a known probability
distribution, fully specified by a finite set of parameters.  Unlike the non-parametric
Kaplan-Meier estimator, parametric models produce smooth survival and hazard curves,
enable extrapolation, and support formal model comparison via information criteria.
This page covers the three most common parametric families --- exponential, Weibull,
and log-normal --- and demonstrates maximum likelihood estimation with censored data.

## Exponential Model

### Specification

The exponential distribution is the simplest parametric survival model, characterized
by a single rate parameter $\lambda > 0$ and a **constant hazard**:

$$
h(t) = \lambda, \qquad S(t) = e^{-\lambda t}, \qquad f(t) = \lambda e^{-\lambda t}
$$

The mean survival time is $E[T] = 1/\lambda$.

### Memoryless Property

The exponential distribution is uniquely characterized by the memoryless property:

$$
P(T > t + s \mid T > t) = P(T > s) \quad \text{for all } t, s \geq 0
$$

This means that the probability of surviving an additional $s$ units does not depend
on how long the subject has already survived.

### Maximum Likelihood Estimation

Given $n$ subjects with times $t_1, \ldots, t_n$ and event indicators
$\delta_1, \ldots, \delta_n$ ($\delta_i = 1$ for events, $0$ for censored), the
log-likelihood is

$$
\ell(\lambda) = d \ln \lambda - \lambda \sum_{i=1}^{n} t_i
$$

where $d = \sum \delta_i$.  Setting $\ell'(\lambda) = 0$ gives

$$
\hat{\lambda} = \frac{d}{\sum_{i=1}^{n} t_i}
$$

The MLE is the number of events divided by the total person-time.

## Weibull Model

### Specification

The Weibull distribution generalizes the exponential by adding a shape parameter
$k > 0$ alongside a scale parameter $\lambda > 0$:

$$
h(t) = \frac{k}{\lambda}\left(\frac{t}{\lambda}\right)^{k-1}
$$

$$
S(t) = \exp\!\left(-\left(\frac{t}{\lambda}\right)^k\right)
$$

$$
f(t) = \frac{k}{\lambda}\left(\frac{t}{\lambda}\right)^{k-1}\exp\!\left(-\left(\frac{t}{\lambda}\right)^k\right)
$$

### Role of the Shape Parameter

| $k$ | Hazard Behavior | Interpretation |
|:---:|:----------------|:---------------|
| $k < 1$ | Decreasing | Early failures dominate; survivors become robust |
| $k = 1$ | Constant | Reduces to exponential($\lambda$) |
| $k > 1$ | Increasing | Wear-out or aging; risk grows with time |

The median survival time is $t_{0.5} = \lambda (\ln 2)^{1/k}$.

### Maximum Likelihood Estimation

The log-likelihood for the Weibull model is

$$
\ell(k, \lambda) = d \ln k - dk \ln \lambda + (k-1)\sum_{i=1}^{n} \delta_i \ln t_i - \sum_{i=1}^{n}\left(\frac{t_i}{\lambda}\right)^k
$$

No closed-form solution exists.  Numerical optimization (e.g., Newton-Raphson or
profile likelihood) is used.

### Checking the Weibull Assumption

The Weibull model implies linearity on a log-log scale:

$$
\ln H(t) = k \ln t - k \ln \lambda
$$

A plot of $\ln \hat{H}(t)$ versus $\ln t$ (using the Nelson-Aalen estimator) should
be approximately linear if the Weibull model is appropriate.

## Log-Normal Model

### Specification

The log-normal model assumes $\ln T \sim N(\mu, \sigma^2)$.  The survival and hazard
functions do not have simple closed forms but are expressed through the standard
normal CDF $\mathcal{N}$:

$$
S(t) = 1 - \mathcal{N}\!\left(\frac{\ln t - \mu}{\sigma}\right)
$$

$$
f(t) = \frac{1}{t\sigma}\phi\!\left(\frac{\ln t - \mu}{\sigma}\right)
$$

$$
h(t) = \frac{f(t)}{S(t)}
$$

where $\phi$ is the standard normal PDF.

### Key Feature

The log-normal hazard is **non-monotone**: it increases initially and then decreases.
This makes it suitable for phenomena where the risk peaks at an intermediate time and
then declines (e.g., recovery from surgery, certain disease relapse patterns).

## Implementation

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

def neg_loglik_exponential(lam, times, events):
    """Negative log-likelihood for the exponential model."""
    d = events.sum()
    total_time = times.sum()
    return -(d * np.log(lam) - lam * total_time)

def neg_loglik_weibull(params, times, events):
    """Negative log-likelihood for the Weibull model."""
    k, lam = params
    d = events.sum()
    ll = (d * np.log(k)
          - d * k * np.log(lam)
          + (k - 1) * np.sum(events * np.log(times + 1e-15))
          - np.sum((times / lam) ** k))
    return -ll

def neg_loglik_lognormal(params, times, events):
    """Negative log-likelihood for the log-normal model."""
    mu, sigma = params
    z = (np.log(times + 1e-15) - mu) / sigma
    ll = np.sum(
        events * norm.logpdf(z) - events * np.log(sigma * times + 1e-15)
        + (1 - events) * norm.logsf(z)
    )
    return -ll
```

Each function computes the negative log-likelihood so that standard minimization
routines can be used.  The event indicator `events[i]` equals 1 for observed events
and 0 for censored observations.  Censored subjects contribute through the survival
function term $\ln S(t_i)$.

## Model Selection

Parametric models are compared using information criteria computed from the maximized
log-likelihood $\hat{\ell}$ and the number of parameters $p$:

$$
\text{AIC} = -2\hat{\ell} + 2p
$$

$$
\text{BIC} = -2\hat{\ell} + p \ln n
$$

Lower values indicate a better trade-off between fit and complexity.  The exponential
model has $p = 1$, the Weibull has $p = 2$, and the log-normal has $p = 2$.

Since the exponential is nested within the Weibull ($k = 1$), a **likelihood ratio
test** can formally test whether the additional shape parameter is needed:

$$
\Lambda = 2[\hat{\ell}_{\text{Weibull}} - \hat{\ell}_{\text{Exp}}] \;\xrightarrow{d}\; \chi^2_1
$$

!!! tip "Graphical Diagnostics"

    Always compare the fitted parametric survival curve against the Kaplan-Meier
    estimate.  Large discrepancies indicate model misspecification regardless of
    what the AIC suggests.

## Interpretation

- **Exponential model**: Appropriate when the hazard is approximately constant.
  Useful as a baseline but rarely exact in practice.
- **Weibull model**: Captures monotone hazards.  The shape parameter $k$ directly
  indicates whether risk increases ($k > 1$) or decreases ($k < 1$) over time.
- **Log-normal model**: Suitable for non-monotone hazards that rise and then fall.
  Common in medical applications where initial risk is high but long-term survivors
  have decreasing hazard.
- **Model selection**: Use AIC/BIC for non-nested comparisons and likelihood ratio
  tests for nested models.  Always supplement with graphical checks.

## Exercises

**Exercise 1.**
Exponential Model MLE

A study follows 25 subjects.  There are 16 observed events ($d = 16$) and the total
person-time is $\sum t_i = 3{,}200$ hours.

**(a)** Compute the MLE $\hat{\lambda}$.

**(b)** Estimate the mean survival time and the survival probability at $t = 100$.

??? success "Solution to Exercise 1"

    **(a)** $\hat{\lambda} = d / \sum t_i = 16 / 3200 = 0.005$ per hour.

    **(b)** Mean survival time: $1/\hat{\lambda} = 200$ hours.

    Survival at $t = 100$:

    $$
    \hat{S}(100) = e^{-0.005 \times 100} = e^{-0.5} = 0.607
    $$

---

**Exercise 2.**
Weibull Shape Interpretation

A Weibull model fitted to equipment failure data yields $\hat{k} = 2.3$ and
$\hat{\lambda} = 800$ hours.

**(a)** Is the hazard increasing or decreasing?

**(b)** Compute the median time to failure.

**(c)** Compare the estimated hazard at $t = 400$ and $t = 600$.

??? success "Solution to Exercise 2"

    **(a)** Since $\hat{k} = 2.3 > 1$, the hazard is **increasing** over time.  The
    equipment wears out.

    **(b)** $t_{0.5} = \lambda (\ln 2)^{1/k} = 800 \times (0.693)^{1/2.3} = 800 \times 0.693^{0.435} = 800 \times 0.856 = 685$ hours.

    **(c)** At $t = 400$:

    $$
    \hat{h}(400) = \frac{2.3}{800}\left(\frac{400}{800}\right)^{1.3} = 0.002875 \times 0.5^{1.3} = 0.002875 \times 0.406 = 0.00117
    $$

    At $t = 600$:

    $$
    \hat{h}(600) = \frac{2.3}{800}\left(\frac{600}{800}\right)^{1.3} = 0.002875 \times 0.75^{1.3} = 0.002875 \times 0.683 = 0.00196
    $$

    The hazard at $t = 600$ is about 1.68 times the hazard at $t = 400$, consistent
    with an increasing hazard.

---

**Exercise 3.**
Log-Normal Hazard Shape

**(a)** Explain why the log-normal hazard function is non-monotone.

**(b)** For a log-normal model with $\mu = 3$ and $\sigma = 0.8$, compute the median
survival time.

??? success "Solution to Exercise 3"

    **(a)** The log-normal hazard $h(t) = f(t)/S(t)$ is the ratio of the PDF to the
    survival function.  For small $t$, the density $f(t)$ increases while $S(t)$
    is close to 1, so the hazard increases.  For large $t$, both $f(t)$ and $S(t)$
    decrease, but $f(t)$ decreases faster than $S(t)$, causing the hazard to
    eventually decline.  This produces a hump-shaped hazard curve.

    **(b)** For the log-normal distribution, the median of $T$ is $e^{\mu}$ (since
    the median of $\ln T \sim N(\mu, \sigma^2)$ is $\mu$, and $e^\mu$ is the
    median of $T$).  Therefore:

    $$
    t_{0.5} = e^{3} = 20.09
    $$

    The median survival time is approximately 20.1 time units.

---

**Exercise 4.**
Model Comparison

An analyst fits three models to the same dataset of 100 subjects.  The results are:

| Model | Parameters | Max log-likelihood |
|:------|:----------:|:------------------:|
| Exponential | 1 | $-312.5$ |
| Weibull | 2 | $-298.1$ |
| Log-normal | 2 | $-300.3$ |

**(a)** Compute the AIC for each model.

**(b)** Perform a likelihood ratio test of exponential vs Weibull at $\alpha = 0.05$.

**(c)** Which model would you select and why?

??? success "Solution to Exercise 4"

    **(a)** AIC $= -2\hat{\ell} + 2p$:

    - Exponential: $-2(-312.5) + 2(1) = 627.0$
    - Weibull: $-2(-298.1) + 2(2) = 600.2$
    - Log-normal: $-2(-300.3) + 2(2) = 604.6$

    **(b)** Likelihood ratio statistic:

    $$
    \Lambda = 2[-298.1 - (-312.5)] = 2 \times 14.4 = 28.8
    $$

    Under $H_0: k = 1$, $\Lambda \sim \chi^2_1$.  The critical value at
    $\alpha = 0.05$ is 3.84.  Since $28.8 \gg 3.84$, we reject the exponential
    model in favor of the Weibull.  The hazard is not constant.

    **(c)** The Weibull model has the lowest AIC (600.2) and is significantly better
    than the exponential by the LRT.  It is preferred over the log-normal model as
    well (AIC 600.2 vs 604.6).  The Weibull model provides the best trade-off
    between fit and parsimony for this dataset.

---

**Exercise 5.**
Likelihood with Censoring

Show that for a parametric survival model with density $f(t)$ and survival function
$S(t)$, the likelihood contribution of a censored observation at time $t_i$ is $S(t_i)$,
not $f(t_i)$.

??? success "Solution to Exercise 5"

    For an observed event at time $t_i$, we know the event occurred in the
    infinitesimal interval $[t_i, t_i + dt)$.  The probability of this is
    $f(t_i)\,dt$, so the likelihood contribution (up to proportionality) is $f(t_i)$.

    For a censored observation at time $t_i$, we know only that the true event time
    $T_i$ exceeds $t_i$.  The probability of this is

    $$
    P(T_i > t_i) = S(t_i)
    $$

    Therefore the likelihood contribution is $S(t_i)$, not $f(t_i)$.  Combining
    both cases, the full likelihood for subject $i$ is

    $$
    L_i = [f(t_i)]^{\delta_i} [S(t_i)]^{1-\delta_i}
    $$

    where $\delta_i = 1$ for events and $\delta_i = 0$ for censored observations.
    This is the foundation of all parametric survival model estimation with censored
    data. $\square$
