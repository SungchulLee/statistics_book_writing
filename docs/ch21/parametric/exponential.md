# Exponential Model

The exponential distribution is the simplest parametric model for survival data.
It assumes that the hazard rate is **constant** over time: the risk of the event
occurring in the next instant does not depend on how long the subject has already
survived.  While this assumption is restrictive, the exponential model serves as
a baseline against which more flexible models are compared, and it is the natural
starting point for parametric survival analysis.

This section defines the exponential survival model, derives its key quantities,
discusses the memoryless property, and obtains the maximum likelihood estimator
for censored data.

## Model Specification

The exponential model has a single parameter $\lambda > 0$, the constant hazard
rate.  The four key functions are:

**Hazard function:**

$$
h(t) = \lambda \qquad \text{for all } t \geq 0
$$

**Cumulative hazard:**

$$
H(t) = \lambda t
$$

**Survival function:**

$$
S(t) = \exp(-\lambda t)
$$

**Density function:**

$$
f(t) = \lambda \exp(-\lambda t)
$$

The mean survival time is $E[T] = 1/\lambda$ and the variance is
$\text{Var}(T) = 1/\lambda^2$.

## The Memoryless Property

The constant hazard implies the **memoryless property**: the probability of
surviving an additional $s$ time units does not depend on how long the subject
has already survived.  Formally,

$$
P(T > t + s \mid T > t) = P(T > s) \qquad \text{for all } t, s \geq 0
$$

**Proof.** Using the survival function:

$$
P(T > t + s \mid T > t) = \frac{S(t + s)}{S(t)} = \frac{e^{-\lambda(t+s)}}{e^{-\lambda t}} = e^{-\lambda s} = S(s)
$$

$\square$

The exponential distribution is the only continuous distribution with this
property.

!!! example "Memoryless Default"

    If loan defaults follow an exponential model with $\lambda = 0.02$ per
    month, a loan that has survived 12 months has the same probability of
    defaulting in the next month as a brand-new loan.  In practice, this
    assumption is often violated---seasoned loans tend to default at different
    rates than new loans---motivating the Weibull and other flexible models.

## Maximum Likelihood Estimation

Given $n$ subjects with observed times $t_1, \ldots, t_n$ and event indicators
$\delta_1, \ldots, \delta_n$, the likelihood is

$$
L(\lambda) = \prod_{i=1}^{n} \bigl[f(t_i)\bigr]^{\delta_i} \bigl[S(t_i)\bigr]^{1-\delta_i} = \prod_{i=1}^{n} \bigl[\lambda e^{-\lambda t_i}\bigr]^{\delta_i} \bigl[e^{-\lambda t_i}\bigr]^{1-\delta_i}
$$

Simplifying:

$$
L(\lambda) = \lambda^d \exp\!\left(-\lambda \sum_{i=1}^{n} t_i\right)
$$

where $d = \sum_{i=1}^{n} \delta_i$ is the total number of observed events.

The log-likelihood is

$$
\ell(\lambda) = d \ln \lambda - \lambda \sum_{i=1}^{n} t_i
$$

Setting $\ell'(\lambda) = 0$:

$$
\frac{d}{\lambda} - \sum_{i=1}^{n} t_i = 0 \implies \hat{\lambda} = \frac{d}{\sum_{i=1}^{n} t_i}
$$

The MLE is the number of events divided by the total observed person-time.

!!! note "Interpretation of the MLE"

    The denominator $\sum_{i=1}^n t_i$ is the total **person-time** at risk.
    Censored subjects contribute their censored time to this total, so they
    inform the estimate even though their event was not observed.

## Variance and Confidence Interval

The Fisher information for a single observation is

$$
I(\lambda) = \frac{d}{\lambda^2}
$$

The asymptotic variance of $\hat{\lambda}$ is

$$
\text{Var}(\hat{\lambda}) \approx \frac{\hat{\lambda}^2}{d}
$$

A 95% confidence interval for $\lambda$ is

$$
\hat{\lambda} \pm 1.96 \cdot \frac{\hat{\lambda}}{\sqrt{d}}
$$

A confidence interval for the mean survival time $1/\lambda$ can be obtained by
inverting the endpoints.

## Worked Example

A reliability study monitors 20 light bulbs.  By the study's end, 12 have
burned out (events) and 8 are still working (censored).  The total observed
time across all 20 bulbs is $\sum t_i = 5{,}000$ hours.

**MLE:**

$$
\hat{\lambda} = \frac{12}{5000} = 0.0024 \text{ per hour}
$$

**Estimated mean lifetime:** $1/\hat{\lambda} = 417$ hours.

**95% CI for $\lambda$:**

$$
0.0024 \pm 1.96 \times \frac{0.0024}{\sqrt{12}} = 0.0024 \pm 0.00136 = (0.0010, 0.0038)
$$

**Estimated survival at 200 hours:**

$$
\hat{S}(200) = \exp(-0.0024 \times 200) = \exp(-0.48) = 0.619
$$

## Checking the Exponential Assumption

The constant-hazard assumption can be assessed graphically:

- **Cumulative hazard plot.** If the Nelson--Aalen estimate $\hat{H}(t)$ is
  approximately linear through the origin, the exponential model is reasonable.
- **KM vs fitted.** Overlay the parametric survival curve
  $\hat{S}(t) = e^{-\hat{\lambda}t}$ on the Kaplan--Meier estimate.
  Systematic deviations indicate a poor fit.

!!! warning "The Exponential Model Is Rarely Exact"

    Few real-world processes have a truly constant hazard.  The exponential
    model is best used as a baseline or when the constant-hazard assumption is
    a reasonable approximation over the time range of interest.  The Weibull
    model (next section) generalizes the exponential by adding a shape
    parameter that allows monotone increasing or decreasing hazard.

## Connection to the Poisson Process

Events from independent exponential survival times form a **Poisson process**.
If each subject has an independent $\text{Exp}(\lambda)$ event time, the number
of events in a time interval of length $t$ among $n$ subjects at risk follows
approximately a Poisson distribution with rate $n\lambda t$.  This connection
links survival analysis to counting process theory and provides an alternative
route to the asymptotic properties of the MLE.
