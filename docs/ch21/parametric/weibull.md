# Weibull Model

The exponential model assumes a constant hazard, which is too restrictive for
many applications.  In practice, the risk of an event often increases with time
(aging, wear-out) or decreases with time (burn-in, early failures).  The
**Weibull model** generalizes the exponential by adding a **shape parameter**
that allows the hazard to be monotonically increasing, decreasing, or constant.

This section defines the Weibull survival model, discusses the role of the shape
parameter, derives the maximum likelihood estimator, and shows how to check the
Weibull assumption graphically.

## Model Specification

The Weibull distribution has two parameters: a **scale parameter** $\lambda > 0$
and a **shape parameter** $k > 0$.

**Hazard function:**

$$
h(t) = \frac{k}{\lambda}\left(\frac{t}{\lambda}\right)^{k-1} \qquad t \geq 0
$$

**Cumulative hazard:**

$$
H(t) = \left(\frac{t}{\lambda}\right)^k
$$

**Survival function:**

$$
S(t) = \exp\!\left(-\left(\frac{t}{\lambda}\right)^k\right)
$$

**Density function:**

$$
f(t) = \frac{k}{\lambda}\left(\frac{t}{\lambda}\right)^{k-1} \exp\!\left(-\left(\frac{t}{\lambda}\right)^k\right)
$$

!!! note "Alternative Parameterization"

    Some references use the parameterization $h(t) = k \alpha t^{k-1}$ with
    $\alpha = 1/\lambda^k$.  The two forms are equivalent; this section uses
    the $(\lambda, k)$ parameterization because it separates scale from shape
    more clearly.

## Role of the Shape Parameter

The shape parameter $k$ determines how the hazard changes over time:

| $k$ | Hazard Behavior | Interpretation |
|:---:|:----------------|:---------------|
| $k < 1$ | Decreasing | Early failures dominate; survivors become more robust |
| $k = 1$ | Constant | Reduces to exponential($\lambda$); memoryless |
| $k > 1$ | Increasing | Risk grows with time; aging or wear-out |

When $k = 1$, the Weibull reduces to the exponential distribution with rate
$1/\lambda$, so the exponential model is nested within the Weibull.

!!! example "Hazard Shapes in Practice"

    - **Infant mortality** ($k < 1$): Electronic components that fail early
      due to manufacturing defects.  Survivors are reliable.
    - **Constant risk** ($k = 1$): Events driven by external random shocks
      (e.g., accidental damage).
    - **Wear-out** ($k > 1$): Mechanical parts that degrade over time
      (bearings, brake pads).

## Mean and Median Survival

The mean survival time is

$$
E[T] = \lambda \,\Gamma\!\left(1 + \frac{1}{k}\right)
$$

where $\Gamma(\cdot)$ is the gamma function.  When $k = 1$, this reduces to
$E[T] = \lambda$.

The median survival time solves $S(t_{0.5}) = 0.5$:

$$
t_{0.5} = \lambda (\ln 2)^{1/k}
$$

## Maximum Likelihood Estimation

Given $n$ subjects with observed times $t_i$ and event indicators $\delta_i$,
the log-likelihood is

$$
\ell(k, \lambda) = d \ln k - dk \ln \lambda + (k-1)\sum_{i=1}^{n} \delta_i \ln t_i - \sum_{i=1}^{n} \left(\frac{t_i}{\lambda}\right)^k
$$

where $d = \sum \delta_i$ is the number of events.

There is no closed-form solution.  The MLEs $(\hat{k}, \hat{\lambda})$ are
obtained by numerical optimization (Newton--Raphson or profile likelihood).

**Profile likelihood approach:**

1. For a fixed $k$, the MLE of $\lambda$ has a closed form:

$$
\hat{\lambda}(k) = \left(\frac{\sum_{i=1}^{n} t_i^k}{d}\right)^{1/k}
$$

2. Substitute into the log-likelihood and maximize the resulting one-dimensional
   profile likelihood over $k$.

## Checking the Weibull Assumption

The Weibull model implies a linear relationship between $\ln H(t)$ and $\ln t$:

$$
\ln H(t) = k \ln t - k \ln \lambda
$$

Therefore, if the plot of $\ln \hat{H}(t)$ (from the Nelson--Aalen estimator)
versus $\ln t$ is approximately linear, the Weibull model is appropriate.

- The **slope** of the line estimates $k$.
- The **intercept** is $-k \ln \lambda$, from which $\lambda$ can be recovered.

!!! tip "Quick Visual Check"

    Plotting $\ln(-\ln \hat{S}_{\text{KM}}(t))$ versus $\ln t$ should yield
    an approximately straight line under the Weibull model.  Curvature suggests
    a non-Weibull hazard shape (e.g., a hump-shaped hazard better captured by
    the log-normal or log-logistic models).

## Worked Example

A study observes 50 machine failures.  Fitting a Weibull model yields
$\hat{k} = 1.8$ and $\hat{\lambda} = 500$ hours.

**Interpretation:** Since $\hat{k} = 1.8 > 1$, the hazard is increasing---the
machines are wearing out over time.

**Estimated median lifetime:**

$$
\hat{t}_{0.5} = 500 \times (\ln 2)^{1/1.8} = 500 \times 0.693^{0.556} = 500 \times 0.813 = 407 \text{ hours}
$$

**Survival at 300 hours:**

$$
\hat{S}(300) = \exp\!\left(-\left(\frac{300}{500}\right)^{1.8}\right) = \exp\!\left(-0.6^{1.8}\right) = \exp(-0.398) = 0.672
$$

**Hazard at 300 hours:**

$$
\hat{h}(300) = \frac{1.8}{500}\left(\frac{300}{500}\right)^{0.8} = 0.0036 \times 0.663 = 0.0024 \text{ per hour}
$$

## Comparison with the Exponential Model

Because the exponential model is nested within the Weibull (set $k = 1$), a
formal test of $H_0: k = 1$ vs $H_1: k \neq 1$ can be conducted using the
likelihood ratio test:

$$
\Lambda = 2\bigl[\ell(\hat{k}, \hat{\lambda}) - \ell(1, \hat{\lambda}_{\text{exp}})\bigr] \;\xrightarrow{d}\; \chi^2_1
$$

Rejecting $H_0$ indicates that the hazard is not constant and the Weibull
model provides a significantly better fit than the exponential.

??? note "Weibull as an Accelerated Failure Time Model"

    The Weibull model has a dual interpretation as an **accelerated failure
    time (AFT) model**.  If $\ln T = \mu + \sigma W$ where $W$ follows a
    standard extreme-value distribution, then $T$ follows a Weibull with
    $\lambda = e^\mu$ and $k = 1/\sigma$.  This AFT representation allows
    covariates to act multiplicatively on survival time rather than on the
    hazard rate.

## Exercises

**Exercise 1.**
Weibull Shape Parameter

A Weibull model fitted to time-to-default data yields $\hat{k} = 0.75$ and
$\hat{\lambda} = 36$ months.

**(a)** Is the hazard increasing or decreasing over time?  Explain.

**(b)** Compute the median time to default.

**(c)** Compute $\hat{S}(24)$, the probability of surviving past 24 months.

??? success "Solution to Exercise 1"

    **(a)** Since $\hat{k} = 0.75 < 1$, the hazard is **decreasing** over time.
    Default risk is highest early and declines among surviving borrowers.

    **(b)** $t_{0.5} = \lambda(\ln 2)^{1/k} = 36 \times (0.693)^{1/0.75} = 36 \times 0.693^{1.333} = 36 \times 0.627 = 22.6$ months.

    **(c)** $\hat{S}(24) = \exp(-(24/36)^{0.75}) = \exp(-0.667^{0.75}) = \exp(-0.740) = 0.477$.
