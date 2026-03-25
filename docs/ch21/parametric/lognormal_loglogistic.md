# Log-Normal and Log-Logistic Models

The exponential and Weibull models assume that the hazard function is monotone
(constant, increasing, or decreasing).  Many real-world processes, however,
exhibit a **non-monotone hazard**: the risk peaks at some time and then
declines.  Loan default rates often rise in the first year and then fall among
surviving borrowers.  Disease recurrence risk may spike shortly after treatment
and then diminish.

The **log-normal** and **log-logistic** models accommodate this hump-shaped
hazard pattern.  Both belong to the accelerated failure time (AFT) family and
model $\ln T$ with a location-scale distribution.

## Log-Normal Model

If $T$ is a survival time such that $\ln T \sim N(\mu, \sigma^2)$, then $T$
follows a **log-normal distribution** with parameters $\mu$ (location) and
$\sigma > 0$ (scale).

**Density:**

$$
f(t) = \frac{1}{t \sigma \sqrt{2\pi}} \exp\!\left(-\frac{(\ln t - \mu)^2}{2\sigma^2}\right) \qquad t > 0
$$

**Survival function:**

$$
S(t) = 1 - \Phi\!\left(\frac{\ln t - \mu}{\sigma}\right)
$$

where $\Phi(\cdot)$ is the standard normal CDF.

**Hazard function:**

$$
h(t) = \frac{f(t)}{S(t)} = \frac{\phi\!\left(\frac{\ln t - \mu}{\sigma}\right)}{t \sigma \left[1 - \Phi\!\left(\frac{\ln t - \mu}{\sigma}\right)\right]}
$$

where $\phi(\cdot)$ is the standard normal PDF.

The log-normal hazard is **not monotone**: it increases from 0 to a peak and
then decreases toward 0 as $t \to \infty$.

**Mean and median:**

$$
E[T] = \exp\!\left(\mu + \frac{\sigma^2}{2}\right), \qquad t_{0.5} = e^\mu
$$

The median has a particularly clean form: it equals $e^\mu$, independent of
$\sigma$.

!!! example "Log-Normal Default Hazard"

    A bank fits a log-normal model to time-to-default data and estimates
    $\hat{\mu} = 3.2$ and $\hat{\sigma} = 0.8$.  The median time to default
    is $e^{3.2} = 24.5$ months.  The hazard peaks before the median and then
    declines, consistent with the observation that default risk rises during
    the loan's seasoning period and then falls.

## Log-Logistic Model

If $\ln T$ follows a logistic distribution with location $\mu$ and scale
$\sigma > 0$, then $T$ follows a **log-logistic distribution**.  The standard
parameterization uses shape $k = 1/\sigma$ and scale $\lambda = e^\mu$.

**Survival function:**

$$
S(t) = \frac{1}{1 + (t/\lambda)^k}
$$

**Hazard function:**

$$
h(t) = \frac{(k/\lambda)(t/\lambda)^{k-1}}{1 + (t/\lambda)^k}
$$

**Density:**

$$
f(t) = \frac{(k/\lambda)(t/\lambda)^{k-1}}{\left[1 + (t/\lambda)^k\right]^2}
$$

The hazard shape depends on $k$:

| $k$ | Hazard Behavior |
|:---:|:----------------|
| $k \leq 1$ | Monotonically decreasing from $\infty$ at $t = 0$ |
| $k > 1$ | Hump-shaped: increases to a peak, then decreases |

When $k > 1$, the log-logistic hazard has the same qualitative shape as the
log-normal hazard (rises then falls), but its survival function has a simpler
closed form.

**Median survival time:**

$$
t_{0.5} = \lambda
$$

This follows directly from $S(\lambda) = 1/(1 + 1) = 0.5$.

## Comparison of Log-Normal and Log-Logistic

| Property | Log-Normal | Log-Logistic |
|:---------|:-----------|:-------------|
| $\ln T$ distribution | Normal | Logistic |
| Survival function | Involves $\Phi$ (no closed form) | Closed form: $1/[1 + (t/\lambda)^k]$ |
| Hazard shape | Always hump-shaped | Hump-shaped ($k > 1$) or decreasing ($k \leq 1$) |
| Tail behavior | Hazard $\to 0$ faster | Hazard $\to 0$ more slowly (heavier tails) |
| Median | $e^\mu$ | $\lambda$ |
| Closed-form $S(t)$ | No | Yes |

The log-logistic model is often preferred for computational convenience because
its survival function does not involve the normal CDF.

## Accelerated Failure Time Interpretation

Both models belong to the **accelerated failure time (AFT)** family.  An AFT
model with covariates $\mathbf{x}$ specifies

$$
\ln T = \mathbf{x}^\top \boldsymbol{\beta} + \sigma W
$$

where $W$ is a standardized error distribution:

- $W \sim N(0, 1)$ gives the log-normal AFT model.
- $W \sim \text{Logistic}(0, 1)$ gives the log-logistic AFT model.

In the AFT interpretation, covariates **accelerate or decelerate** the time
scale.  A covariate with coefficient $\beta_j > 0$ extends survival time by a
factor of $e^{\beta_j}$; a coefficient $\beta_j < 0$ shortens it.

!!! note "AFT vs Proportional Hazards"

    The Cox model (Section 21.4) assumes covariates act multiplicatively on
    the hazard.  AFT models assume covariates act multiplicatively on the
    time scale.  The Weibull is the only model that satisfies both the
    proportional hazards and AFT properties simultaneously.

## Checking Model Fit

**Log-normal check.** If $T$ is log-normal, then $\Phi^{-1}(1 - \hat{S}(t))$
plotted against $\ln t$ should be approximately linear with slope $1/\sigma$
and intercept $-\mu/\sigma$.

**Log-logistic check.** If $T$ is log-logistic, then
$\ln[\hat{S}(t)^{-1} - 1]$ plotted against $\ln t$ should be approximately
linear with slope $k$ and intercept $-k \ln \lambda$.

These graphical checks complement formal goodness-of-fit tests and help
distinguish between the log-normal and log-logistic when both fit reasonably
well.

!!! tip "When to Choose These Models"

    Use the log-normal or log-logistic model when the cumulative hazard plot
    (Nelson--Aalen) shows curvature inconsistent with the Weibull, and when
    substantive knowledge suggests the hazard peaks and then declines.  If the
    hazard is monotone, the Weibull model is a better and simpler choice.
