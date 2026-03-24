# Survival and Hazard Functions

Once the structure of time-to-event data and censoring is understood, the next
step is to define the mathematical quantities that describe how event times are
distributed.  In ordinary statistics a distribution is characterized by its PDF
and CDF.  Survival analysis uses the same information but repackages it into
two functions that are more natural for duration data: the **survival function**
and the **hazard function**.

This section defines both functions, derives the cumulative hazard, and
establishes the relationships that connect all four quantities.

## The Survival Function

Let $T$ be a non-negative continuous random variable representing the time until
an event.  The **survival function** is the probability that the event has not
yet occurred by time $t$:

$$
S(t) = P(T > t) = 1 - F(t)
$$

where $F(t) = P(T \leq t)$ is the usual cumulative distribution function.

**Properties of the survival function:**

1. $S(0) = 1$ --- every subject is alive (event-free) at time zero.
2. $\lim_{t \to \infty} S(t) = 0$ --- the event eventually occurs for everyone.
3. $S(t)$ is non-increasing in $t$.

!!! note "Proper vs Improper Survival Functions"

    Some models allow $\lim_{t \to \infty} S(t) = p > 0$, meaning a fraction
    $p$ of the population never experiences the event (a "cured" fraction).
    These are called **cure rate models** and lie outside the scope of this
    chapter.

## The Hazard Function

While the survival function answers "what fraction survives past time $t$?",
the **hazard function** answers a more local question: "given that a subject
has survived to time $t$, how likely is the event in the next instant?"

The hazard function is defined as

$$
h(t) = \lim_{\Delta t \to 0} \frac{P(t \leq T < t + \Delta t \mid T \geq t)}{\Delta t}
$$

This is not a probability (it can exceed 1) but an instantaneous rate.  It
measures the intensity of the event at time $t$ among those still at risk.

**Relationship to the density and survival function.**  Using the definition of
conditional probability,

$$
h(t) = \frac{f(t)}{S(t)}
$$

where $f(t) = F'(t) = -S'(t)$ is the probability density function.

!!! example "Interpreting the Hazard"

    Suppose $h(t) = 0.03$ per month at $t = 24$ months.  Among subjects who
    have survived to month 24, approximately 3% will experience the event in
    the next month.  This interpretation holds for small time increments:
    $P(T \in [t, t + \Delta t) \mid T \geq t) \approx h(t) \cdot \Delta t$.

## The Cumulative Hazard Function

The **cumulative hazard function** accumulates the instantaneous hazard over
time:

$$
H(t) = \int_0^t h(u)\,du
$$

The cumulative hazard has a useful interpretation: it measures the total amount
of risk that has been accumulated by time $t$.  Unlike $h(t)$, which is a rate,
$H(t)$ is a dimensionless quantity that ranges from 0 to $\infty$.

## Relationships Among the Four Functions

The survival function, hazard function, cumulative hazard, and density are all
deterministic transformations of one another.  Specifying any one of them
determines the other three.

**From $S(t)$ to $H(t)$.**  Starting from $h(t) = -S'(t)/S(t)$, integrate both
sides:

$$
H(t) = -\ln S(t)
$$

**From $H(t)$ to $S(t)$.**  Exponentiating the above:

$$
S(t) = \exp\!\bigl(-H(t)\bigr) = \exp\!\left(-\int_0^t h(u)\,du\right)
$$

**From $h(t)$ to $f(t)$.**  Combining $f(t) = h(t) S(t)$:

$$
f(t) = h(t) \exp\!\bigl(-H(t)\bigr)
$$

The following table summarizes the conversions.

| Given | $S(t)$ | $h(t)$ | $H(t)$ | $f(t)$ |
|:------|:-------|:-------|:--------|:-------|
| $S(t)$ | --- | $-S'(t)/S(t)$ | $-\ln S(t)$ | $-S'(t)$ |
| $h(t)$ | $e^{-\int_0^t h}$ | --- | $\int_0^t h$ | $h \cdot e^{-\int_0^t h}$ |
| $H(t)$ | $e^{-H(t)}$ | $H'(t)$ | --- | $H'(t) e^{-H(t)}$ |
| $f(t)$ | $1 - \int_0^t f$ | $f/(1-\int_0^t f)$ | $-\ln(1 - \int_0^t f)$ | --- |

## Shapes of the Hazard Function

Different event-generating processes produce different hazard shapes.  The
shape of $h(t)$ is often the primary object of scientific interest.

- **Constant hazard:** $h(t) = \lambda$.  The event rate does not change over
  time.  This corresponds to the exponential distribution and implies the
  memoryless property.
- **Increasing hazard:** $h(t)$ grows with $t$.  Components wear out over time
  (aging, fatigue).
- **Decreasing hazard:** $h(t)$ decreases with $t$.  Early failures are most
  likely; survivors become more robust ("burn-in" effects).
- **Bathtub curve:** $h(t)$ decreases initially (infant mortality), remains
  roughly constant (useful life), then increases (wear-out).  Common in
  reliability engineering.
- **Hump-shaped hazard:** $h(t)$ rises to a peak and then declines.  Seen in
  some disease processes where risk peaks shortly after diagnosis.

The parametric models in Section 21.3 formalize these shapes: the exponential
model captures constant hazard, the Weibull model captures monotone hazard,
and the log-normal and log-logistic models capture hump-shaped hazard.

## Worked Example

Consider the exponential distribution with rate $\lambda > 0$.  Its density is
$f(t) = \lambda e^{-\lambda t}$ for $t \geq 0$.

**Survival function:**

$$
S(t) = P(T > t) = \int_t^{\infty} \lambda e^{-\lambda u}\,du = e^{-\lambda t}
$$

**Hazard function:**

$$
h(t) = \frac{f(t)}{S(t)} = \frac{\lambda e^{-\lambda t}}{e^{-\lambda t}} = \lambda
$$

The hazard is constant, confirming the memoryless property.

**Cumulative hazard:**

$$
H(t) = \int_0^t \lambda\,du = \lambda t
$$

**Verification:** $S(t) = e^{-H(t)} = e^{-\lambda t}$, which matches.

??? tip "Why the Hazard Function Matters"

    In many applications the hazard function carries more scientific meaning
    than the survival function.  A clinician wants to know whether the risk of
    relapse increases or decreases over time.  A credit analyst wants to know
    whether default risk peaks in the first year or grows steadily.  The
    survival function answers cumulative questions; the hazard function answers
    instantaneous questions about risk dynamics.
