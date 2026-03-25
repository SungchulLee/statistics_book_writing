# Kaplan-Meier Estimator

Estimating the survival function $S(t)$ from observed data is the first task
in any survival analysis.  When no distributional assumptions are made about
the event times, the standard tool is the **Kaplan--Meier estimator**, also
called the **product-limit estimator**.  Introduced by Kaplan and Meier (1958),
it constructs a non-parametric estimate of $S(t)$ that correctly handles
right-censored observations.

This section derives the estimator, discusses its properties, and works through
a numerical example.

## Setup and Notation

Suppose $n$ subjects are observed, producing ordered distinct event times

$$
t_{(1)} < t_{(2)} < \cdots < t_{(K)}
$$

At each event time $t_{(j)}$, define:

- $d_j$ = number of events (deaths, defaults, etc.) at time $t_{(j)}$.
- $n_j$ = number of subjects **at risk** just before time $t_{(j)}$ --- that is,
  subjects who have neither experienced the event nor been censored before
  $t_{(j)}$.

Subjects who are censored at exactly $t_{(j)}$ are conventionally included in
the risk set $n_j$ (they were still at risk just before the event time).

## Derivation

The key idea is to decompose the survival probability into a product of
conditional probabilities.  The probability of surviving past time $t_{(j)}$,
given survival to just before $t_{(j)}$, is estimated by

$$
\hat{P}\bigl(T > t_{(j)} \mid T \geq t_{(j)}\bigr) = 1 - \frac{d_j}{n_j}
$$

The survival function at any time $t$ is the product of these conditional
survival probabilities over all event times up to $t$:

$$
\hat{S}(t) = \prod_{j:\, t_{(j)} \leq t} \left(1 - \frac{d_j}{n_j}\right)
$$

This is the **Kaplan--Meier estimator**.  It produces a step function that
decreases only at observed event times.

!!! note "Role of Censored Observations"

    Censored subjects do not cause a drop in $\hat{S}(t)$, but they reduce the
    risk set $n_j$ at subsequent event times.  This is how the Kaplan--Meier
    estimator incorporates partial information from censored observations
    without treating them as events.

## Properties

1. **Non-parametric.** No distributional assumption is required.
2. **Maximum likelihood.** The Kaplan--Meier estimator is the non-parametric
   maximum likelihood estimator of $S(t)$.
3. **Consistency.** $\hat{S}(t) \xrightarrow{P} S(t)$ as $n \to \infty$ under
   independent censoring.
4. **Step function.** $\hat{S}(t)$ is constant between event times and drops
   at each event time.
5. **Right-continuous.** By convention, $\hat{S}(t)$ is a right-continuous
   function with left limits (cadlag).

## Handling Ties

When multiple events and censorings occur at the same time, the convention is:

1. Events at time $t_{(j)}$ are processed first (they contribute to $d_j$).
2. Censorings at time $t_{(j)}$ are processed after (they reduce the risk set
   for subsequent times but are included in $n_j$).

This convention ensures that censored subjects at time $t_{(j)}$ are counted
as at risk at that time.

## Worked Example

Consider 8 subjects with the following observed times and event indicators:

| Subject | Time | $\delta$ |
|:-------:|:----:|:--------:|
| 1 | 1 | 1 |
| 2 | 2 | 0 |
| 3 | 3 | 1 |
| 4 | 4 | 0 |
| 5 | 5 | 1 |
| 6 | 5 | 1 |
| 7 | 7 | 0 |
| 8 | 9 | 1 |

The distinct event times are $t_{(1)} = 1$, $t_{(2)} = 3$, $t_{(3)} = 5$,
$t_{(4)} = 9$.

**Step-by-step computation:**

| $t_{(j)}$ | $n_j$ | $d_j$ | $1 - d_j/n_j$ | $\hat{S}(t_{(j)})$ |
|:----------:|:-----:|:-----:|:--------------:|:-------------------:|
| 1 | 8 | 1 | 7/8 = 0.875 | 0.875 |
| 3 | 6 | 1 | 5/6 = 0.833 | 0.875 $\times$ 0.833 = 0.729 |
| 5 | 4 | 2 | 2/4 = 0.500 | 0.729 $\times$ 0.500 = 0.365 |
| 9 | 1 | 1 | 0/1 = 0.000 | 0.365 $\times$ 0.000 = 0.000 |

At $t_{(2)} = 3$: the risk set is $n_2 = 6$ because subject 1 experienced the
event at $t = 1$ and subject 2 was censored at $t = 2$, leaving 6 subjects.

At $t_{(3)} = 5$: the risk set is $n_3 = 4$ because subjects 1, 2, 3, 4 have
left (2 events and 2 censorings).

The Kaplan--Meier estimate is:

- $\hat{S}(t) = 1.000$ for $t < 1$
- $\hat{S}(t) = 0.875$ for $1 \leq t < 3$
- $\hat{S}(t) = 0.729$ for $3 \leq t < 5$
- $\hat{S}(t) = 0.365$ for $5 \leq t < 9$
- $\hat{S}(t) = 0.000$ for $t \geq 9$

## Median Survival Time

The **median survival time** is the time $\hat{t}_{0.5}$ at which the
Kaplan--Meier curve first crosses (or reaches) 0.5:

$$
\hat{t}_{0.5} = \inf\{t : \hat{S}(t) \leq 0.5\}
$$

In the worked example, $\hat{S}(5) = 0.365 < 0.5$ and $\hat{S}(3) = 0.729 > 0.5$,
so $\hat{t}_{0.5} = 5$ months.

!!! warning "Median May Not Exist"

    If the Kaplan--Meier curve does not drop below 0.5 (e.g., heavy censoring
    causes the curve to plateau above 0.5), the median survival time is
    undefined.  The mean survival time can be estimated as the area under the
    Kaplan--Meier curve, but it requires extrapolation if the curve does not
    reach zero.

## Limitations

- The Kaplan--Meier estimator handles only right-censored data.  Interval
  censoring and left truncation require other methods.
- It is a univariate method: it estimates $S(t)$ without adjusting for
  covariates.  Covariate-adjusted survival curves require the Cox model
  (Section 21.4) or stratified Kaplan--Meier estimates.
- At late time points where few subjects remain at risk, the estimate has
  high variance.  Confidence intervals (Section 21.2) quantify this
  uncertainty.

??? tip "Connection to the Nelson--Aalen Estimator"

    The Nelson--Aalen estimator (next section) provides an alternative
    non-parametric estimate via the cumulative hazard.  For large samples the
    two estimators are nearly identical, since
    $\hat{S}_{\text{KM}}(t) \approx \exp(-\hat{H}_{\text{NA}}(t))$.
