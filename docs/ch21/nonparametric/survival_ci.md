# Confidence Intervals for Survival Curves

The Kaplan--Meier estimator $\hat{S}(t)$ is a point estimate of the survival
function.  Like any estimator, it is subject to sampling variability: a
different sample would produce a different curve.  To quantify this uncertainty,
we need confidence intervals at each time point.

This section derives the variance of the Kaplan--Meier estimator using
Greenwood's formula, constructs pointwise confidence intervals, and discusses
transformations that improve coverage in small samples.

## Greenwood's Formula

The variance of $\hat{S}(t)$ is estimated by **Greenwood's formula** (1926):

$$
\widehat{\text{Var}}\bigl(\hat{S}(t)\bigr) = \hat{S}(t)^2 \sum_{j:\, t_{(j)} \leq t} \frac{d_j}{n_j(n_j - d_j)}
$$

where the sum runs over all event times $t_{(j)}$ up to $t$.

**Derivation sketch.**  The Kaplan--Meier estimator is a product of independent
factors $(1 - d_j / n_j)$.  Applying the delta method to the logarithm of the
product:

$$
\ln \hat{S}(t) = \sum_{j:\, t_{(j)} \leq t} \ln\!\left(1 - \frac{d_j}{n_j}\right)
$$

Each term has approximate variance $d_j / [n_j(n_j - d_j)]$ (from the binomial
variance of $d_j$ given $n_j$).  Summing and applying the delta method back to
the original scale gives Greenwood's formula.

The standard error of $\hat{S}(t)$ is

$$
\text{se}\bigl(\hat{S}(t)\bigr) = \hat{S}(t) \sqrt{\sum_{j:\, t_{(j)} \leq t} \frac{d_j}{n_j(n_j - d_j)}}
$$

## Pointwise Linear Confidence Interval

The simplest confidence interval uses a normal approximation on the original
scale:

$$
\hat{S}(t) \pm z_{\alpha/2} \cdot \text{se}\bigl(\hat{S}(t)\bigr)
$$

where $z_{\alpha/2}$ is the upper $\alpha/2$ quantile of the standard normal
distribution.

!!! warning "Limitations of the Linear Interval"

    This interval can produce values outside $[0, 1]$, especially when
    $\hat{S}(t)$ is close to 0 or 1.  It also tends to have below-nominal
    coverage in small samples because the normal approximation to the
    distribution of $\hat{S}(t)$ is poor near the boundaries.

## Log Transformation

A better approach applies a log transformation before constructing the interval.
Define $\theta(t) = \ln \hat{S}(t)$.  By the delta method,

$$
\text{se}\bigl(\theta(t)\bigr) = \frac{\text{se}(\hat{S}(t))}{\hat{S}(t)} = \sqrt{\sum_{j:\, t_{(j)} \leq t} \frac{d_j}{n_j(n_j - d_j)}}
$$

The $100(1 - \alpha)\%$ confidence interval for $\ln S(t)$ is

$$
\ln \hat{S}(t) \pm z_{\alpha/2} \cdot \text{se}\bigl(\theta(t)\bigr)
$$

Exponentiating the endpoints gives the confidence interval for $S(t)$:

$$
\left[\hat{S}(t)^{\exp(c)},\; \hat{S}(t)^{\exp(-c)}\right]
$$

where $c = z_{\alpha/2} \cdot \text{se}(\theta(t)) / \ln \hat{S}(t)$.

Wait---a cleaner way to express this is:

$$
\left[\exp\!\bigl(\ln \hat{S}(t) - z_{\alpha/2} \cdot \text{se}(\theta)\bigr),\; \exp\!\bigl(\ln \hat{S}(t) + z_{\alpha/2} \cdot \text{se}(\theta)\bigr)\right]
$$

This interval is always contained in $(0, 1]$, which is an improvement over the
linear interval.

## Log-Log Transformation

The **log-log transformation** (also called the complementary log-log
transformation) provides the best coverage in practice.  Define

$$
\phi(t) = \ln\!\bigl(-\ln \hat{S}(t)\bigr) = \ln \hat{H}(t)
$$

By the delta method,

$$
\text{se}\bigl(\phi(t)\bigr) = \frac{1}{\ln \hat{S}(t)} \sqrt{\sum_{j:\, t_{(j)} \leq t} \frac{d_j}{n_j(n_j - d_j)}}
$$

Construct the interval for $\phi(t)$:

$$
\phi(t) \pm z_{\alpha/2} \cdot \text{se}\bigl(\phi(t)\bigr)
$$

Then back-transform to the survival scale.  If $(\phi_L, \phi_U)$ is the
interval for $\phi(t)$, the confidence interval for $S(t)$ is

$$
\left[\exp\!\bigl(-e^{\phi_U}\bigr),\; \exp\!\bigl(-e^{\phi_L}\bigr)\right]
$$

Note the reversal: the upper bound for $\phi$ gives the lower bound for $S$.

!!! tip "Default in Most Software"

    The log-log transformation is the default method for Kaplan--Meier
    confidence intervals in most statistical software (R's `survfit`,
    Python's `lifelines`).  It consistently provides closer-to-nominal
    coverage than the linear or log methods.

## Worked Example

Continuing the example from the Kaplan--Meier section, at $t = 3$:

- $\hat{S}(3) = 0.729$
- Greenwood's sum: $\frac{1}{8 \cdot 7} + \frac{1}{6 \cdot 5} = 0.01786 + 0.03333 = 0.05119$
- $\text{se}(\hat{S}(3)) = 0.729 \sqrt{0.05119} = 0.729 \times 0.2263 = 0.165$

**Linear 95% CI:**

$$
0.729 \pm 1.96 \times 0.165 = (0.406,\; 1.052)
$$

The upper bound exceeds 1, illustrating the limitation of the linear method.

**Log-log 95% CI:**

$$
\phi(3) = \ln(-\ln 0.729) = \ln(0.3161) = -1.152
$$

$$
\text{se}(\phi) = \frac{\sqrt{0.05119}}{|\ln 0.729|} = \frac{0.2263}{0.3161} = 0.716
$$

$$
\phi(3) \pm 1.96 \times 0.716 = (-2.555,\; 0.251)
$$

Back-transforming:

$$
S_L = \exp(-e^{0.251}) = \exp(-1.285) = 0.277
$$

$$
S_U = \exp(-e^{-2.555}) = \exp(-0.0778) = 0.925
$$

The log-log interval $(0.277, 0.925)$ stays within $[0, 1]$ and is the
recommended interval.

## Comparison of Methods

| Method | Formula Basis | Stays in $[0,1]$ | Small-Sample Coverage |
|:-------|:-------------|:-----------------:|:---------------------:|
| Linear | $\hat{S} \pm z \cdot \text{se}$ | No | Below nominal |
| Log | $\exp(\ln \hat{S} \pm z \cdot \text{se}_\theta)$ | Yes (in $(0,1]$) | Moderate |
| Log-log | Back-transform of $\ln(-\ln \hat{S}) \pm z \cdot \text{se}_\phi$ | Yes | Best |

## Simultaneous Confidence Bands

The intervals above are **pointwise**: each covers $S(t_0)$ at a single fixed
$t_0$ with probability $1 - \alpha$.  A **simultaneous confidence band** covers
the entire curve $S(t)$ for all $t$ in an interval $[t_L, t_U]$
simultaneously.

The Hall--Wellner band and the equal-precision (EP) band are two common
approaches.  Both are wider than pointwise intervals because they must account
for the multiple comparisons across all time points.

??? note "When to Use Simultaneous Bands"

    Simultaneous bands are needed when the analyst wants to make statements
    about the entire survival curve (e.g., "the true curve lies within this
    band at all time points").  For reporting survival at a specific landmark
    time (e.g., 5-year survival), pointwise intervals suffice.

## Exercises

**Exercise 1.**
Greenwood's Formula

Using the Kaplan--Meier estimates from Exercise 2, compute the standard error
of $\hat{S}(8)$ using Greenwood's formula and construct a 95% pointwise
confidence interval using the linear method.

??? success "Solution to Exercise 1"

    $\hat{S}(8) = 0.675$. Greenwood's sum through $t = 8$:

    $$
    \frac{1}{10 \times 9} + \frac{1}{8 \times 7} + \frac{1}{7 \times 6} = 0.0111 + 0.0179 + 0.0238 = 0.0528
    $$

    $$
    \text{se}(\hat{S}(8)) = 0.675 \times \sqrt{0.0528} = 0.675 \times 0.2298 = 0.155
    $$

    95% CI (linear): $0.675 \pm 1.96 \times 0.155 = (0.371, 0.979)$.
