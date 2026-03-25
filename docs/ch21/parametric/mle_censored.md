# Maximum Likelihood for Censored Data

The previous sections introduced several parametric survival models---exponential,
Weibull, log-normal, and log-logistic---and derived the MLE for the exponential
case.  This section develops the **general likelihood framework for censored
data**, which underpins parameter estimation in all parametric survival models.

The key insight is that observed events and censored observations contribute
different factors to the likelihood: an event at time $t_i$ contributes the
density $f(t_i)$, while a censoring at time $t_i$ contributes the survival
probability $S(t_i)$.

## The Censored-Data Likelihood

Let $(t_1, \delta_1), \ldots, (t_n, \delta_n)$ denote the observed data, where
$t_i$ is the observed time and $\delta_i \in \{0, 1\}$ is the event indicator
($\delta_i = 1$ for events, $\delta_i = 0$ for censored observations).

Under the **independent censoring assumption**, the likelihood for parameter
vector $\boldsymbol{\theta}$ is

$$
L(\boldsymbol{\theta}) = \prod_{i=1}^{n} \bigl[f(t_i; \boldsymbol{\theta})\bigr]^{\delta_i} \bigl[S(t_i; \boldsymbol{\theta})\bigr]^{1-\delta_i}
$$

**Interpretation:**

- When $\delta_i = 1$: the subject experienced the event at exactly $t_i$, so
  the contribution is the density $f(t_i)$.
- When $\delta_i = 0$: the subject was censored at $t_i$, so all we know is
  $T_i > t_i$; the contribution is $P(T_i > t_i) = S(t_i)$.

!!! note "Why Not Just Use the Density?"

    Dropping censored observations (using only events) wastes information and
    biases estimates toward shorter survival times.  Including censored
    observations with their $S(t_i)$ contribution retains the information that
    these subjects survived at least to $t_i$.

## Log-Likelihood

Using $f(t) = h(t) S(t)$ and $S(t) = \exp(-H(t))$, the likelihood becomes

$$
L(\boldsymbol{\theta}) = \prod_{i=1}^{n} \bigl[h(t_i)\bigr]^{\delta_i} \exp\!\bigl(-H(t_i)\bigr)
$$

The log-likelihood is

$$
\ell(\boldsymbol{\theta}) = \sum_{i=1}^{n} \left[\delta_i \ln h(t_i; \boldsymbol{\theta}) - H(t_i; \boldsymbol{\theta})\right]
$$

This separable form is computationally convenient: the hazard and cumulative
hazard are the building blocks for any parametric model.

## Score Equations

The score function is the gradient of the log-likelihood:

$$
U(\boldsymbol{\theta}) = \frac{\partial \ell}{\partial \boldsymbol{\theta}} = \sum_{i=1}^{n} \left[\delta_i \frac{\partial \ln h(t_i)}{\partial \boldsymbol{\theta}} - \frac{\partial H(t_i)}{\partial \boldsymbol{\theta}}\right]
$$

The MLE $\hat{\boldsymbol{\theta}}$ solves $U(\hat{\boldsymbol{\theta}}) = \mathbf{0}$.

## Information Matrix

The observed Fisher information matrix is

$$
\mathcal{I}(\boldsymbol{\theta}) = -\frac{\partial^2 \ell}{\partial \boldsymbol{\theta}\, \partial \boldsymbol{\theta}^\top}
$$

evaluated at $\hat{\boldsymbol{\theta}}$.  The asymptotic covariance of the MLE
is $\text{Var}(\hat{\boldsymbol{\theta}}) \approx \mathcal{I}(\hat{\boldsymbol{\theta}})^{-1}$.

Confidence intervals for individual parameters are

$$
\hat{\theta}_j \pm z_{\alpha/2} \sqrt{[\mathcal{I}^{-1}]_{jj}}
$$

## Numerical Optimization

Except for the exponential model, the score equations for survival models do
not have closed-form solutions.  Standard numerical methods include:

1. **Newton--Raphson.** Uses the Hessian (observed information) to update the
   parameter estimate at each iteration:

$$
\boldsymbol{\theta}^{(m+1)} = \boldsymbol{\theta}^{(m)} + \mathcal{I}(\boldsymbol{\theta}^{(m)})^{-1} U(\boldsymbol{\theta}^{(m)})
$$

2. **Fisher scoring.** Replaces the observed information with the expected
   information.  Equivalent to Newton--Raphson when the model is correctly
   specified.

3. **Quasi-Newton methods (BFGS, L-BFGS).** Approximate the Hessian to avoid
   computing second derivatives.  Widely used in software packages.

!!! tip "Starting Values"

    Good initial values accelerate convergence.  For the Weibull model, start
    with $k = 1$ (exponential) and $\lambda = \sum t_i / d$.  For log-normal
    and log-logistic models, use the sample mean and variance of $\ln t_i$
    among uncensored observations.

## Example: Weibull Likelihood

For the Weibull model with parameters $(k, \lambda)$:

$$
h(t) = \frac{k}{\lambda}\left(\frac{t}{\lambda}\right)^{k-1}, \qquad H(t) = \left(\frac{t}{\lambda}\right)^k
$$

The log-likelihood is

$$
\ell(k, \lambda) = \sum_{i=1}^{n} \delta_i \left[\ln k - k \ln \lambda + (k-1)\ln t_i\right] - \sum_{i=1}^{n}\left(\frac{t_i}{\lambda}\right)^k
$$

$$
= d \ln k - dk \ln \lambda + (k-1)\sum_{i=1}^{n}\delta_i \ln t_i - \sum_{i=1}^{n}\left(\frac{t_i}{\lambda}\right)^k
$$

## Model Comparison via Likelihood

Parametric models can be compared using likelihood-based criteria:

**Likelihood ratio test** (for nested models).  To test the exponential ($k=1$)
against the Weibull ($k$ free):

$$
\Lambda = 2[\ell_{\text{Weibull}} - \ell_{\text{Exp}}] \;\xrightarrow{d}\; \chi^2_1
$$

**Akaike Information Criterion** (for non-nested models):

$$
\text{AIC} = -2\ell(\hat{\boldsymbol{\theta}}) + 2p
$$

where $p$ is the number of parameters.  Lower AIC indicates a better trade-off
between fit and complexity.  Section 21.5 discusses AIC and other selection
criteria in detail.

## Handling Left and Interval Censoring

The likelihood extends to other censoring types:

- **Left censoring** ($T_i < t_i$): contributes $F(t_i) = 1 - S(t_i)$.
- **Interval censoring** ($L_i < T_i \leq R_i$): contributes
  $S(L_i) - S(R_i) = F(R_i) - F(L_i)$.

The general likelihood for mixed censoring types is

$$
L(\boldsymbol{\theta}) = \prod_{i \in \mathcal{E}} f(t_i) \prod_{i \in \mathcal{R}} S(t_i) \prod_{i \in \mathcal{L}} F(t_i) \prod_{i \in \mathcal{I}} [S(L_i) - S(R_i)]
$$

where $\mathcal{E}$, $\mathcal{R}$, $\mathcal{L}$, and $\mathcal{I}$ denote
the sets of exact, right-censored, left-censored, and interval-censored
observations, respectively.

!!! warning "Identifiability with Heavy Censoring"

    When the censoring fraction is very high (e.g., more than 80% of
    observations are censored), the likelihood surface becomes flat and the
    MLE may be unreliable.  In such cases, consider increasing the sample
    size, extending the follow-up period, or using Bayesian methods with
    informative priors.
