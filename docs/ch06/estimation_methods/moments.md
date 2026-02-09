# Method of Moments

## Introduction

The **Method of Moments (MoM)** is one of the oldest and most intuitive approaches to parameter estimation. The idea is simple: equate population moments (which are functions of unknown parameters) to their sample counterparts, then solve for the parameters. This yields estimators that are easy to compute, often available in closed form, and provide good starting points for more sophisticated methods like MLE.

While MLE is generally more efficient, the Method of Moments remains widely used in practice — particularly when likelihoods are intractable, as a quick preliminary estimator, or in the Generalized Method of Moments (GMM) framework that dominates empirical finance and econometrics.

## Definitions

### Population Moments

The $k$-th **population moment** (raw moment) of a random variable $X$ with distribution $f(x; \theta)$ is:

$$\mu_k' = E[X^k] = \int x^k f(x; \theta) \, dx$$

The $k$-th **central moment** is:

$$\mu_k = E[(X - \mu)^k]$$

where $\mu = E[X] = \mu_1'$.

These moments are functions of the unknown parameter $\theta$:
- $\mu_1'(\theta) = E_\theta[X]$ (mean)
- $\mu_2'(\theta) = E_\theta[X^2]$ (second raw moment)
- $\mu_2(\theta) = \text{Var}_\theta(X)$ (variance)
- $\mu_3(\theta)$ relates to skewness
- $\mu_4(\theta)$ relates to kurtosis

### Sample Moments

The $k$-th **sample moment** (raw) is:

$$m_k' = \frac{1}{n}\sum_{i=1}^n X_i^k$$

The $k$-th **sample central moment** is:

$$m_k = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^k$$

By the Law of Large Numbers, $m_k' \xrightarrow{P} \mu_k'$ as $n \to \infty$.

## The Method of Moments Procedure

### General Recipe

Suppose the distribution depends on $p$ unknown parameters $\theta = (\theta_1, \ldots, \theta_p)^T$. The Method of Moments proceeds as follows:

**Step 1.** Express the first $p$ population moments as functions of the parameters:

$$\mu_k'(\theta) = E_\theta[X^k], \quad k = 1, 2, \ldots, p$$

**Step 2.** Set population moments equal to sample moments:

$$\mu_k'(\theta) = m_k', \quad k = 1, 2, \ldots, p$$

**Step 3.** Solve the system of $p$ equations in $p$ unknowns for $\hat{\theta}_1, \ldots, \hat{\theta}_p$.

The resulting estimators $\hat{\theta}_{\text{MoM}}$ are the **Method of Moments estimators**.

### When to Use Central Moments

Sometimes it is more convenient to match central moments instead of (or in addition to) raw moments. For example, if $\theta = (\mu, \sigma^2)$, one can set:
- $E[X] = \bar{X}$ (first raw moment)
- $\text{Var}(X) = m_2$ (second central moment)

This is equivalent and often simplifies the algebra.

## Worked Examples

### Example 1: Normal Distribution

Let $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$. Two unknown parameters require two moment equations.

**Population moments:**
- $\mu_1' = E[X] = \mu$
- $\mu_2' = E[X^2] = \sigma^2 + \mu^2$

**Moment equations:**

$$\mu = m_1' = \bar{X}$$

$$\sigma^2 + \mu^2 = m_2' = \frac{1}{n}\sum X_i^2$$

**Solution:**

$$\hat{\mu}_{\text{MoM}} = \bar{X}$$

$$\hat{\sigma}^2_{\text{MoM}} = m_2' - (m_1')^2 = \frac{1}{n}\sum X_i^2 - \bar{X}^2 = \frac{1}{n}\sum (X_i - \bar{X})^2$$

Note that $\hat{\sigma}^2_{\text{MoM}}$ divides by $n$ (biased), just like the MLE. In this case, MoM and MLE give the same estimators.

### Example 2: Exponential Distribution

Let $X_1, \ldots, X_n \sim \text{Exp}(\lambda)$ with $E[X] = 1/\lambda$. One parameter requires one moment equation.

**Moment equation:**

$$\frac{1}{\lambda} = \bar{X}$$

**Solution:**

$$\hat{\lambda}_{\text{MoM}} = \frac{1}{\bar{X}}$$

This coincides with the MLE.

### Example 3: Gamma Distribution

Let $X_1, \ldots, X_n \sim \text{Gamma}(\alpha, \beta)$ with density $f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$. Two parameters require two equations.

**Population moments:**
- $E[X] = \alpha/\beta$
- $\text{Var}(X) = \alpha/\beta^2$

**Moment equations:**

$$\frac{\alpha}{\beta} = \bar{X}, \qquad \frac{\alpha}{\beta^2} = \frac{1}{n}\sum (X_i - \bar{X})^2$$

**Solution:** From the ratio $\text{Var}(X)/E[X] = 1/\beta$:

$$\hat{\beta}_{\text{MoM}} = \frac{\bar{X}}{m_2}, \qquad \hat{\alpha}_{\text{MoM}} = \frac{\bar{X}^2}{m_2}$$

where $m_2 = \frac{1}{n}\sum(X_i - \bar{X})^2$.

The MoM estimators are available in closed form, while the MLE for the Gamma distribution requires numerical optimization — a key practical advantage.

### Example 4: Uniform Distribution

Let $X_1, \ldots, X_n \sim \text{Uniform}(a, b)$. Two parameters require two equations.

**Population moments:**
- $E[X] = (a + b)/2$
- $\text{Var}(X) = (b - a)^2/12$

**Moment equations:**

$$\frac{a + b}{2} = \bar{X}, \qquad \frac{(b-a)^2}{12} = m_2$$

**Solution:**

$$\hat{a}_{\text{MoM}} = \bar{X} - \sqrt{3 m_2}, \qquad \hat{b}_{\text{MoM}} = \bar{X} + \sqrt{3 m_2}$$

**Note:** These estimators can fall inside the range of the data (i.e., $\hat{a} > \min(X_i)$ or $\hat{b} < \max(X_i)$), which is logically inconsistent. This is a known limitation of MoM — it does not always respect the support constraints of the distribution. The MLE ($\hat{a} = \min(X_i)$, $\hat{b} = \max(X_i)$) does not have this problem.

### Example 5: Beta Distribution

Let $X_1, \ldots, X_n \sim \text{Beta}(\alpha, \beta)$.

**Population moments:**
- $E[X] = \frac{\alpha}{\alpha + \beta}$
- $\text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$

Let $\bar{x} = m_1'$ and $s^2 = m_2$ be the sample mean and variance. Solving:

$$\hat{\alpha}_{\text{MoM}} = \bar{x}\left(\frac{\bar{x}(1-\bar{x})}{s^2} - 1\right), \qquad \hat{\beta}_{\text{MoM}} = (1 - \bar{x})\left(\frac{\bar{x}(1-\bar{x})}{s^2} - 1\right)$$

This requires $s^2 < \bar{x}(1 - \bar{x})$, which holds for reasonable data.

## Properties of MoM Estimators

### Consistency

By the Law of Large Numbers, sample moments converge to population moments. If the function mapping moments to parameters is continuous, MoM estimators are **consistent** by the Continuous Mapping Theorem:

$$\hat{\theta}_{\text{MoM}} \xrightarrow{P} \theta_0 \quad \text{as } n \to \infty$$

### Asymptotic Normality

Under regularity conditions, MoM estimators are asymptotically normal. For a single parameter estimated from the first moment:

$$\sqrt{n}(\hat{\theta}_{\text{MoM}} - \theta_0) \xrightarrow{d} N(0, V)$$

The asymptotic variance $V$ is determined by the Delta Method applied to the function that maps moments to parameters.

### Efficiency

MoM estimators are generally **less efficient** than MLEs. The asymptotic relative efficiency is:

$$\text{ARE}(\hat{\theta}_{\text{MoM}}, \hat{\theta}_{\text{MLE}}) = \frac{\text{Var}_{\text{asymp}}(\hat{\theta}_{\text{MLE}})}{\text{Var}_{\text{asymp}}(\hat{\theta}_{\text{MoM}})} \leq 1$$

The efficiency loss can range from negligible (e.g., normal distribution, where MoM = MLE) to substantial (e.g., some heavy-tailed distributions).

### Comparison with MLE

| Property | Method of Moments | MLE |
|----------|------------------|-----|
| Computation | Often closed-form | May require numerical optimization |
| Efficiency | Less efficient (generally) | Asymptotically efficient |
| Consistency | Yes (under regularity) | Yes (under regularity) |
| Invariance | Not invariant in general | Invariant to reparameterization |
| Robustness | Sensitive to outliers in higher moments | Sensitive to model misspecification |
| Uniqueness | Not always unique | Usually unique (under regularity) |

## Generalized Method of Moments (GMM)

### Motivation

In many applications — especially in econometrics and finance — we have **more moment conditions than parameters**. GMM extends the Method of Moments to exploit these extra conditions optimally.

### Setup

Suppose we have $q$ moment conditions but only $p < q$ parameters. Define the **moment function** $g(X_i, \theta)$ such that at the true parameter $\theta_0$:

$$E[g(X_i, \theta_0)] = 0$$

The sample analog is:

$$\bar{g}_n(\theta) = \frac{1}{n}\sum_{i=1}^n g(X_i, \theta)$$

With $q > p$, we cannot set all $q$ sample moments exactly to zero. Instead, GMM minimizes a quadratic form:

$$\hat{\theta}_{\text{GMM}} = \arg\min_\theta \bar{g}_n(\theta)^T W \bar{g}_n(\theta)$$

where $W$ is a positive definite **weighting matrix**.

### Optimal Weighting Matrix

The **efficient GMM** estimator uses the optimal weighting matrix:

$$W^* = \left[E[g(X_i, \theta_0) g(X_i, \theta_0)^T]\right]^{-1} = S^{-1}$$

where $S$ is the long-run covariance matrix of the moment conditions. This yields the most efficient GMM estimator among all choices of $W$.

In practice, $S$ is unknown and estimated from the data, typically using a two-step procedure:
1. Estimate $\hat{\theta}^{(1)}$ with $W = I$ (identity matrix)
2. Estimate $\hat{S}$ using residuals from step 1
3. Re-estimate $\hat{\theta}^{(2)}$ with $W = \hat{S}^{-1}$

### Hansen's J-Test

With more moment conditions than parameters ($q > p$), the **overidentifying restrictions** can be tested. The **J-statistic** is:

$$J = n \cdot \bar{g}_n(\hat{\theta})^T \hat{S}^{-1} \bar{g}_n(\hat{\theta}) \xrightarrow{d} \chi^2_{q-p}$$

A large J-statistic suggests that the model's moment conditions are incompatible with the data.

## Connections to Finance

The Method of Moments and GMM are extensively used in quantitative finance:

- **Asset pricing**: GMM is the standard method for estimating and testing asset pricing models (CAPM, Fama-French). The Euler equation conditions $E[m_t R_t - 1] = 0$ provide the moment conditions, where $m_t$ is the stochastic discount factor.
- **GARCH estimation**: Quasi-MLE is standard, but MoM provides closed-form initial estimates for $(\omega, \alpha, \beta)$ by matching the unconditional variance and autocorrelation of squared returns.
- **Distribution fitting**: For fitting heavy-tailed distributions (Student-$t$, stable distributions) to return data, MoM provides quick estimates when the likelihood is complex or slow to evaluate.
- **Yield curve modeling**: GMM estimation of affine term structure models using yield-level or yield-change moments.
- **Realized volatility**: MoM estimators based on high-frequency return moments are used to estimate integrated volatility and its properties.
- **Portfolio theory**: Estimating expected returns and covariances from sample moments is the simplest (MoM) approach to portfolio optimization.

## Advanced Topics

### Method of L-Moments

**L-moments** are linear combinations of order statistics that provide an alternative to conventional moments. They are:
- More robust to outliers than conventional moments
- Always uniquely define a distribution (unlike conventional moments, which may not exist for heavy-tailed distributions)
- Particularly useful for fitting extreme value distributions in risk management

### Higher Moment Matching

For distributions with more than two parameters, higher moments (skewness, kurtosis) are matched:
- $\hat{\gamma} = m_3/m_2^{3/2}$ matches population skewness
- $\hat{\kappa} = m_4/m_2^2$ matches population kurtosis

However, higher sample moments are increasingly noisy, which is a practical limitation. The estimation variance of the $k$-th sample moment involves the $2k$-th moment of the distribution.

### Simulated Method of Moments (SMM)

When the theoretical moments $\mu_k'(\theta)$ cannot be computed analytically but the model can be simulated, **SMM** replaces population moments with simulated moments:

1. For a candidate $\theta$, simulate $S$ samples of size $n$ from the model
2. Compute the average simulated moments $\tilde{m}_k'(\theta)$
3. Choose $\theta$ to minimize the distance between sample and simulated moments

SMM is used for complex financial models where the likelihood is intractable (e.g., agent-based models, complex derivative pricing models).

## Summary

The Method of Moments provides estimators by equating sample moments to population moments. While generally less efficient than MLE, it offers computational simplicity, closed-form solutions, and the powerful GMM extension for overidentified models. In finance, GMM has become the workhorse for estimating and testing asset pricing models, while standard MoM remains valuable for quick estimation and initialization.

## Key Formulas

| Quantity | Formula |
|----------|---------|
| $k$-th sample moment | $m_k' = \frac{1}{n}\sum_{i=1}^n X_i^k$ |
| MoM equation | $\mu_k'(\theta) = m_k'$ for $k = 1, \ldots, p$ |
| GMM objective | $\min_\theta \bar{g}_n(\theta)^T W \bar{g}_n(\theta)$ |
| Optimal weight | $W^* = S^{-1}$ |
| J-test | $J = n \bar{g}_n(\hat{\theta})^T \hat{S}^{-1} \bar{g}_n(\hat{\theta}) \sim \chi^2_{q-p}$ |
