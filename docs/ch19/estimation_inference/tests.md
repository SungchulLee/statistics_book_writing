# Wald and Likelihood Ratio Tests

## Overview

After fitting a logistic regression model by maximum likelihood, we
typically want to test whether individual coefficients (or groups of
coefficients) are significantly different from zero.  Two classical
approaches are the **Wald test** and the **likelihood ratio test (LRT)**.

## Wald Test

### Idea

Under regularity conditions, the MLE
$\hat{\boldsymbol{\theta}}$ is asymptotically normal:

$$
\hat{\theta}_j \;\stackrel{a}{\sim}\;
  \mathcal{N}\!\bigl(\theta_j,\;[\mathcal{I}(\boldsymbol{\theta})^{-1}]_{jj}\bigr)
$$

where $\mathcal{I}$ is the Fisher information matrix.  For logistic
regression $\mathcal{I}(\boldsymbol{\theta}) = A^TBA$ (the Hessian of
the cross-entropy loss).

### Test Statistic

To test $H_0\colon\theta_j=0$:

$$
W_j = \frac{\hat{\theta}_j}{\operatorname{se}(\hat{\theta}_j)},
\qquad
\operatorname{se}(\hat{\theta}_j) = \sqrt{[(A^TBA)^{-1}]_{jj}}
$$

Under $H_0$, $W_j\sim\mathcal{N}(0,1)$ (or equivalently $W_j^2\sim\chi^2_1$).

### Interpretation

The Wald test is reported by default in most software (e.g., the
`summary` output of `statsmodels` or R's `glm`).  It is quick to
compute because it only requires the fitted model, but it can be
unreliable when the MLE is far from the null or when the sample is
small.

## Likelihood Ratio Test (LRT)

### Idea

Compare the maximized log-likelihood of the full model to that of a
restricted (nested) model:

$$
\Lambda = -2\bigl[\ell(\hat{\boldsymbol{\theta}}_{\text{restricted}})
                  - \ell(\hat{\boldsymbol{\theta}}_{\text{full}})\bigr]
$$

Under $H_0$ (the restrictions hold), $\Lambda\sim\chi^2_q$ where $q$ is
the number of restrictions.

### Single Coefficient

To test $H_0\colon\theta_j=0$, fit the model with and without feature
$j$:

$$
\Lambda = -2\bigl[\ell_{\text{without }j} - \ell_{\text{with }j}\bigr]
\sim \chi^2_1
$$

### Multiple Coefficients

The LRT generalizes naturally.  To test whether a group of $q$
coefficients is jointly zero, $\Lambda\sim\chi^2_q$.

## Comparison

| | Wald Test | Likelihood Ratio Test |
|---|---|---|
| Models fitted | 1 (full only) | 2 (full + restricted) |
| Computational cost | Low | Higher |
| Small-sample behavior | Can be unreliable | Generally more reliable |
| Software default | Often reported automatically | Requires explicit comparison |

In practice the LRT is preferred for formal hypothesis testing, while
the Wald statistic is convenient for quick screening of individual
coefficients.

## Connection to the Hessian

Both tests rely on the curvature of the log-likelihood at the MLE.
Recall the Hessian derived earlier:

$$
\nabla^2\ell = A^TBA,
\qquad
B = \operatorname{diag}\!\bigl(\sigma^{(i)}(1-\sigma^{(i)})\bigr)
$$

The inverse of the Hessian provides the asymptotic covariance matrix of
$\hat{\boldsymbol{\theta}}$.  The Wald test uses diagonal entries of
this inverse, while the LRT uses the difference in log-likelihoods
evaluated at two points.

## Example in Python

```python
import numpy as np
import statsmodels.api as sm

# Fit full model
X_full = sm.add_constant(X)
model_full = sm.Logit(y, X_full).fit(disp=0)
print(model_full.summary())  # Wald z-statistics shown by default

# LRT: compare full vs restricted (drop last feature)
model_restricted = sm.Logit(y, X_full[:, :-1]).fit(disp=0)
lr_stat = -2 * (model_restricted.llf - model_full.llf)
p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
```
