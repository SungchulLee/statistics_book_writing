# L1 and L2 Regularization for Logistic Regression

## Motivation

When the number of predictors $p$ is large relative to the sample size $n$,
the maximum likelihood estimator for logistic regression can overfit: the
coefficients grow large to separate the training classes as sharply as possible,
producing extreme predicted probabilities and poor generalization.
**Regularization** adds a penalty to the log-likelihood that shrinks the
coefficients toward zero, trading a small increase in bias for a large decrease
in variance.

## Penalized Log-Likelihood

The regularized estimator maximizes

$$
\ell_{\text{pen}}(\boldsymbol{\theta})
= \ell(\boldsymbol{\theta}) - \lambda\,\Omega(\boldsymbol{\theta})
$$

where $\ell(\boldsymbol{\theta})$ is the unpenalized log-likelihood,
$\lambda \ge 0$ controls the penalty strength, and
$\Omega(\boldsymbol{\theta})$ is a penalty function.  The intercept
$\theta_0$ is **not penalized** — only the slope coefficients
$\theta_1, \ldots, \theta_{p-1}$ are shrunk.

## L2 Penalty (Ridge Logistic Regression)

The L2 penalty is the sum of squared coefficients:

$$
\Omega_{\text{L2}}(\boldsymbol{\theta})
= \frac{1}{2}\sum_{j=1}^{p-1}\theta_j^2
= \frac{1}{2}\lVert\boldsymbol{\theta}_{-0}\rVert_2^2
$$

The penalized log-likelihood becomes

$$
\ell_{\text{ridge}}(\boldsymbol{\theta})
= \sum_{i=1}^{n}\bigl[y_i\,\mathbf{x}_i^T\boldsymbol{\theta}
  - \log(1+e^{\mathbf{x}_i^T\boldsymbol{\theta}})\bigr]
  - \frac{\lambda}{2}\lVert\boldsymbol{\theta}_{-0}\rVert_2^2
$$

Ridge regression shrinks all coefficients toward zero but sets none to exactly
zero.  It is especially useful when predictors are correlated
(multicollinearity), because it stabilizes the coefficient estimates.

### Bayesian Interpretation

The L2 penalty corresponds to a **Gaussian prior** on each coefficient:
$\theta_j \sim \mathcal{N}(0, 1/\lambda)$.  The ridge estimator is therefore
the **maximum a posteriori (MAP)** estimate under this prior.

## L1 Penalty (Lasso Logistic Regression)

The L1 penalty is the sum of absolute values:

$$
\Omega_{\text{L1}}(\boldsymbol{\theta})
= \sum_{j=1}^{p-1}|\theta_j|
= \lVert\boldsymbol{\theta}_{-0}\rVert_1
$$

The penalized log-likelihood becomes

$$
\ell_{\text{lasso}}(\boldsymbol{\theta})
= \sum_{i=1}^{n}\bigl[y_i\,\mathbf{x}_i^T\boldsymbol{\theta}
  - \log(1+e^{\mathbf{x}_i^T\boldsymbol{\theta}})\bigr]
  - \lambda\,\lVert\boldsymbol{\theta}_{-0}\rVert_1
$$

The L1 penalty produces **sparse** solutions: for large enough $\lambda$, some
coefficients are driven to exactly zero.  This makes L1 logistic regression a
simultaneous classifier and feature selector.

### Bayesian Interpretation

The L1 penalty corresponds to independent **Laplace priors**:
$\theta_j \sim \text{Laplace}(0, 1/\lambda)$.  The heavy tails and sharp peak
at zero encourage sparsity.

## Elastic Net

The elastic net combines both penalties:

$$
\Omega_{\text{EN}}(\boldsymbol{\theta})
= \alpha\,\lVert\boldsymbol{\theta}_{-0}\rVert_1
  + \frac{1-\alpha}{2}\lVert\boldsymbol{\theta}_{-0}\rVert_2^2
$$

where $\alpha \in [0,1]$ controls the mix.  Setting $\alpha = 1$ recovers the
lasso; $\alpha = 0$ recovers ridge.  The elastic net inherits the sparsity of
L1 while handling correlated predictors better than the pure lasso.

## The C Parameter in scikit-learn

Scikit-learn's `LogisticRegression` parameterizes the penalty strength as
$C = 1/\lambda$.  A **larger** $C$ means less regularization (closer to
unpenalized MLE); a **smaller** $C$ means stronger regularization.

| `LogisticRegression` argument | Equivalent |
|---|---|
| `penalty='l2', C=1.0` | Ridge with $\lambda = 1$ |
| `penalty='l1', C=0.1, solver='liblinear'` | Lasso with $\lambda = 10$ |
| `penalty='elasticnet', l1_ratio=0.5, C=1.0, solver='saga'` | Elastic net with $\alpha = 0.5$, $\lambda = 1$ |

!!! tip "Feature Standardization"
    Because the penalty treats all coefficients equally, predictors should be
    **standardized** (mean zero, unit variance) before fitting a regularized
    model.  Otherwise, coefficients on larger-scale features are penalized more
    heavily, producing biased results.

## Optimization

The L2-penalized log-likelihood is still strictly concave and can be maximized
with Newton-Raphson or IRLS, adding $\lambda\mathbf{I}$ to the Hessian.  The
L1-penalized objective is non-differentiable at $\theta_j = 0$, so specialized
algorithms are needed:

- **Coordinate descent:** Updates one coefficient at a time using soft
  thresholding.  This is the approach used by `glmnet` and scikit-learn's
  `saga` solver.
- **Proximal gradient methods:** Combine a gradient step on the smooth part
  with a proximal operator for the L1 term.

## Regularization Path

Fitting the model across a grid of $\lambda$ values (from large to small)
produces a **regularization path**.  Each coefficient traces a curve as
$\lambda$ decreases: L2 paths are smooth; L1 paths are piecewise linear,
with coefficients entering the model one at a time.  The optimal $\lambda$ is
selected by cross-validation.

??? example "Choosing C by Cross-Validation"
    Using 5-fold CV on a dataset with $n = 500$ and $p = 20$:

    | $C$ | Mean CV accuracy | Non-zero coefficients |
    |---|---|---|
    | 0.01 | 0.72 | 3 |
    | 0.1 | 0.81 | 8 |
    | 1.0 | 0.84 | 15 |
    | 10.0 | 0.83 | 19 |
    | 100.0 | 0.82 | 20 |

    The best CV accuracy occurs at $C = 1.0$.  Increasing $C$ beyond this
    point adds predictors without improving (and slightly degrading)
    performance.
