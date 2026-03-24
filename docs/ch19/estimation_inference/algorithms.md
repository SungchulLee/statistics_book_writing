# Newton-Raphson and Iteratively Reweighted Least Squares

## Why Iterative Algorithms Are Needed

In linear regression the ordinary least squares estimator has a closed-form
solution $\hat{\boldsymbol{\theta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$.
Logistic regression has no such luxury.  The log-likelihood is nonlinear in the
parameters, so the maximum likelihood estimates must be found by iterative
numerical optimization.  This section develops the two most widely used
algorithms: **Newton-Raphson** and **Iteratively Reweighted Least Squares
(IRLS)**.

## The Log-Likelihood for Logistic Regression

Recall from the [likelihood section](../logistic_regression/likelihood.md) that
for $n$ independent observations $(y_i, \mathbf{x}_i)$ with
$y_i \in \{0,1\}$ the log-likelihood is

$$
\ell(\boldsymbol{\theta})
= \sum_{i=1}^{n}\bigl[y_i\,\mathbf{x}_i^T\boldsymbol{\theta}
  - \log\bigl(1 + e^{\mathbf{x}_i^T\boldsymbol{\theta}}\bigr)\bigr]
$$

where $\boldsymbol{\theta} \in \mathbb{R}^{p}$ collects the intercept and
slope coefficients.  The predicted probability for observation $i$ is
$\hat{p}_i = \sigma(\mathbf{x}_i^T\boldsymbol{\theta})$ with $\sigma$ the
sigmoid function.

## The Score Equation

The **score** (gradient of the log-likelihood) is

$$
\mathbf{s}(\boldsymbol{\theta})
= \frac{\partial \ell}{\partial \boldsymbol{\theta}}
= \sum_{i=1}^{n}(y_i - \hat{p}_i)\,\mathbf{x}_i
= \mathbf{X}^T(\mathbf{y} - \hat{\mathbf{p}})
$$

where $\mathbf{X}$ is the $n \times p$ design matrix and
$\hat{\mathbf{p}} = (\hat{p}_1, \ldots, \hat{p}_n)^T$.  Setting the score
equal to zero yields a system of $p$ nonlinear equations — there is no
closed-form solution because $\hat{\mathbf{p}}$ depends on
$\boldsymbol{\theta}$ through the sigmoid.

## The Hessian Matrix

The second derivative of the log-likelihood is the **Hessian**:

$$
\mathbf{H}(\boldsymbol{\theta})
= \frac{\partial^2 \ell}{\partial \boldsymbol{\theta}\,\partial \boldsymbol{\theta}^T}
= -\sum_{i=1}^{n}\hat{p}_i(1-\hat{p}_i)\,\mathbf{x}_i\mathbf{x}_i^T
= -\mathbf{X}^T\mathbf{W}\mathbf{X}
$$

where $\mathbf{W} = \operatorname{diag}\bigl(\hat{p}_1(1-\hat{p}_1),\ldots,
\hat{p}_n(1-\hat{p}_n)\bigr)$ is an $n \times n$ diagonal weight matrix.
Because every diagonal entry of $\mathbf{W}$ is strictly positive (as long as
$0 < \hat{p}_i < 1$), the Hessian is negative definite.  This guarantees that
the log-likelihood is **strictly concave**, so any local maximum is the unique
global maximum.

## Newton-Raphson Algorithm

Newton-Raphson finds a root of the score equation by iterating

$$
\boldsymbol{\theta}^{(t+1)}
= \boldsymbol{\theta}^{(t)}
  - \bigl[\mathbf{H}(\boldsymbol{\theta}^{(t)})\bigr]^{-1}
    \mathbf{s}(\boldsymbol{\theta}^{(t)})
$$

Substituting the score and Hessian derived above:

$$
\boldsymbol{\theta}^{(t+1)}
= \boldsymbol{\theta}^{(t)}
  + \bigl(\mathbf{X}^T\mathbf{W}^{(t)}\mathbf{X}\bigr)^{-1}
    \mathbf{X}^T\bigl(\mathbf{y} - \hat{\mathbf{p}}^{(t)}\bigr)
$$

where $\mathbf{W}^{(t)}$ and $\hat{\mathbf{p}}^{(t)}$ are evaluated at the
current iterate $\boldsymbol{\theta}^{(t)}$.

### Convergence

Because the log-likelihood is strictly concave, Newton-Raphson converges to the
unique MLE from any starting point.  Near the optimum the convergence is
**quadratic**: the number of correct digits roughly doubles at each iteration.
In practice three to eight iterations usually suffice.

## Fisher Scoring and IRLS

**Fisher scoring** replaces the observed Hessian with its expected value.  For
logistic regression these two quantities are identical because

$$
-\mathbf{H}(\boldsymbol{\theta})
= \mathbf{X}^T\mathbf{W}\mathbf{X}
= \mathcal{I}(\boldsymbol{\theta})
$$

where $\mathcal{I}(\boldsymbol{\theta})$ is the **Fisher information matrix**.
Therefore Fisher scoring and Newton-Raphson produce the same iterates for
logistic regression.

### IRLS Interpretation

Rewrite the Newton-Raphson update as

$$
\boldsymbol{\theta}^{(t+1)}
= \bigl(\mathbf{X}^T\mathbf{W}^{(t)}\mathbf{X}\bigr)^{-1}
  \mathbf{X}^T\mathbf{W}^{(t)}\mathbf{z}^{(t)}
$$

where the **working response** is

$$
\mathbf{z}^{(t)}
= \mathbf{X}\boldsymbol{\theta}^{(t)}
  + \bigl(\mathbf{W}^{(t)}\bigr)^{-1}
    \bigl(\mathbf{y} - \hat{\mathbf{p}}^{(t)}\bigr)
$$

Each iteration solves a **weighted least squares** problem with weights
$\mathbf{W}^{(t)}$ and response $\mathbf{z}^{(t)}$.  Because the weights
change at every step, the procedure is called **Iteratively Reweighted Least
Squares**.

## Algorithm Summary

| Step | Action |
|---|---|
| 0 | Initialize $\boldsymbol{\theta}^{(0)}$ (e.g., all zeros) |
| 1 | Compute $\hat{\mathbf{p}}^{(t)} = \sigma(\mathbf{X}\boldsymbol{\theta}^{(t)})$ |
| 2 | Form $\mathbf{W}^{(t)} = \operatorname{diag}(\hat{p}_i(1-\hat{p}_i))$ |
| 3 | Compute working response $\mathbf{z}^{(t)}$ |
| 4 | Solve $\boldsymbol{\theta}^{(t+1)} = (\mathbf{X}^T\mathbf{W}^{(t)}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{W}^{(t)}\mathbf{z}^{(t)}$ |
| 5 | Repeat steps 1-4 until $\lVert\boldsymbol{\theta}^{(t+1)} - \boldsymbol{\theta}^{(t)}\rVert < \varepsilon$ |

??? example "Worked Example: Two Iterations by Hand"
    Suppose $n=4$ observations with a single predictor (plus intercept):

    | $i$ | $x_i$ | $y_i$ |
    |---|---|---|
    | 1 | 1.0 | 1 |
    | 2 | 2.0 | 1 |
    | 3 | 3.0 | 0 |
    | 4 | 4.0 | 0 |

    **Iteration 0.** Start with $\boldsymbol{\theta}^{(0)} = (0, 0)^T$.  Then
    $\hat{p}_i = 0.5$ for all $i$, and $w_i = 0.25$ for all $i$.

    The score is $\mathbf{X}^T(\mathbf{y} - \hat{\mathbf{p}})$ where
    $\mathbf{y} - \hat{\mathbf{p}} = (0.5, 0.5, -0.5, -0.5)^T$.

    **Iteration 1.** Solve the weighted least squares problem to obtain
    $\boldsymbol{\theta}^{(1)}$.  The predicted probabilities shift: observations
    with $x_i = 1,2$ get higher $\hat{p}_i$ and those with $x_i = 3,4$ get
    lower $\hat{p}_i$.

    After a few more iterations the algorithm converges to the MLE.

## Separation and Non-Convergence

When the two classes are **perfectly separated** — a hyperplane in feature space
classifies every training point correctly — the MLE does not exist.  The
log-likelihood approaches its supremum as $\lVert\boldsymbol{\theta}\rVert
\to \infty$, and Newton-Raphson diverges.  Symptoms include coefficients growing
without bound and very large standard errors.

!!! warning "Detecting Separation"
    Modern software (e.g., R's `glm` or Python's `statsmodels`) issues warnings
    when separation is detected.  Remedies include adding a small ridge penalty
    (Firth's penalized likelihood) or using exact conditional logistic
    regression.

## Connection to Gradient Descent

Gradient descent updates $\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)}
+ \eta\,\mathbf{s}(\boldsymbol{\theta}^{(t)})$ using a fixed step size $\eta$.
Newton-Raphson can be viewed as gradient descent with an adaptive step size
given by the inverse Hessian.  This curvature information is what gives
Newton-Raphson its fast convergence, at the cost of computing and inverting the
$p \times p$ Hessian at each step.  For high-dimensional problems ($p$ large),
quasi-Newton methods such as L-BFGS approximate the Hessian to reduce
computational cost.
