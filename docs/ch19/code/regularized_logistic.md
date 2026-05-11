# Regularized Logistic Regression


## Overview

Regularized logistic regression adds a penalty term to the log-likelihood to
prevent overfitting and improve generalization, especially when the number of
features is large relative to the sample size.  This page covers L2 (ridge),
L1 (lasso), and elastic net penalties, their effects on coefficient estimates,
and practical implementation in scikit-learn.

## Unregularized Logistic Regression

The standard logistic regression model maximizes the log-likelihood:

$$
\ell(\boldsymbol\beta) = \sum_{i=1}^{n}
  \bigl[y_i \log \hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)\bigr]
$$

where $\hat{p}_i = \sigma(\mathbf{x}_i^T \boldsymbol\beta)$ and
$\sigma(z) = 1/(1+e^{-z})$.

## L2 Regularization (Ridge)

Ridge logistic regression adds a squared-norm penalty:

$$
\hat{\boldsymbol\beta}_{\text{ridge}}
  = \arg\max_{\boldsymbol\beta}\;
    \ell(\boldsymbol\beta) - \frac{\lambda}{2}\|\boldsymbol\beta\|_2^2
$$

Equivalently, in scikit-learn's parameterization with $C = 1/\lambda$:

$$
\hat{\boldsymbol\beta}_{\text{ridge}}
  = \arg\min_{\boldsymbol\beta}\;
    -\ell(\boldsymbol\beta) + \frac{1}{2C}\|\boldsymbol\beta\|_2^2
$$

The L2 penalty shrinks all coefficients toward zero but does not set any
exactly to zero.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

np.random.seed(42)
n, p = 200, 20
X = np.random.randn(n, p)
true_beta = np.zeros(p)
true_beta[:5] = [1.5, -1.0, 0.8, -0.5, 0.3]
logit = X @ true_beta
prob = 1 / (1 + np.exp(-logit))
y = np.random.binomial(1, prob)

ridge_model = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs',
                                  max_iter=1000)
ridge_model.fit(X, y)
print("Ridge coefficients:", np.round(ridge_model.coef_[0], 3))
```

## L1 Regularization (Lasso)

Lasso logistic regression replaces the squared penalty with the absolute-value
norm:

$$
\hat{\boldsymbol\beta}_{\text{lasso}}
  = \arg\min_{\boldsymbol\beta}\;
    -\ell(\boldsymbol\beta) + \frac{1}{C}\|\boldsymbol\beta\|_1
$$

The L1 penalty induces **sparsity**: sufficiently small coefficients are driven
exactly to zero, performing automatic feature selection.

```python
lasso_model = LogisticRegression(penalty='l1', C=1.0, solver='saga',
                                  max_iter=5000)
lasso_model.fit(X, y)
print("Lasso coefficients:", np.round(lasso_model.coef_[0], 3))
print(f"Non-zero coefficients: {np.sum(lasso_model.coef_[0] != 0)} / {p}")
```

## Elastic Net

Elastic net combines L1 and L2 penalties with a mixing parameter
$\alpha \in [0,1]$ (called `l1_ratio` in scikit-learn):

$$
\text{Penalty} = \frac{1-\alpha}{2}\|\boldsymbol\beta\|_2^2

  + \alpha\,\|\boldsymbol\beta\|_1
$$

When $\alpha = 0$ this reduces to ridge; when $\alpha = 1$ it reduces to
lasso.  The elastic net is useful when there are groups of correlated features:
L1 alone would select one from each group, while the L2 component encourages
sharing the weight among correlated predictors.

```python
enet_model = LogisticRegression(penalty='elasticnet', C=1.0,
                                 solver='saga', l1_ratio=0.5,
                                 max_iter=5000)
enet_model.fit(X, y)
print("Elastic Net coefficients:", np.round(enet_model.coef_[0], 3))
```

## Effect of Regularization Strength

As $C$ increases (weaker regularization), the estimates approach the
unregularized MLE.  As $C$ decreases (stronger regularization), the
coefficients shrink toward zero.

```python
import matplotlib.pyplot as plt

C_values = np.logspace(-3, 3, 50)
coefs = []

for C in C_values:
    model = LogisticRegression(penalty='l2', C=C, solver='lbfgs',
                                max_iter=2000)
    model.fit(X, y)
    coefs.append(model.coef_[0])

coefs = np.array(coefs)

plt.figure(figsize=(10, 5))
for j in range(p):
    plt.plot(np.log10(C_values), coefs[:, j],
             linewidth=2 if j < 5 else 0.8,
             alpha=1.0 if j < 5 else 0.3)
plt.xlabel('log10(C)')
plt.ylabel('Coefficient value')
plt.title('Ridge Logistic Regression: Coefficient Paths')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
```

## Cross-Validation for Tuning C

Scikit-learn provides `LogisticRegressionCV` which performs cross-validation
over a grid of $C$ values:

```python
from sklearn.linear_model import LogisticRegressionCV

model_cv = LogisticRegressionCV(
    Cs=20, penalty='l2', cv=5, scoring='accuracy',
    solver='lbfgs', max_iter=2000
)
model_cv.fit(X, y)
print(f"Best C: {model_cv.C_[0]:.4f}")
print(f"Best CV accuracy: {model_cv.scores_[1].mean(axis=0).max():.4f}")
```

## Interpretation

- **Ridge** (L2) is preferred when all features are expected to contribute and
  multicollinearity is present; it stabilizes the coefficient estimates.
- **Lasso** (L1) is preferred when a sparse model is desired; it performs
  variable selection by zeroing out irrelevant features.
- **Elastic net** provides a compromise, useful when features are correlated
  and sparsity is still desired.
- The regularization strength $C$ (or $\lambda = 1/C$) controls the bias-variance
  trade-off: smaller $C$ increases bias but reduces variance.

## Exercises

**Exercise 1.**
Generate a dataset with $n = 200$ observations and $p = 50$ features, where
only the first 5 features have non-zero true coefficients.  Fit L1-regularized
logistic regression for $C \in \{0.01, 0.1, 1.0, 10.0\}$.  For each value of
$C$, report the number of non-zero estimated coefficients.

??? success "Solution to Exercise 1"

    ```python
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    np.random.seed(42)
    n, p = 200, 50
    X = np.random.randn(n, p)
    true_beta = np.zeros(p)
    true_beta[:5] = [2.0, -1.5, 1.0, -0.8, 0.5]
    logit = X @ true_beta
    prob = 1 / (1 + np.exp(-logit))
    y = np.random.binomial(1, prob)

    for C in [0.01, 0.1, 1.0, 10.0]:
        model = LogisticRegression(penalty='l1', C=C, solver='saga',
                                    max_iter=5000)
        model.fit(X, y)
        nnz = np.sum(model.coef_[0] != 0)
        print(f"C = {C:5.2f}: {nnz} non-zero coefficients out of {p}")
    ```

    As $C$ increases (weaker penalty), more coefficients become non-zero.
    At small $C$ only the strongest signals survive; at large $C$ the model
    approaches the unregularized fit. $\square$

---

**Exercise 2.**
Show that the ridge penalty $\|\boldsymbol\beta\|_2^2$ is equivalent to
placing an independent $N(0, \sigma^2)$ prior on each $\beta_j$ in a Bayesian
logistic regression, with $\sigma^2 = C$.

??? success "Solution to Exercise 2"

    In Bayesian logistic regression the posterior is proportional to the
    likelihood times the prior:

    $$
    p(\boldsymbol\beta \mid \mathbf{y})
      \propto \prod_{i=1}^n p(y_i \mid \mathbf{x}_i, \boldsymbol\beta)
      \;\cdot\; \prod_{j=1}^p \frac{1}{\sqrt{2\pi\sigma^2}}
      \exp\Bigl(-\frac{\beta_j^2}{2\sigma^2}\Bigr)
    $$

    Taking the log and ignoring constants:

    $$
    \log p(\boldsymbol\beta \mid \mathbf{y})
      = \ell(\boldsymbol\beta) - \frac{1}{2\sigma^2}\sum_{j=1}^p \beta_j^2 + \text{const}
    $$

    This is exactly the ridge objective with $\lambda = 1/\sigma^2$, or
    equivalently $C = \sigma^2$.  Maximizing the log-posterior (MAP estimation)
    is therefore identical to ridge logistic regression. $\square$

---

**Exercise 3.**
Explain geometrically why the L1 penalty produces sparse solutions while the
L2 penalty does not.

??? success "Solution to Exercise 3"

    The constraint region for L1 is a diamond (in 2D, the set
    $\{(\beta_1, \beta_2) : |\beta_1| + |\beta_2| \leq t\}$), which has
    corners on the coordinate axes.  The constraint region for L2 is a circle
    ($\beta_1^2 + \beta_2^2 \leq t^2$), which is smooth everywhere.

    The contours of the log-likelihood are typically elliptical.  The
    constrained optimum lies where the likelihood contour first touches the
    constraint region.  For the diamond, this intersection is much more likely
    to occur at a corner where one or more coordinates are exactly zero.  For
    the circle, tangency at a point on a coordinate axis requires a special
    alignment that occurs with probability zero for generic data.

    This geometric argument generalizes to higher dimensions, where the L1
    ball has $2^p$ vertices and the contact point is typically at a vertex or
    face where many coordinates vanish. $\square$

---

**Exercise 4.**
Using `LogisticRegressionCV` with `penalty='l1'`, `solver='saga'`, and
5-fold cross-validation, find the optimal $C$ for the dataset from Exercise 1.
Report the selected $C$ and the corresponding CV accuracy.

??? success "Solution to Exercise 4"

    ```python
    from sklearn.linear_model import LogisticRegressionCV

    model_cv = LogisticRegressionCV(
        Cs=20, penalty='l1', cv=5, scoring='accuracy',
        solver='saga', max_iter=5000
    )
    model_cv.fit(X, y)
    print(f"Best C: {model_cv.C_[0]:.4f}")

    best_idx = np.argmax(model_cv.scores_[1].mean(axis=0))
    best_acc = model_cv.scores_[1].mean(axis=0)[best_idx]
    print(f"Best CV accuracy: {best_acc:.4f}")
    print(f"Non-zero coefficients: "
          f"{np.sum(model_cv.coef_[0] != 0)} / {p}")
    ```

    The cross-validated $C$ balances model complexity with predictive
    performance.  The selected model typically identifies the 5 true non-zero
    features while keeping most noise features at zero. $\square$

---

**Exercise 5.**
Prove that the elastic net penalty

$$
\alpha\|\boldsymbol\beta\|_1 + \frac{1-\alpha}{2}\|\boldsymbol\beta\|_2^2
$$

is a convex function of $\boldsymbol\beta$ for any $\alpha \in [0,1]$.

??? success "Solution to Exercise 5"

    Both $\|\boldsymbol\beta\|_1 = \sum_j |\beta_j|$ and
    $\|\boldsymbol\beta\|_2^2 = \sum_j \beta_j^2$ are convex functions of
    $\boldsymbol\beta$.  The L1 norm is convex because it is a sum of convex
    functions $|\beta_j|$.  The squared L2 norm is strictly convex because
    its Hessian is $2I$, which is positive definite.

    A non-negative weighted sum of convex functions is convex.  Since
    $\alpha \geq 0$ and $(1-\alpha)/2 \geq 0$, the elastic net penalty

    $$
    P(\boldsymbol\beta) = \alpha\|\boldsymbol\beta\|_1 + \frac{1-\alpha}{2}\|\boldsymbol\beta\|_2^2
    $$

    is convex.  Moreover, for $\alpha < 1$ the L2 term makes $P$ strictly
    convex, guaranteeing a unique minimizer of the penalized loss. $\square$
