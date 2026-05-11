# Linear and Quadratic Discriminant Analysis Classification


## Overview

Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA)
are **generative classifiers** that model the class-conditional densities as
multivariate Gaussians and apply Bayes' theorem to obtain posterior
probabilities.  This page compares LDA, QDA, Gaussian Naive Bayes, and
logistic regression on synthetic 2D data under two scenarios: shared covariance
(where LDA excels) and different covariances (where QDA excels).

## Generative Classification Framework

For $K$ classes, each generative classifier models the class-conditional density
$f_k(\mathbf{x}) = p(\mathbf{x} \mid Y = k)$ and applies Bayes' theorem:

$$
P(Y = k \mid \mathbf{x}) = \frac{\pi_k\,f_k(\mathbf{x})}{\sum_{l=1}^K \pi_l\,f_l(\mathbf{x})}
$$

where $\pi_k = P(Y = k)$ is the prior probability of class $k$.

## LDA: Shared Covariance

LDA assumes each class-conditional density is multivariate Gaussian with a
**shared** covariance matrix $\boldsymbol\Sigma$:

$$
f_k(\mathbf{x}) = \frac{1}{(2\pi)^{p/2}|\boldsymbol\Sigma|^{1/2}}
  \exp\!\Bigl(-\frac{1}{2}(\mathbf{x}-\boldsymbol\mu_k)^T\boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu_k)\Bigr)
$$

Because $\boldsymbol\Sigma$ is the same for all classes, the log-posterior
ratio between classes $k$ and $l$ is linear in $\mathbf{x}$, producing
**linear** decision boundaries.

### Discriminant Function

The linear discriminant function for class $k$ is

$$
\delta_k(\mathbf{x}) = \mathbf{x}^T\boldsymbol\Sigma^{-1}\boldsymbol\mu_k

  - \frac{1}{2}\boldsymbol\mu_k^T\boldsymbol\Sigma^{-1}\boldsymbol\mu_k
  + \log\pi_k
$$

We assign the observation to the class with the largest $\delta_k(\mathbf{x})$.

## QDA: Class-Specific Covariance

QDA relaxes the shared-covariance assumption.  Each class has its own
$\boldsymbol\Sigma_k$:

$$
f_k(\mathbf{x}) = \frac{1}{(2\pi)^{p/2}|\boldsymbol\Sigma_k|^{1/2}}
  \exp\!\Bigl(-\frac{1}{2}(\mathbf{x}-\boldsymbol\mu_k)^T\boldsymbol\Sigma_k^{-1}(\mathbf{x}-\boldsymbol\mu_k)\Bigr)
$$

The log-posterior ratio is now **quadratic** in $\mathbf{x}$, yielding curved
decision boundaries.

### Discriminant Function

$$
\delta_k(\mathbf{x}) = -\frac{1}{2}\log|\boldsymbol\Sigma_k|

  - \frac{1}{2}(\mathbf{x}-\boldsymbol\mu_k)^T\boldsymbol\Sigma_k^{-1}(\mathbf{x}-\boldsymbol\mu_k)
  + \log\pi_k
$$

## Gaussian Naive Bayes

Naive Bayes further simplifies by assuming features are **conditionally
independent** within each class, making each $\boldsymbol\Sigma_k$ diagonal:

$$
f_k(\mathbf{x}) = \prod_{j=1}^p f_{kj}(x_j)
$$

where each $f_{kj}$ is a univariate Gaussian.  This dramatically reduces the
number of parameters from $O(p^2)$ per class to $O(p)$.

## Data Generation

Two scenarios illustrate when each method is preferred:

**Scenario A (Shared Covariance):** Both classes share $\boldsymbol\Sigma = \begin{pmatrix} 1 & 0.5 \\ 0.5 & 1 \end{pmatrix}$ with means $\boldsymbol\mu_0 = (0,0)^T$ and $\boldsymbol\mu_1 = (2, 1.5)^T$.

**Scenario B (Different Covariances):** Class 0 has $\boldsymbol\Sigma_0 = \begin{pmatrix} 1 & 0 \\ 0 & 0.3 \end{pmatrix}$ and class 1 has $\boldsymbol\Sigma_1 = \begin{pmatrix} 0.3 & 0 \\ 0 & 2 \end{pmatrix}$.

```python
import numpy as np

np.random.seed(42)

def generate_shared_cov(n_per_class=200):
    """Two classes with the SAME covariance."""
    cov = [[1.0, 0.5], [0.5, 1.0]]
    X0 = np.random.multivariate_normal([0, 0], cov, n_per_class)
    X1 = np.random.multivariate_normal([2, 1.5], cov, n_per_class)
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y

def generate_diff_cov(n_per_class=200):
    """Two classes with DIFFERENT covariances."""
    cov0 = [[1.0, 0.0], [0.0, 0.3]]
    cov1 = [[0.3, 0.0], [0.0, 2.0]]
    X0 = np.random.multivariate_normal([0, 0], cov0, n_per_class)
    X1 = np.random.multivariate_normal([1.5, 1.5], cov1, n_per_class)
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y
```

## Fitting and Comparing Classifiers

```python
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

classifiers = {
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Naive Bayes": GaussianNB(),
    "Logistic Reg": LogisticRegression(),
}

for scenario_name, (X, y) in [
    ("Shared Cov", generate_shared_cov()),
    ("Diff Cov", generate_diff_cov()),
]:
    print(f"\n--- {scenario_name} ---")
    for name, clf in classifiers.items():
        clf.fit(X, y)
        cv_acc = cross_val_score(clf, X, y, cv=10,
                                  scoring="accuracy").mean()
        print(f"  {name:15s}: 10-fold CV accuracy = {cv_acc:.4f}")
```

## Decision Boundary Visualization

```python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(ax, clf, X, y, title):
    """Plot 2D decision boundary with scatter overlay."""
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(["#FFAAAA", "#AAAAFF"])
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="red", s=10,
               edgecolors="none", alpha=0.6, label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", s=10,
               edgecolors="none", alpha=0.6, label="Class 1")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
scenarios = {
    "Shared Covariance": generate_shared_cov(),
    "Different Covariances": generate_diff_cov(),
}
for row, (scenario_name, (X, y)) in enumerate(scenarios.items()):
    for col, (name, clf) in enumerate(classifiers.items()):
        clf.fit(X, y)
        cv_acc = cross_val_score(clf, X, y, cv=10,
                                  scoring="accuracy").mean()
        plot_decision_boundary(
            axes[row, col], clf, X, y,
            f"{name}\nCV acc = {cv_acc:.3f}")
        if col == 0:
            axes[row, col].set_ylabel(scenario_name, fontsize=11)

plt.suptitle("Generative Classifiers: Decision Boundaries", fontsize=13)
plt.tight_layout()
plt.show()
```

## Interpretation

- **Shared covariance scenario:** LDA and logistic regression perform
  comparably because the true boundary is linear.  QDA and Naive Bayes also
  perform well but use extra parameters unnecessarily.
- **Different covariance scenario:** QDA outperforms LDA because the true
  boundary is quadratic.  LDA's linear boundary cannot capture the curvature.
- **Naive Bayes** performs well when features are approximately independent
  within each class, but degrades when off-diagonal covariance entries are
  large.
- **Parameter count trade-off:** LDA estimates $O(p^2)$ parameters (one shared
  covariance), QDA estimates $O(Kp^2)$ (one per class), and Naive Bayes
  estimates $O(Kp)$ (diagonal covariances only).

## Exercises

**Exercise 1.**
For two-class LDA with equal priors ($\pi_0 = \pi_1 = 0.5$), show that the
decision boundary is the set of points equidistant (in the Mahalanobis sense)
from the two class means.

??? success "Solution to Exercise 1"

    With equal priors the $\log\pi_k$ terms cancel.  The decision boundary is
    where $\delta_0(\mathbf{x}) = \delta_1(\mathbf{x})$:

    $$
    \mathbf{x}^T\boldsymbol\Sigma^{-1}\boldsymbol\mu_0

      - \tfrac{1}{2}\boldsymbol\mu_0^T\boldsymbol\Sigma^{-1}\boldsymbol\mu_0
    = \mathbf{x}^T\boldsymbol\Sigma^{-1}\boldsymbol\mu_1

      - \tfrac{1}{2}\boldsymbol\mu_1^T\boldsymbol\Sigma^{-1}\boldsymbol\mu_1
    $$

    Rearranging:

    $$
    \mathbf{x}^T\boldsymbol\Sigma^{-1}(\boldsymbol\mu_0 - \boldsymbol\mu_1)
    = \tfrac{1}{2}(\boldsymbol\mu_0 + \boldsymbol\mu_1)^T\boldsymbol\Sigma^{-1}(\boldsymbol\mu_0 - \boldsymbol\mu_1)
    $$

    This defines a hyperplane passing through the midpoint
    $\frac{1}{2}(\boldsymbol\mu_0 + \boldsymbol\mu_1)$ with normal vector
    $\boldsymbol\Sigma^{-1}(\boldsymbol\mu_0 - \boldsymbol\mu_1)$.  Points
    on this hyperplane are equidistant (in Mahalanobis distance) from both
    means. $\square$

---

**Exercise 2.**
Explain why QDA has a higher risk of overfitting than LDA, and describe a
situation where you would prefer LDA despite QDA's greater flexibility.

??? success "Solution to Exercise 2"

    QDA estimates a separate $p \times p$ covariance matrix for each class,
    requiring $K \cdot p(p+1)/2$ parameters for the covariance structure alone,
    compared to $p(p+1)/2$ for LDA.  When the sample size per class is small
    relative to $p$, QDA's many parameters lead to high variance and
    overfitting.

    We would prefer LDA when:

    - The training sample is small relative to $p^2$.
    - Exploratory analysis suggests the class covariances are similar.
    - Cross-validation shows QDA does not improve over LDA.

    As a rule of thumb, if the ratio $n_k / p^2$ is less than about 5 for
    any class, QDA's covariance estimates may be unreliable. $\square$

---

**Exercise 3.**
Derive the QDA discriminant function $\delta_k(\mathbf{x})$ starting from the
multivariate Gaussian density and Bayes' theorem.

??? success "Solution to Exercise 3"

    The posterior is

    $$
    P(Y=k \mid \mathbf{x}) \propto \pi_k\,f_k(\mathbf{x})
    $$

    Taking the log:

    $$
    \log P(Y=k \mid \mathbf{x}) = \log\pi_k + \log f_k(\mathbf{x}) + \text{const}
    $$

    Substituting the Gaussian density:

    $$
    \log f_k(\mathbf{x}) = -\frac{p}{2}\log(2\pi) - \frac{1}{2}\log|\boldsymbol\Sigma_k|

      - \frac{1}{2}(\mathbf{x}-\boldsymbol\mu_k)^T\boldsymbol\Sigma_k^{-1}(\mathbf{x}-\boldsymbol\mu_k)
    $$

    Dropping terms that do not depend on $k$ (the $-\frac{p}{2}\log(2\pi)$
    term and the normalizing constant from Bayes' theorem):

    $$
    \delta_k(\mathbf{x}) = -\frac{1}{2}\log|\boldsymbol\Sigma_k|

      - \frac{1}{2}(\mathbf{x}-\boldsymbol\mu_k)^T\boldsymbol\Sigma_k^{-1}(\mathbf{x}-\boldsymbol\mu_k)
      + \log\pi_k
    $$

    The quadratic term in $\mathbf{x}$ (from the class-specific
    $\boldsymbol\Sigma_k^{-1}$) is what gives QDA its name and its curved
    decision boundaries. $\square$

---

**Exercise 4.**
Using the shared-covariance data generated above, fit LDA manually by computing
the pooled covariance matrix, class means, and discriminant function.  Compare
your predictions to scikit-learn's `LinearDiscriminantAnalysis`.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    np.random.seed(42)
    cov = [[1.0, 0.5], [0.5, 1.0]]
    X0 = np.random.multivariate_normal([0, 0], cov, 200)
    X1 = np.random.multivariate_normal([2, 1.5], cov, 200)
    X = np.vstack([X0, X1])
    y = np.array([0]*200 + [1]*200)

    mu0, mu1 = X[y == 0].mean(axis=0), X[y == 1].mean(axis=0)
    S0 = np.cov(X[y == 0].T)
    S1 = np.cov(X[y == 1].T)
    S_pooled = 0.5 * (S0 + S1)
    S_inv = np.linalg.inv(S_pooled)

    def predict_lda(x):
        d0 = x @ S_inv @ mu0 - 0.5 * mu0 @ S_inv @ mu0
        d1 = x @ S_inv @ mu1 - 0.5 * mu1 @ S_inv @ mu1
        return (d1 > d0).astype(int)

    y_manual = predict_lda(X)
    lda = LinearDiscriminantAnalysis().fit(X, y)
    y_sklearn = lda.predict(X)
    agreement = np.mean(y_manual == y_sklearn)
    print(f"Agreement with sklearn: {agreement:.4f}")
    ```

    The manual and scikit-learn predictions should agree on nearly all
    observations (any small differences arise from implementation details
    such as shrinkage). $\square$

---

**Exercise 5.**
Prove that when the class covariances are equal ($\boldsymbol\Sigma_k = \boldsymbol\Sigma$ for all $k$), QDA reduces to LDA.

??? success "Solution to Exercise 5"

    The QDA discriminant function is

    $$
    \delta_k(\mathbf{x}) = -\frac{1}{2}\log|\boldsymbol\Sigma_k|

      - \frac{1}{2}(\mathbf{x}-\boldsymbol\mu_k)^T\boldsymbol\Sigma_k^{-1}(\mathbf{x}-\boldsymbol\mu_k)
      + \log\pi_k
    $$

    When $\boldsymbol\Sigma_k = \boldsymbol\Sigma$ for all $k$, the
    $-\frac{1}{2}\log|\boldsymbol\Sigma_k|$ term becomes a constant
    independent of $k$ and can be dropped from the comparison.  Expanding the
    quadratic form:

    $$
    (\mathbf{x}-\boldsymbol\mu_k)^T\boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu_k)
    = \mathbf{x}^T\boldsymbol\Sigma^{-1}\mathbf{x}

      - 2\mathbf{x}^T\boldsymbol\Sigma^{-1}\boldsymbol\mu_k
      + \boldsymbol\mu_k^T\boldsymbol\Sigma^{-1}\boldsymbol\mu_k
    $$

    The term $\mathbf{x}^T\boldsymbol\Sigma^{-1}\mathbf{x}$ does not depend
    on $k$ and can also be dropped.  What remains is

    $$
    \delta_k(\mathbf{x}) = \mathbf{x}^T\boldsymbol\Sigma^{-1}\boldsymbol\mu_k

      - \frac{1}{2}\boldsymbol\mu_k^T\boldsymbol\Sigma^{-1}\boldsymbol\mu_k
      + \log\pi_k
    $$

    which is exactly the LDA discriminant function.  Therefore QDA reduces
    to LDA when the class covariances are identical. $\square$
