# Principal Components and Partial Least Squares Examples

## Overview

Principal Components Regression (PCR) and Partial Least Squares (PLS) are dimension-reduction
approaches to regression. PCR first reduces the predictor space using PCA (an unsupervised
method) and then regresses the response on the leading principal components. PLS finds
directions in the predictor space that have high covariance with the response (a supervised
method). This page demonstrates both on housing data and compares them with OLS and Ridge.

## Principal Components Regression (PCR)

### Idea

PCR proceeds in two stages:

1. **Dimension reduction.** Compute the principal components $Z_1, \dots, Z_M$ of $X$, where
   $Z_m = X v_m$ and $v_m$ is the $m$-th eigenvector of $X^\top X$.
2. **Regression.** Regress $y$ on $Z_1, \dots, Z_M$ using OLS.

The number of components $M \le p$ is a tuning parameter selected by cross-validation.

### Mathematical Formulation

Let $X = U D V^\top$ be the SVD. The principal components are $Z = XV = UD$. PCR with $M$
components fits

$$
\hat{y}^{\text{PCR}} = Z_M (Z_M^\top Z_M)^{-1} Z_M^\top y = \sum_{m=1}^{M} z_m \frac{z_m^\top y}{\|z_m\|^2},
$$

where $Z_M = [z_1 \mid \cdots \mid z_M]$ contains the first $M$ principal components.

### Connection to Ridge

PCR and Ridge both shrink along the principal component directions, but differently:

- **Ridge** shrinks the $m$-th component by the factor $d_m^2/(d_m^2 + \lambda)$, which is
  continuous.
- **PCR** either keeps a component (factor 1) or drops it (factor 0), which is discrete.

Ridge is therefore a "smooth" version of PCR.

## Partial Least Squares (PLS)

### Idea

PLS finds directions in $X$-space that are most correlated with $y$, unlike PCA which finds
directions of maximum variance in $X$ alone. PLS components $T_1, \dots, T_M$ are constructed
iteratively:

1. Compute the weight vector $w_m = X^\top y / \|X^\top y\|$ (direction of maximum covariance
   with $y$).
2. Form the component $t_m = X w_m$.
3. Deflate: regress $X$ and $y$ on $t_m$ and replace with residuals.
4. Repeat.

### When PLS Outperforms PCR

PLS tends to outperform PCR when:

- The directions of highest variance in $X$ are not aligned with the response.
- A small number of supervised components suffice to capture the $X$--$y$ relationship.

## Code: Data Loading and OLS Baseline

```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA = Path(__file__).parent.parent.parent / 'data'
house = pd.read_csv(DATA / 'house_sales.csv', sep='\t')

numeric_features = [
    'SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
    'BldgGrade', 'NbrLivingUnits', 'SqFtFinBasement', 'YrBuilt', 'YrRenovated'
]
X = house[numeric_features].values
y = house['AdjSalePrice'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

ols_model = LinearRegression().fit(X_scaled, y)
ols_r2 = r2_score(y, ols_model.predict(X_scaled))
ols_rmse = np.sqrt(mean_squared_error(y, ols_model.predict(X_scaled)))
```

## Code: PCR with Cross-Validation

```python
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cumsum_var = np.cumsum(explained_var)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
pcr_mse_scores = []

for M in range(1, X_scaled.shape[1] + 1):
    reg = LinearRegression()
    cv_scores = cross_val_score(
        reg, X_pca[:, :M], y,
        cv=kfold, scoring='neg_mean_squared_error'
    )
    pcr_mse_scores.append(-cv_scores.mean())

M_opt_pcr = np.argmin(pcr_mse_scores) + 1
pcr_cv_rmse = np.sqrt(pcr_mse_scores[M_opt_pcr - 1])
```

The scree plot of explained variance helps visualize how many components capture most of the
variability in $X$.

## Code: PLS with Cross-Validation

```python
from sklearn.cross_decomposition import PLSRegression

pls_mse_scores = []
for M in range(1, X_scaled.shape[1] + 1):
    pls = PLSRegression(n_components=M)
    cv_scores = cross_val_score(
        pls, X_scaled, y,
        cv=kfold, scoring='neg_mean_squared_error'
    )
    pls_mse_scores.append(-cv_scores.mean())

M_opt_pls = np.argmin(pls_mse_scores) + 1
pls_cv_rmse = np.sqrt(pls_mse_scores[M_opt_pls - 1])
```

## Code: Ridge Regression for Comparison

```python
from sklearn.linear_model import RidgeCV

ridge_cv = RidgeCV(alphas=np.logspace(-2, 5, 100), cv=10)
ridge_cv.fit(X_scaled, y)
ridge_r2 = r2_score(y, ridge_cv.predict(X_scaled))
ridge_rmse = np.sqrt(mean_squared_error(y, ridge_cv.predict(X_scaled)))
```

## Model Comparison

| Model | Hyperparameter | Key Property |
|---|---|---|
| OLS | None | Unbiased, highest variance |
| Ridge | $\lambda$ | Continuous shrinkage, all features retained |
| PCR | $M$ (components) | Unsupervised dimension reduction |
| PLS | $M$ (components) | Supervised dimension reduction |

Typical results on housing data show that PCR and PLS achieve $R^2$ values close to OLS with
fewer effective parameters. PLS often needs fewer components than PCR because it targets
directions relevant to the response.

## Interpretation

- **PCR** discards principal components that explain little variance in $X$, which may or may
  not be related to $y$. It is possible for a low-variance component to be highly predictive of
  the response, in which case PCR would miss it.
- **PLS** directly targets covariance with $y$, so it typically requires fewer components. This
  makes it particularly useful in chemometrics and spectroscopy where $p \gg n$.
- **Ridge** achieves similar results through continuous shrinkage rather than discrete component
  selection. It is computationally cheaper (closed-form solution).
- **Choosing between methods** depends on the goal: if interpretability in terms of latent
  components is important, use PCR or PLS; if prediction with all original features is
  preferred, use Ridge.

## Exercises

**Exercise 1.** Show that PCR with $M = p$ components is equivalent to OLS.

??? success "Solution to Exercise 1"

    The principal components are $Z = XV$, where $V$ is the $p \times p$ orthogonal matrix of
    eigenvectors of $X^\top X$. With $M = p$ components, PCR regresses $y$ on all columns of
    $Z$:

    $$
    \hat{\beta}^{\text{PCR}} = V (Z^\top Z)^{-1} Z^\top y.
    $$

    Since $Z = XV$ and $V$ is orthogonal ($V^\top V = I$):

    $$
    Z^\top Z = V^\top X^\top X V = D^2,
    $$

    where $D^2 = \text{diag}(d_1^2, \dots, d_p^2)$ contains the eigenvalues. Also
    $Z^\top y = V^\top X^\top y$. Therefore:

    $$
    \hat{\beta}^{\text{PCR}} = V D^{-2} V^\top X^\top y = (V D^2 V^\top)^{-1} X^\top y = (X^\top X)^{-1} X^\top y = \hat{\beta}^{\text{OLS}}.
    $$

    With all components retained, no information is discarded. $\square$

---

**Exercise 2.** Explain why standardizing the predictors is essential for PCR but not
strictly necessary for OLS. What goes wrong with PCR if predictors are not standardized?

??? success "Solution to Exercise 2"

    PCA finds directions of maximum variance. If predictors are on different scales (e.g.,
    square footage in thousands vs. number of bedrooms as single digits), the first principal
    components will be dominated by the high-variance (large-scale) predictors, regardless of
    their predictive importance.

    **Example:** If `SqFtLot` ranges from 1,000 to 500,000 and `Bedrooms` ranges from 1 to 6,
    the first PC will align almost entirely with `SqFtLot` simply because of its larger
    numerical variance. PCR would then base its regression on lot size and ignore bedroom count,
    even if bedroom count is more predictive.

    OLS does not have this problem because it directly minimizes the residual sum of squares
    without an intermediate variance-maximization step. The OLS coefficients automatically
    adjust to the scale of each predictor. (Standardization is still good practice for
    numerical stability, but it does not change the OLS fit.) $\square$

---

**Exercise 3.** In the housing example, suppose the optimal PCR uses $M = 7$ out of $p = 9$
components. What does this tell you about the data? Would you expect PLS to need more or fewer
components?

??? success "Solution to Exercise 3"

    If PCR needs 7 out of 9 components, it means that the last 2 principal components, despite
    explaining little variance in $X$, still contain information useful for predicting $y$.
    Dropping them slightly hurts prediction. This suggests the signal in the data is spread
    across many directions in predictor space, not concentrated in a low-dimensional subspace.

    PLS would likely need **fewer** components because it constructs directions that maximize
    covariance with $y$ rather than variance in $X$ alone. Even if a direction explains little
    variance in $X$, PLS will pick it up early if it is strongly associated with $y$. Empirically,
    PLS often requires 2--5 components to achieve comparable or better performance than PCR
    with 7 components, because it is directly optimizing for predictive relevance. $\square$

---

**Exercise 4.** Implement PCR from scratch (without using scikit-learn's PCA) using the SVD of
the centered and scaled design matrix. Verify that your implementation matches scikit-learn's
results on a small test dataset.

??? success "Solution to Exercise 4"

    ```python
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression

    np.random.seed(42)
    n, p = 50, 5
    X = np.random.randn(n, p)
    beta_true = np.array([3, -1, 2, 0, 0])
    y = X @ beta_true + np.random.randn(n) * 0.5

    # Standardize
    X_s = StandardScaler().fit_transform(X)

    # Manual PCR via SVD
    U, D, Vt = np.linalg.svd(X_s, full_matrices=False)
    M = 3  # number of components
    Z = U[:, :M] * D[:M]  # first M principal components
    gamma = np.linalg.lstsq(Z, y, rcond=None)[0]
    beta_pcr_manual = Vt[:M].T @ gamma
    y_pred_manual = X_s @ beta_pcr_manual

    # Scikit-learn PCR
    pca = PCA(n_components=M)
    Z_sk = pca.fit_transform(X_s)
    reg = LinearRegression().fit(Z_sk, y)
    y_pred_sk = reg.predict(Z_sk)

    print(f"Max prediction difference: {np.max(np.abs(y_pred_manual - y_pred_sk)):.2e}")
    ```

    The maximum prediction difference should be near machine epsilon (roughly $10^{-14}$),
    confirming the implementations are equivalent. $\square$

---

**Exercise 5.** Prove that the first PLS direction $w_1$ maximizes $\text{Cov}(Xw, y)^2$
subject to $\|w\| = 1$, and show that this is equivalent to $w_1 \propto X^\top y$.

??? success "Solution to Exercise 5"

    We want to find

    $$
    w_1 = \arg\max_{\|w\|=1} \left[\text{Cov}(Xw, y)\right]^2.
    $$

    Assuming $X$ and $y$ are centered, $\text{Cov}(Xw, y) = \frac{1}{n-1}(Xw)^\top y = \frac{1}{n-1}w^\top X^\top y$. Maximizing $[w^\top X^\top y]^2$ subject to $\|w\| = 1$ is
    equivalent to maximizing $|w^\top X^\top y|$ subject to $\|w\| = 1$.

    By the Cauchy--Schwarz inequality:

    $$
    |w^\top (X^\top y)| \le \|w\| \cdot \|X^\top y\| = \|X^\top y\|,
    $$

    with equality when $w \propto X^\top y$. Therefore

    $$
    w_1 = \frac{X^\top y}{\|X^\top y\|}.
    $$

    This shows that the first PLS direction is simply the (normalized) vector of marginal
    covariances between each predictor and the response. $\square$
