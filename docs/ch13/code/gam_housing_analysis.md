# Generalized Additive Model Housing Analysis

## Overview

This page demonstrates Generalized Additive Models (GAMs) for house price prediction using the King County housing dataset. We compare linear regression, polynomial regression, and GAMs (via both statsmodels and pyGAM), visualize partial dependence plots, and assess model performance through RMSE and $R^2$ metrics.

## Mathematical Background

A Generalized Additive Model extends linear regression by allowing each predictor to have a nonlinear effect through smooth functions:

$$
y = \beta_0 + f_1(x_1) + f_2(x_2) + \cdots + f_p(x_p) + \varepsilon,
$$

where each $f_j$ is a smooth function estimated from the data (typically using splines). The model is "additive" because each predictor contributes independently to the response.

### Smoothing Splines

Each $f_j$ is represented using a basis expansion, commonly B-splines:

$$
f_j(x_j) = \sum_{m=1}^{M_j} \gamma_{jm}\, B_{jm}(x_j),
$$

where $B_{jm}$ are basis functions. The smoothness is controlled by a **smoothing parameter** $\lambda_j$, and the objective becomes

$$
\min_{\gamma} \sum_{i=1}^n \!\left(y_i - \beta_0 - \sum_{j=1}^p f_j(x_{ij})\right)^{\!2} + \sum_{j=1}^p \lambda_j \int [f_j''(t)]^2\, dt.
$$

The penalty $\lambda_j \int [f_j'']^2\,dt$ controls the wiggliness of $f_j$: larger $\lambda_j$ produces smoother curves.

### Partial Dependence

The **partial dependence** of $y$ on $x_j$ is the function $f_j(x_j)$, which shows the marginal effect of predictor $j$ on the response while averaging over the other predictors.

## Code

### Linear and Polynomial Models

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade']
outcome = 'AdjSalePrice'

# Linear model
X_linear = house_98105[predictors].assign(const=1)
result_linear = sm.OLS(house_98105[outcome], X_linear).fit()

# Polynomial model
formula_poly = ('AdjSalePrice ~ SqFtTotLiving + np.power(SqFtTotLiving, 2) + '
                'SqFtLot + Bathrooms + Bedrooms + BldgGrade')
result_poly = smf.ols(formula=formula_poly, data=house_98105).fit()
```

### GAM with statsmodels

```python
from statsmodels.gam.api import GLMGam, BSplines

x_spline = house_98105[predictors]
bs = BSplines(x_spline, df=[10, 3, 3, 3, 3], degree=[3, 2, 2, 2, 2])
alpha = np.array([0] * 5)

gam_sm = GLMGam.from_formula(
    'AdjSalePrice ~ SqFtTotLiving + SqFtLot + Bathrooms + Bedrooms + BldgGrade',
    data=house_98105, smoother=bs, alpha=alpha
)
res_sm = gam_sm.fit()
```

### GAM with pyGAM

```python
from pygam import LinearGAM, s, l

X_gam = house_98105[predictors].values
y_gam = house_98105[outcome].values

gam_py = LinearGAM(
    s(0, n_splines=12) +  # SqFtTotLiving: smooth
    l(1) +                 # SqFtLot: linear
    l(2) +                 # Bathrooms: linear
    l(3) +                 # Bedrooms: linear
    l(4)                   # BldgGrade: linear
)
gam_py.gridsearch(X_gam, y_gam)
```

## Interpretation

- **Linear model**: Assumes each predictor has a constant marginal effect on price. Simple but may miss curvature in the relationship between living area and price.
- **Polynomial model**: Captures curvature in $\text{SqFtTotLiving}$ via a quadratic term but imposes a global shape (the polynomial applies everywhere).
- **GAM**: Allows the effect of $\text{SqFtTotLiving}$ to vary freely through a smooth spline while keeping other predictors linear. This flexibility typically improves fit.
- **Partial dependence plots** reveal the shape of each $f_j$. A nearly linear partial dependence suggests the linear term is adequate; curvature justifies the smooth term.
- **Effective degrees of freedom** (EDF) measure the complexity of each smooth term. Higher EDF means more wiggliness and greater risk of overfitting.

## Exercises

**Exercise 1.** Compare the $R^2$ and RMSE of the linear, polynomial, and GAM models. Which model performs best, and is the improvement substantial?

??? success "Solution to Exercise 1"

    ```python
    from sklearn.metrics import mean_squared_error, r2_score

    models = {
        'Linear': result_linear.fittedvalues,
        'Polynomial': result_poly.fittedvalues,
        'GAM (pyGAM)': gam_py.predict(X_gam),
    }
    for name, pred in models.items():
        rmse = np.sqrt(mean_squared_error(y_gam, pred))
        r2 = r2_score(y_gam, pred)
        print(f"{name}: R2={r2:.4f}, RMSE={rmse:.0f}")
    ```

    The GAM typically shows modest improvement over the linear model and comparable performance to the polynomial. The improvement depends on how nonlinear the true relationship is. $\square$

---

**Exercise 2.** Modify the pyGAM specification to use smooth splines for all predictors instead of linear terms. Does this improve the fit? Discuss the risk of overfitting.

??? success "Solution to Exercise 2"

    ```python
    gam_all_smooth = LinearGAM(
        s(0) + s(1) + s(2) + s(3) + s(4)
    )
    gam_all_smooth.gridsearch(X_gam, y_gam)
    ```

    Using smooth terms for all predictors increases flexibility and typically improves training $R^2$, but may overfit. The effective degrees of freedom increase, and the model may capture noise rather than signal. Cross-validation should be used to assess whether the additional flexibility improves out-of-sample prediction. $\square$

---

**Exercise 3.** Explain why GAMs are called "additive." What assumption does the additive structure impose, and when might it be violated?

??? success "Solution to Exercise 3"

    GAMs assume the response is the sum of individual smooth functions: $y = \beta_0 + f_1(x_1) + \cdots + f_p(x_p) + \varepsilon$. This means the effect of each predictor is independent of the values of other predictors (no interactions). This assumption is violated when, for example, the effect of living area on price depends on the building grade (an interaction effect). Extensions such as tensor product smooths can handle interactions. $\square$

---

**Exercise 4.** The smoothing parameter $\lambda$ controls the bias-variance tradeoff. Explain what happens as $\lambda \to 0$ and $\lambda \to \infty$.

??? success "Solution to Exercise 4"

    As $\lambda \to 0$, the penalty vanishes and $f_j$ interpolates the data (high variance, low bias, overfitting). As $\lambda \to \infty$, the penalty forces $f_j'' \equiv 0$, meaning $f_j$ must be linear (low variance, potentially high bias, underfitting). The optimal $\lambda$ balances these extremes. In pyGAM, `gridsearch` selects $\lambda$ by minimizing generalized cross-validation (GCV) or a similar criterion. $\square$

---

**Exercise 5.** Derive the matrix representation of the penalized least squares problem for a single-predictor GAM with B-spline basis. Show that the solution involves $(B^\top B + \lambda D)^{-1} B^\top y$ where $B$ is the basis matrix and $D$ is a penalty matrix.

??? success "Solution to Exercise 5"

    Let $f(x) = \sum_{m=1}^M \gamma_m B_m(x)$, so $\mathbf{f} = \mathbf{B}\boldsymbol{\gamma}$ where $\mathbf{B}$ is the $n \times M$ basis matrix. The roughness penalty is $\int [f'']^2\,dt = \boldsymbol{\gamma}^\top\mathbf{D}\boldsymbol{\gamma}$ where $D_{jk} = \int B_j''(t) B_k''(t)\,dt$. The penalized objective is

    $$
    (\mathbf{y} - \mathbf{B}\boldsymbol{\gamma})^\top(\mathbf{y} - \mathbf{B}\boldsymbol{\gamma}) + \lambda\,\boldsymbol{\gamma}^\top\mathbf{D}\boldsymbol{\gamma}.
    $$

    Differentiating with respect to $\boldsymbol{\gamma}$ and setting to zero:

    $$
    -2\mathbf{B}^\top(\mathbf{y} - \mathbf{B}\boldsymbol{\gamma}) + 2\lambda\mathbf{D}\boldsymbol{\gamma} = \mathbf{0} \implies \hat{\boldsymbol{\gamma}} = (\mathbf{B}^\top\mathbf{B} + \lambda\mathbf{D})^{-1}\mathbf{B}^\top\mathbf{y}.
    $$

    This is a ridge-like regression on the basis coefficients, where $\lambda\mathbf{D}$ acts as a structured penalty. $\square$
