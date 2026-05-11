# Splines with Patsy

## Overview

This page demonstrates spline regression using the patsy formula interface for constructing B-spline and natural spline basis matrices. Using the King County housing dataset, we compare B-splines, natural splines with various degrees of freedom, custom knot placements, and contrast these flexible methods with linear and polynomial regression.

## Mathematical Background

### B-Splines (Basis Splines)

A B-spline of degree $d$ with knots $\xi_1 < \cdots < \xi_K$ is a piecewise polynomial that is $(d-1)$-times continuously differentiable at each knot. The regression model is

$$
f(x) = \sum_{m=1}^{K+d+1} \gamma_m B_{m,d}(x),
$$

where $B_{m,d}$ are the B-spline basis functions. The number of basis functions is $K + d + 1$ (interior knots + degree + 1).

### Natural Splines

Natural cubic splines add the constraint that $f(x)$ is linear beyond the boundary knots. This reduces the degrees of freedom by 4 (two constraints at each boundary: $f'' = 0$ and $f''' = 0$), producing more stable extrapolation behavior.

### Degrees of Freedom

The degrees of freedom (df) of a spline controls its flexibility:

- For B-splines: $\mathrm{df} = K + d + 1 - 1$ (number of basis functions minus intercept)
- Higher df allows more wiggly fits
- The optimal df can be chosen via cross-validation

### Patsy Syntax

- `bs(x, df=4, degree=3)`: B-spline with 4 df and cubic degree
- `bs(x, knots=[20, 40, 60])`: B-spline with specified interior knots
- `cr(x, df=4)`: natural cubic regression spline with 4 df

## Code

### B-Spline Regression

```python
import numpy as np
import pandas as pd
from patsy import dmatrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

age = 2024 - house['YrBuilt'].values
df = pd.DataFrame({'age': age, 'price': house['AdjSalePrice'].values})

# B-spline with df=4
bs_design = dmatrix("bs(age, df=4, degree=3, include_intercept=False) - 1",
                    {"age": df['age']}, return_type='dataframe')
bs_model = LinearRegression().fit(bs_design, df['price'])
bs_r2 = r2_score(df['price'], bs_model.predict(bs_design))
```

### B-Spline with Custom Knots

```python
knots_custom = [20, 40, 60]
bs_custom_design = dmatrix(
    f"bs(age, knots={knots_custom}, degree=3, include_intercept=False) - 1",
    {"age": df['age']}, return_type='dataframe'
)
bs_custom_model = LinearRegression().fit(bs_custom_design, df['price'])
```

### Natural Splines

```python
cs_design = dmatrix("cr(age, df=4) - 1",
                    {"age": df['age']}, return_type='dataframe')
cs_model = LinearRegression().fit(cs_design, df['price'])
cs_r2 = r2_score(df['price'], cs_model.predict(cs_design))
```

### Prediction on a Grid

```python
age_grid = np.linspace(df['age'].min(), df['age'].max(), 300)

# B-spline predictions
bs_grid = dmatrix("bs(age, df=4, degree=3, include_intercept=False) - 1",
                  {"age": age_grid}, return_type='dataframe')
bs_pred = bs_model.predict(bs_grid)

# Natural spline predictions
cs_grid = dmatrix("cr(age, df=4) - 1",
                  {"age": age_grid}, return_type='dataframe')
cs_pred = cs_model.predict(cs_grid)
```

## Interpretation

- **B-splines** provide local control: each basis function is nonzero only over a small range, so the fit at one region is relatively independent of distant data. This makes B-splines numerically stable and interpretable.
- **Natural splines** are more stable at the boundaries because they constrain the fit to be linear beyond the extreme knots. This avoids the wild extrapolation behavior that unconstrained cubic splines exhibit.
- **Degrees of freedom**: Increasing df improves fit on training data but risks overfitting. The optimal df can be selected via cross-validation or information criteria.
- **Knot placement**: Automatic placement (uniform quantiles) works well in most cases. Custom knots are useful when domain knowledge suggests where the relationship changes character.
- **Comparison**: Splines typically outperform polynomials of the same complexity because they provide local flexibility rather than imposing a single global shape.

## Exercises

**Exercise 1.** Fit B-splines with df ranging from 2 to 8 and plot the resulting curves. At what df does the curve begin to show signs of overfitting?

??? success "Solution to Exercise 1"

    ```python
    for d in range(2, 9):
        design = dmatrix(f"bs(age, df={d}, degree=3, include_intercept=False) - 1",
                         {"age": df['age']}, return_type='dataframe')
        model = LinearRegression().fit(design, df['price'])
        grid_design = dmatrix(f"bs(age, df={d}, degree=3, include_intercept=False) - 1",
                              {"age": age_grid}, return_type='dataframe')
        plt.plot(age_grid, model.predict(grid_design), label=f'df={d}')
    ```

    Overfitting manifests as excessive wiggliness, particularly near the boundaries where data is sparse. Typically df > 6 begins to show overfitting for this dataset. $\square$

---

**Exercise 2.** Explain why a natural spline with $K$ interior knots has $K + 1$ degrees of freedom (not $K + 4$ as for an unconstrained cubic spline). Where do the 3 "lost" degrees of freedom go?

??? success "Solution to Exercise 2"

    An unconstrained cubic spline with $K$ interior knots has $K + 4$ basis functions ($K + 3 + 1$, accounting for the cubic polynomial in each region minus continuity constraints). Natural splines impose linearity beyond the boundary knots, adding 4 constraints (second and third derivatives equal zero at each boundary). However, the net reduction is $K + 4 - 3 = K + 1$ effective parameters because 3 of the 4 boundary constraints remove 3 degrees of freedom. The "lost" degrees of freedom correspond to the cubic and quadratic terms at the boundaries that are set to zero. $\square$

---

**Exercise 3.** Compare the predictions of B-spline and natural spline models at ages beyond the data range (e.g., age = 150 or age = 0). Which extrapolates more sensibly?

??? success "Solution to Exercise 3"

    ```python
    age_extrap = np.array([0, 5, 145, 150])
    bs_extrap = dmatrix(f"bs(age, df=4, degree=3, include_intercept=False) - 1",
                        {"age": age_extrap}, return_type='dataframe')
    cs_extrap = dmatrix("cr(age, df=4) - 1",
                        {"age": age_extrap}, return_type='dataframe')
    print("B-spline:", bs_model.predict(bs_extrap))
    print("Natural:", cs_model.predict(cs_extrap))
    ```

    Natural splines extrapolate linearly (using the slope at the boundary), which is generally more reasonable than the cubic extrapolation of B-splines, which can produce extreme and unrealistic predictions. $\square$

---

**Exercise 4.** Implement a roughness penalty for B-splines by adding a ridge-like term $\lambda \|\boldsymbol{\gamma}\|^2$ to the fitting objective. How does this relate to smoothing splines?

??? success "Solution to Exercise 4"

    ```python
    from sklearn.linear_model import Ridge

    design = dmatrix("bs(age, df=8, degree=3, include_intercept=False) - 1",
                     {"age": df['age']}, return_type='dataframe')
    ridge_model = Ridge(alpha=1e4).fit(design, df['price'])
    ```

    Adding $\lambda\|\boldsymbol{\gamma}\|^2$ shrinks the basis coefficients toward zero, producing a smoother curve. This is an approximation to the smoothing spline penalty $\lambda\int [f'']^2$, which penalizes the second derivative. The exact smoothing spline uses a penalty matrix $\mathbf{D}$ where $D_{jk} = \int B_j''(t)B_k''(t)\,dt$, but $\lambda\mathbf{I}$ (ridge) provides a simpler approximation. $\square$

---

**Exercise 5.** Prove that for a natural cubic spline with knots at the data points, the smoothing spline estimator minimizes $\sum(y_i - f(x_i))^2 + \lambda\int[f''(t)]^2\,dt$ over all twice-differentiable functions.

??? success "Solution to Exercise 5"

    This is a classical result in nonparametric regression. The key insight is that among all functions $g$ that interpolate given values at the knots, the natural cubic spline has the smallest $\int [g'']^2\,dt$ (the roughness). This follows from a variational argument: write $g = f + h$ where $f$ is the natural cubic spline. Then

    $$
    \int [g'']^2 = \int [f'']^2 + 2\int f'' h'' + \int [h'']^2.
    $$

    The cross term $\int f'' h'' = 0$ by integration by parts and the boundary conditions of the natural spline, so $\int[g'']^2 \geq \int[f'']^2$. Therefore the natural cubic spline is the unique minimizer of the penalized objective for any $\lambda > 0$. $\square$
