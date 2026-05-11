# 3D Regression Plane

## Overview

This page visualizes multiple linear regression with two predictors as a plane in three-dimensional space. Using synthetic advertising data (TV and Radio spending predicting Sales), we fit the regression plane, scatter the data points in 3D, and draw residual lines to illustrate the geometric interpretation of multiple regression.

## Mathematical Background

For two predictors, the multiple linear regression model is

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \varepsilon_i.
$$

The fitted values $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_{i1} + \hat{\beta}_2 x_{i2}$ define a **plane** in $(x_1, x_2, y)$-space. The residuals $e_i = y_i - \hat{y}_i$ are the vertical distances from data points to the plane.

The coefficient of determination measures the fraction of variance explained:

$$
R^2 = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}.
$$

Each coefficient has a **partial interpretation**: $\hat{\beta}_1$ measures the expected change in $y$ per unit increase in $x_1$, holding $x_2$ constant. Geometrically, $\hat{\beta}_1$ is the slope of the plane in the $x_1$-direction.

## Code

### Data Generation and Fitting

```python
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
n = 150
TV = np.random.uniform(0, 300, n)
Radio = np.random.uniform(0, 50, n)
Sales = 5 + 0.04 * TV + 0.15 * Radio + np.random.normal(0, 1.5, n)

X = np.column_stack([Radio, TV])
y = Sales

model = LinearRegression()
model.fit(X, y)

beta_0 = model.intercept_
beta_1 = model.coef_[0]  # Radio
beta_2 = model.coef_[1]  # TV
```

### Creating the Regression Plane Mesh

```python
Radio_range = np.arange(0, 50, 5)
TV_range = np.arange(0, 300, 30)
Radio_mesh, TV_mesh = np.meshgrid(Radio_range, TV_range)

Sales_mesh = beta_0 + beta_1 * Radio_mesh + beta_2 * TV_mesh
```

### 3D Visualization

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Regression plane
ax.plot_surface(Radio_mesh, TV_mesh, Sales_mesh,
                alpha=0.3, cmap='coolwarm')

# Data points
ax.scatter(Radio, TV, Sales, c='blue', s=50, alpha=0.6)

# Residual lines (every 5th point)
y_pred = model.predict(X)
for i in range(0, n, 5):
    ax.plot([X[i, 0], X[i, 0]], [X[i, 1], X[i, 1]],
            [y[i], y_pred[i]], 'r-', alpha=0.3)

ax.set_xlabel('Radio')
ax.set_ylabel('TV')
ax.set_zlabel('Sales')
plt.tight_layout()
plt.show()
```

## Interpretation

- **Regression plane**: The colored surface represents the model's prediction for every combination of TV and Radio spending. The plane's orientation reflects the relative magnitudes of the coefficients.
- **Data points**: Blue dots scattered around the plane. Points above the plane have positive residuals; points below have negative residuals.
- **Residual lines**: Red vertical segments connecting data points to the plane. OLS minimizes the sum of squared lengths of these segments.
- **Coefficient meaning**: If $\hat{\beta}_{\text{Radio}} = 0.15$ and $\hat{\beta}_{\text{TV}} = 0.04$, then a \$1,000 increase in Radio spending is associated with 0.15 more units of Sales (holding TV constant), while a \$1,000 increase in TV spending is associated with 0.04 more units.
- **Limitations**: With 3 or more predictors, the regression surface becomes a hyperplane that cannot be visualized directly. The 3D visualization is a pedagogical tool limited to two predictors.

## Exercises

**Exercise 1.** Add a third predictor (e.g., Newspaper spending) to the model. Explain why the resulting regression surface cannot be plotted in 3D and suggest alternatives for visualization.

??? success "Solution to Exercise 1"

    With three predictors, the regression surface is a hyperplane in 4D space ($x_1, x_2, x_3, y$), which cannot be rendered directly. Alternatives include: (1) partial regression plots (plot $y$ vs $x_j$ after removing the linear effect of other predictors), (2) slice plots (fix two predictors at their means and plot $y$ vs the third), (3) added-variable plots, or (4) coefficient plots showing point estimates with confidence intervals. $\square$

---

**Exercise 2.** Compute the $R^2$ value manually from RSS and TSS. Verify it matches `model.score(X, y)`.

??? success "Solution to Exercise 2"

    ```python
    y_pred = model.predict(X)
    RSS = np.sum((y - y_pred) ** 2)
    TSS = np.sum((y - y.mean()) ** 2)
    R2_manual = 1 - RSS / TSS
    R2_sklearn = model.score(X, y)
    print(f"Manual R2: {R2_manual:.4f}")
    print(f"Sklearn R2: {R2_sklearn:.4f}")
    print(f"Match: {np.isclose(R2_manual, R2_sklearn)}")
    ```

    Both values are identical by definition. $\square$

---

**Exercise 3.** Rotate the 3D plot to different viewing angles. From which angle do the residuals appear smallest? Explain geometrically.

??? success "Solution to Exercise 3"

    When viewing the plot along the normal vector to the regression plane, the plane appears edge-on (as a line), and residuals are maximally visible. When viewing perpendicular to the normal (i.e., looking along the plane itself), residuals appear smallest because the plane faces the viewer. The angle minimizing apparent residual size is the one where the line of sight is parallel to the plane, which collapses the residual direction. $\square$

---

**Exercise 4.** Prove that the OLS residuals satisfy $\sum_{i=1}^n e_i = 0$ and $\sum_{i=1}^n x_{ij} e_i = 0$ for each predictor $j$ (when an intercept is included).

??? success "Solution to Exercise 4"

    The normal equations are $\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = \mathbf{0}$, i.e., $\mathbf{X}^\top\mathbf{e} = \mathbf{0}$. Since the first column of $\mathbf{X}$ is $\mathbf{1}$ (the intercept column), the first equation gives $\mathbf{1}^\top\mathbf{e} = \sum e_i = 0$. The $(j+1)$-th equation gives $\mathbf{x}_j^\top\mathbf{e} = \sum x_{ij}e_i = 0$. These orthogonality conditions are fundamental properties of OLS. $\square$

---

**Exercise 5.** In the model $\text{Sales} = \beta_0 + \beta_1 \cdot \text{Radio} + \beta_2 \cdot \text{TV} + \varepsilon$, explain the difference between $\beta_1$ (the partial coefficient) and the coefficient obtained from the simple regression of Sales on Radio alone.

??? success "Solution to Exercise 5"

    The simple regression coefficient $\tilde{\beta}_1$ captures the total association between Radio and Sales, including any indirect association through TV (e.g., if companies that spend more on Radio also spend more on TV). The partial coefficient $\hat{\beta}_1$ isolates the direct effect of Radio after removing the linear influence of TV from both Radio and Sales. Formally, $\hat{\beta}_1$ equals the slope from regressing the residuals of Sales on TV against the residuals of Radio on TV (the Frisch-Waugh-Lovell theorem). The two coefficients coincide only when Radio and TV are uncorrelated. $\square$
