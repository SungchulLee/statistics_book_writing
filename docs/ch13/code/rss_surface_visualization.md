# Residual Sum of Squares Surface Visualization

## Overview

This page creates 3D visualizations of the Residual Sum of Squares (RSS) as a function of the regression coefficients $\beta_0$ (intercept) and $\beta_1$ (slope). These visualizations provide geometric intuition for why OLS produces unique optimal estimates and how the RSS surface relates to the convex optimization problem solved by least squares.

## Mathematical Background

For the simple linear regression model $y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$, the RSS is a function of the coefficients:

$$
\mathrm{RSS}(\beta_0, \beta_1) = \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2.
$$

Expanding this expression reveals that RSS is a **quadratic function** (a paraboloid) in $(\beta_0, \beta_1)$:

$$
\mathrm{RSS}(\beta_0, \beta_1) = n\beta_0^2 + \beta_1^2 \sum x_i^2 + 2\beta_0\beta_1\sum x_i - 2\beta_0\sum y_i - 2\beta_1\sum x_i y_i + \sum y_i^2.
$$

The Hessian matrix of RSS is

$$
\mathbf{H} = 2\mathbf{X}^\top\mathbf{X} = 2\begin{pmatrix} n & \sum x_i \\ \sum x_i & \sum x_i^2 \end{pmatrix},
$$

which is positive definite (assuming the $x_i$ are not all equal), guaranteeing that the RSS surface is **strictly convex** with a unique global minimum.

## Code

### Data Generation and Model Fitting

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_samples = 100
TV = np.random.uniform(0, 300, n_samples)
Sales = 7 + 0.05 * TV + np.random.normal(0, 2, n_samples)

X = TV.reshape(-1, 1)
scaler = StandardScaler(with_mean=True, with_std=False)
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, Sales)
beta_0 = model.intercept_
beta_1 = model.coef_[0]
```

### Computing the RSS Surface

```python
B0_range = np.linspace(beta_0 - 2, beta_0 + 2, 50)
B1_range = np.linspace(beta_1 - 0.05, beta_1 + 0.05, 50)
B0_mesh, B1_mesh = np.meshgrid(B0_range, B1_range)

RSS = np.zeros_like(B0_mesh)
for i in range(B0_mesh.shape[0]):
    for j in range(B0_mesh.shape[1]):
        y_pred = B0_mesh[i, j] + B1_mesh[i, j] * X_scaled
        RSS[i, j] = np.sum((Sales - y_pred) ** 2)
```

### Visualization

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 6))

# Contour plot
ax1 = fig.add_subplot(121)
contour = ax1.contour(B0_mesh, B1_mesh, RSS / 1000, levels=20, cmap='viridis')
ax1.plot(beta_0, beta_1, 'r*', markersize=20, label='Optimal')
ax1.set_xlabel('beta_0 (Intercept)')
ax1.set_ylabel('beta_1 (Slope)')
ax1.set_title('RSS Contour Plot')

# 3D surface
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(B0_mesh, B1_mesh, RSS / 1000, cmap='viridis', alpha=0.8)
ax2.set_xlabel('beta_0')
ax2.set_ylabel('beta_1')
ax2.set_zlabel('RSS / 1000')
ax2.set_title('RSS 3D Surface')

plt.tight_layout()
plt.show()
```

## Interpretation

- **Convexity**: The RSS surface is bowl-shaped (a paraboloid) with a single global minimum. This means gradient descent from any starting point will converge to the OLS solution.
- **Contour shape**: The elliptical contours reflect the correlation structure of the predictors. When predictors are uncorrelated (after centering), the contours are aligned with the axes; when correlated, they tilt.
- **Sensitivity**: Tightly packed contour lines indicate the RSS changes rapidly in that direction, meaning the corresponding coefficient is well-determined. Widely spaced contours indicate poor identifiability.
- **Optimal point**: The red star marks $(\hat{\beta}_0, \hat{\beta}_1)$, the OLS solution where $\nabla \mathrm{RSS} = \mathbf{0}$.

## Exercises

**Exercise 1.** Verify analytically that the gradient of RSS is zero at the OLS solution by computing $\partial \mathrm{RSS}/\partial \beta_0$ and $\partial \mathrm{RSS}/\partial \beta_1$ and setting them to zero.

??? success "Solution to Exercise 1"

    $$
    \frac{\partial \mathrm{RSS}}{\partial \beta_0} = -2\sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i) = 0 \implies n\hat{\beta}_0 + \hat{\beta}_1\sum x_i = \sum y_i.
    $$

    $$
    \frac{\partial \mathrm{RSS}}{\partial \beta_1} = -2\sum_{i=1}^n x_i(y_i - \beta_0 - \beta_1 x_i) = 0 \implies \hat{\beta}_0\sum x_i + \hat{\beta}_1\sum x_i^2 = \sum x_i y_i.
    $$

    These are the normal equations $\mathbf{X}^\top\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^\top\mathbf{y}$, confirming the gradient is zero at the OLS solution. $\square$

---

**Exercise 2.** Show that the Hessian $\mathbf{H} = 2\mathbf{X}^\top\mathbf{X}$ is positive definite when the $x_i$ are not all equal. What happens when they are all equal?

??? success "Solution to Exercise 2"

    The Hessian is $2\mathbf{X}^\top\mathbf{X}$ where $\mathbf{X} = [\mathbf{1} \mid \mathbf{x}]$. This is positive definite if and only if $\mathbf{X}$ has full column rank 2. If all $x_i$ are equal, the second column of $\mathbf{X}$ is a constant times the first column, so $\mathbf{X}$ is rank 1 and $\mathbf{X}^\top\mathbf{X}$ is singular. The RSS surface degenerates: the minimum lies along a line rather than at a unique point, reflecting the fact that $\beta_0$ and $\beta_1$ are not separately identifiable. $\square$

---

**Exercise 3.** Generate a dataset with a higher noise level ($\sigma = 10$) and recreate the surface. How does the shape change compared to $\sigma = 2$?

??? success "Solution to Exercise 3"

    With higher noise, the minimum RSS value increases (the bowl is higher), but the shape of the surface and the location of the minimum remain qualitatively similar. The contours expand because the RSS values are larger everywhere. The OLS estimates remain at the minimum but have larger standard errors, meaning the "valley" around the minimum is wider and shallower relative to the total RSS. $\square$

---

**Exercise 4.** Implement gradient descent to find the minimum of the RSS surface. Compare the number of iterations needed with different learning rates.

??? success "Solution to Exercise 4"

    ```python
    lr = 0.0001
    beta = np.array([0.0, 0.0])  # initial guess
    for step in range(1000):
        residuals = Sales - beta[0] - beta[1] * X_scaled.flatten()
        grad = np.array([-2 * np.sum(residuals),
                         -2 * np.sum(residuals * X_scaled.flatten())])
        beta -= lr * grad
    print(f"GD solution: beta_0={beta[0]:.4f}, beta_1={beta[1]:.4f}")
    ```

    Smaller learning rates require more iterations but converge reliably. Larger learning rates converge faster but risk overshooting. The optimal rate depends on the eigenvalues of $\mathbf{X}^\top\mathbf{X}$. $\square$

---

**Exercise 5.** Explain why the contour ellipses are aligned with the coordinate axes when $\bar{x} = 0$ (centered predictors), and tilt when $\bar{x} \neq 0$.

??? success "Solution to Exercise 5"

    The contour shape is determined by $\mathbf{X}^\top\mathbf{X}$. When predictors are centered ($\bar{x} = 0$), the off-diagonal element $\sum x_i = n\bar{x} = 0$, making $\mathbf{X}^\top\mathbf{X}$ diagonal. Diagonal matrices produce axis-aligned ellipses. When $\bar{x} \neq 0$, the off-diagonal is nonzero, introducing correlation between $\beta_0$ and $\beta_1$ and tilting the ellipses. Centering predictors orthogonalizes the intercept and slope estimates, simplifying both visualization and numerical computation. $\square$
