# Ridge Regression (L2 Regularization)


## Motivation: The Problem with OLS

Ordinary least squares (OLS) minimizes the residual sum of squares:

$$
\hat{\boldsymbol{\beta}}_{\text{OLS}} = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2
$$

with the closed-form solution $\hat{\boldsymbol{\beta}}_{\text{OLS}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}$.

This estimator is **unbiased** and, by the Gauss–Markov theorem, has the minimum variance among all linear unbiased estimators (BLUE). However, OLS can perform poorly when:

1. **Multicollinearity.** When predictors are highly correlated, $\mathbf{X}^\top\mathbf{X}$ is nearly singular. Small changes in the data cause large changes in $\hat{\boldsymbol{\beta}}$, inflating variance.

2. **High-dimensional settings.** When $p$ (number of predictors) is close to or exceeds $n$ (number of observations), OLS is ill-defined or severely overfit.

3. **Prediction accuracy.** The bias–variance tradeoff (Chapter 6) tells us that introducing a small amount of bias can substantially reduce variance, improving MSE and prediction accuracy.

## Ridge Regression Formulation

**Ridge regression** (Hoerl and Kennard, 1970) adds an L2 penalty to the OLS objective:

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|^2 \right\}
$$

where $\lambda \geq 0$ is the **regularization parameter** (also called the **tuning parameter** or **penalty strength**), and $\|\boldsymbol{\beta}\|^2 = \sum_{j=1}^p \beta_j^2$.

!!! note "Centering and Scaling"
    The intercept $\beta_0$ is typically **not penalized**. In practice, we center $\mathbf{y}$ and standardize each column of $\mathbf{X}$ to have mean 0 and unit variance before applying ridge regression, so that the penalty treats all coefficients equally.

## Closed-Form Solution

The ridge estimator has a closed-form solution:

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}
$$

*Derivation.* Take the gradient of the objective and set to zero:

$$
\frac{\partial}{\partial \boldsymbol{\beta}}\left[\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda\|\boldsymbol{\beta}\|^2\right] = -2\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\beta}) + 2\lambda\boldsymbol{\beta} = \mathbf{0}
$$

$$
(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})\boldsymbol{\beta} = \mathbf{X}^\top\mathbf{y}
$$

Since $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$ is positive definite for $\lambda > 0$ (even when $\mathbf{X}^\top\mathbf{X}$ is singular), the solution always exists and is unique.

## Geometric Interpretation

Ridge regression can be equivalently formulated as a **constrained optimization**:

$$
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 \quad \text{subject to} \quad \|\boldsymbol{\beta}\|^2 \leq t
$$

where $t$ is determined by $\lambda$ through the KKT conditions. Geometrically:

- The OLS objective defines elliptical contours in coefficient space
- The constraint $\|\boldsymbol{\beta}\|^2 \leq t$ is a **sphere** (in $p$ dimensions, a hypersphere)
- The ridge solution is where the smallest elliptical contour touches the sphere

This sphere constraint shrinks all coefficients toward zero but typically does **not** set any exactly to zero.

## SVD Interpretation

Using the singular value decomposition $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$ (where $d_1 \geq d_2 \geq \cdots \geq d_p \geq 0$ are singular values):

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda} \cdot \frac{\mathbf{u}_j^\top \mathbf{y}}{d_j}\,\mathbf{v}_j
$$

Compare with OLS:

$$
\hat{\boldsymbol{\beta}}_{\text{OLS}} = \sum_{j=1}^p \frac{\mathbf{u}_j^\top \mathbf{y}}{d_j}\,\mathbf{v}_j
$$

The factor $\frac{d_j^2}{d_j^2 + \lambda}$ is a **shrinkage factor** between 0 and 1. Components with small singular values (the unstable directions) are shrunk the most. This is precisely where OLS has high variance, so ridge regression stabilizes the estimate.

## Bias and Variance of Ridge

$$
\text{Bias}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = -\lambda(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\boldsymbol{\beta}
$$

$$
\text{Cov}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = \sigma^2 (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}
$$

As $\lambda$ increases:

- Bias increases (coefficients are pulled further from their OLS values)
- Variance decreases (the inverse is better conditioned)
- There exists an optimal $\lambda^*$ that minimizes total MSE

**Theorem (Hoerl and Kennard, 1970).** There always exists a $\lambda > 0$ such that $\text{MSE}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) < \text{MSE}(\hat{\boldsymbol{\beta}}_{\text{OLS}})$.

## Choosing lambda: Cross-Validation
The optimal $\lambda$ is unknown and must be estimated from data. The standard approach is **$k$-fold cross-validation**:

1. Divide data into $k$ folds (typically $k = 5$ or $10$)
2. For each candidate $\lambda$ in a grid:
    - For each fold $j$: train on $k-1$ folds, predict on fold $j$, record prediction error
    - Average the prediction error across folds: $\text{CV}(\lambda) = \frac{1}{k}\sum_{j=1}^k \text{MSE}_j(\lambda)$
3. Select $\hat{\lambda} = \arg\min_\lambda \text{CV}(\lambda)$

!!! tip "One-Standard-Error Rule"
    Instead of choosing the $\lambda$ with minimum CV error, select the largest $\lambda$ within one standard error of the minimum. This gives a more parsimonious (simpler) model with similar predictive performance.

**Leave-one-out CV for ridge** has a closed-form shortcut:

$$
\text{CV}_{\text{LOO}}(\lambda) = \frac{1}{n}\sum_{i=1}^n \left(\frac{y_i - \hat{y}_i(\lambda)}{1 - h_{ii}(\lambda)}\right)^2
$$

where $h_{ii}(\lambda) = [\mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top]_{ii}$ is the $i$-th diagonal of the hat matrix.

## Ridge Regression as Bayesian MAP

Ridge regression is equivalent to the **maximum a posteriori** (MAP) estimate in a Bayesian linear model with a Gaussian prior:

$$
\boldsymbol{\beta} \sim N(\mathbf{0}, \tau^2 \mathbf{I}), \quad \mathbf{y} \mid \boldsymbol{\beta} \sim N(\mathbf{X}\boldsymbol{\beta}, \sigma^2\mathbf{I})
$$

The posterior mode is:

$$
\hat{\boldsymbol{\beta}}_{\text{MAP}} = (\mathbf{X}^\top\mathbf{X} + \frac{\sigma^2}{\tau^2}\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}
$$

Setting $\lambda = \sigma^2/\tau^2$ recovers the ridge solution. Larger $\lambda$ corresponds to a stronger prior belief that $\boldsymbol{\beta}$ is close to zero.

## Effective Degrees of Freedom

In OLS, the degrees of freedom equal the number of parameters $p$. In ridge regression, the **effective degrees of freedom** are:

$$
\text{df}(\lambda) = \text{tr}\left[\mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\right] = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}
$$

As $\lambda \to 0$, $\text{df} \to p$ (OLS). As $\lambda \to \infty$, $\text{df} \to 0$ (constant model). This allows direct comparison of model complexity across different $\lambda$ values.

## Key Properties Summary

| Property | Value |
|---|---|
| Penalty | $\lambda\sum_{j=1}^p \beta_j^2$ (L2) |
| Solution | Closed-form: $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| Feature selection | No (shrinks but doesn't zero out) |
| Handles multicollinearity | Yes |
| Works when $p > n$ | Yes |
| Bayesian interpretation | Gaussian prior on $\boldsymbol{\beta}$ |
| Geometric constraint | $\ell_2$-ball (sphere) |


## Exercises

**Exercise 1.**
Write the ridge regression optimization problem and its closed-form solution. How does $\lambda$ affect the solution?

??? success "Solution to Exercise 1"
    The ridge optimization problem is:

    $$
    \hat{\boldsymbol{\beta}}_{\text{ridge}} = \arg\min_{\boldsymbol{\beta}} \left\{ \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert^2 + \lambda \lVert \boldsymbol{\beta} \rVert^2 \right\}
    $$

    The closed-form solution is:

    $$
    \hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}
    $$

    As $\lambda \to 0$, $\hat{\boldsymbol{\beta}}_{\text{ridge}} \to \hat{\boldsymbol{\beta}}_{\text{OLS}}$. As $\lambda \to \infty$, $\hat{\boldsymbol{\beta}}_{\text{ridge}} \to \mathbf{0}$. The penalty $\lambda$ controls the bias-variance tradeoff: larger $\lambda$ increases bias but reduces variance.

---

**Exercise 2.**
Show that ridge regression is biased. Derive the bias in terms of $\lambda$ and the true $\boldsymbol{\beta}$.

??? success "Solution to Exercise 2"
    $$
    E[\hat{\boldsymbol{\beta}}_{\text{ridge}}] = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T E[\mathbf{y}] = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}
    $$

    $$
    = \mathbf{W}\boldsymbol{\beta} \neq \boldsymbol{\beta}
    $$

    where $\mathbf{W} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{X}$. The bias is:

    $$
    \text{Bias} = (\mathbf{W} - \mathbf{I})\boldsymbol{\beta} = -\lambda(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\boldsymbol{\beta}
    $$

    The bias increases with $\lambda$ and with the magnitude of $\boldsymbol{\beta}$. Despite this bias, the total MSE can be smaller than OLS because the variance reduction exceeds the squared bias. $\square$

---

**Exercise 3.**
Explain why ridge regression helps with multicollinearity. What happens to the condition number of $\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I}$?

??? success "Solution to Exercise 3"
    Multicollinearity means $\mathbf{X}^T\mathbf{X}$ has eigenvalues near zero, making it nearly singular with a large condition number $\kappa = \lambda_{\max}/\lambda_{\min}$.

    Adding $\lambda\mathbf{I}$ shifts all eigenvalues by $\lambda$: if the eigenvalues of $\mathbf{X}^T\mathbf{X}$ are $d_1 \geq \dots \geq d_p$, the eigenvalues of $\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I}$ are $d_1 + \lambda \geq \dots \geq d_p + \lambda$. The new condition number is:

    $$
    \kappa_{\text{ridge}} = \frac{d_1 + \lambda}{d_p + \lambda} < \frac{d_1}{d_p} = \kappa_{\text{OLS}}
    $$

    The smallest eigenvalue increases from $d_p$ (near zero) to $d_p + \lambda$, dramatically improving numerical stability and reducing the variance of $\hat{\boldsymbol{\beta}}$.

---

**Exercise 4.**
A ridge regression with $\lambda = 1$ on standardized data gives $\hat{\beta}_1 = 0.45$ and $\hat{\beta}_2 = 0.38$. OLS gives $\hat{\beta}_1 = 1.2$ and $\hat{\beta}_2 = -0.8$. Interpret the difference.

??? success "Solution to Exercise 4"
    The OLS estimates ($1.2$ and $-0.8$) are large in magnitude and have opposite signs, which is characteristic of multicollinearity: the predictors are correlated, and OLS produces unstable, inflated coefficients that partially cancel each other.

    Ridge regression ($0.45$ and $0.38$) shrinks both coefficients toward zero and produces more similar magnitudes. The sign reversal of $\hat{\beta}_2$ (from $-0.8$ to $+0.38$) suggests the OLS negative coefficient was an artifact of collinearity, not a genuine negative relationship. Ridge produces more stable, interpretable estimates at the cost of some bias.
